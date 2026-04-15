"""
QA Dataset Generator for RAG system — local GPU inference via vLLM.

Reads articles from article/weapon/*.json and article/mnd/*.json and generates
exactly one question-answer pair per document using Qwen2.5-32B-Instruct
running locally on your GPUs.

Usage:
    python generate_qa.py [--input-dirs PATH ...] [--output PATH] [--max-docs N]
    python generate_qa.py --max-docs 5               # smoke-test on 5 docs
    python generate_qa.py --resume                   # continue interrupted run
    python generate_qa.py --tensor-parallel 1        # single GPU (default: 2)
"""

import argparse
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_INPUT_DIRS = [
    Path(__file__).parent / "article" / "weapon",
    Path(__file__).parent / "article" / "mnd",
]
DEFAULT_OUTPUT = Path(__file__).parent / "qa_dataset.jsonl"
MAX_CONTENT_CHARS = 4000   # truncate very long articles to stay within context
BATCH_SIZE = 16            # number of prompts to submit per vLLM call


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "당신은 군사 무기 및 방산 분야의 전문가입니다. "
    "주어진 문서를 읽고, RAG(Retrieval-Augmented Generation) 시스템 평가에 "
    "사용할 질문-답변 쌍을 생성합니다.\n\n"
    "규칙:\n"
    "1. 질문은 문서 내용에 기반해야 하며, 문서를 보지 않고는 답하기 어려운 구체적인 질문이어야 합니다.\n"
    "2. 답변은 문서에서 직접 도출할 수 있어야 하며, 간결하고 정확해야 합니다.\n"
    "3. 다양한 유형의 질문을 생성하세요: 사실 확인, 수치/사양, 운용 국가, 개발 배경, 주요 특징 등.\n"
    "4. 반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.\n\n"
    "출력 형식:\n"
    "[\n"
    "  {\"question\": \"...\", \"answer\": \"...\"},\n"
    "  ...\n"
    "]"
)


def build_chat_messages(article: dict, qa_count: int) -> list[dict]:
    content = article["content"]
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "..."

    category = article.get("category", "")
    category_line = f"[카테고리]: {category}\n" if category else ""
    user_text = (
        f"다음 문서를 읽고 {qa_count}개의 질문-답변 쌍을 생성하세요.\n\n"
        f"[문서 제목]: {article.get('title', '')}\n"
        f"{category_line}"
        f"[날짜]: {article.get('date') or '미상'}\n\n"
        f"[문서 내용]:\n{content}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> list[dict]:
    """Extract the first JSON array from the model response."""
    text = text.strip()

    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Greedy: find outermost [...]
    start = text.find("[")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start : i + 1])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break

    return []


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_articles(input_dirs: list[Path]) -> list[dict]:
    articles = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"  [warn] Directory not found, skipping: {input_dir}")
            continue
        for json_file in sorted(input_dir.glob("*.json")):
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                articles.extend(data)
                print(f"  Loaded {len(data):>4} articles from {input_dir.name}/{json_file.name}")
    return articles


def load_done_urls(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(obj["metadata"]["url"])
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate QA dataset from weapon/mnd articles (local vLLM)")
    parser.add_argument("--input-dirs", type=Path, nargs="+", default=DEFAULT_INPUT_DIRS,
                        help="Directories containing *.json article files (default: article/weapon, article/mnd)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL file path")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limit to first N documents (useful for testing)")
    parser.add_argument("--tensor-parallel", type=int, default=2,
                        help="Number of GPUs for tensor parallelism (default: 2)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip documents whose URLs are already in the output file")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Prompts per vLLM batch (default: {BATCH_SIZE})")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load articles
    # ------------------------------------------------------------------
    print(f"\nLoading articles from: {[str(d) for d in args.input_dirs]}")
    articles = load_articles(args.input_dirs)
    print(f"Total articles loaded: {len(articles)}")

    if args.max_docs:
        articles = articles[: args.max_docs]
        print(f"Limited to first {args.max_docs} documents.")

    # Filter empty content
    articles = [a for a in articles if a.get("content", "").strip()]

    # Resume: skip already-processed URLs
    done_urls: set[str] = set()
    if args.resume:
        done_urls = load_done_urls(args.output)
        before = len(articles)
        articles = [a for a in articles if a.get("url", "") not in done_urls]
        print(f"Resuming — skipped {before - len(articles)} already-processed documents. "
              f"{len(articles)} remaining.")

    if not articles:
        print("Nothing to process. Exiting.")
        return

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\nLoading {MODEL_ID} with tensor_parallel_size={args.tensor_parallel} ...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=1024,
    )

    # Retrieve the tokenizer's chat template for apply_chat_template
    tokenizer = llm.get_tokenizer()

    # ------------------------------------------------------------------
    # Build all prompts
    # ------------------------------------------------------------------
    print(f"\nBuilding prompts for {len(articles)} articles (1 QA pair each)...")
    prompts: list[str] = []
    for article in articles:
        messages = build_chat_messages(article, 1)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------
    print(f"Running inference in batches of {args.batch_size}...\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    total_qa = 0
    total_skipped = 0

    with args.output.open(mode, encoding="utf-8") as out_f:
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_articles = articles[batch_start:batch_end]

            print(f"Batch {batch_start // args.batch_size + 1} | "
                  f"docs {batch_start + 1}–{batch_end} / {len(articles)}")

            outputs = llm.generate(batch_prompts, sampling_params)

            for article, output in zip(batch_articles, outputs):
                raw = output.outputs[0].text
                pairs = extract_json(raw)

                title = article.get("title", "(제목 없음)")
                url = article.get("url", "")

                if not pairs:
                    print(f"  [warn] No JSON parsed for: {title[:50]}")
                    print(f"         Raw output: {raw[:150]!r}")
                    total_skipped += 1
                    continue

                # Enforce exactly 1 QA pair per document
                pair = pairs[0]
                record = {
                    "question": pair.get("question", ""),
                    "answer": pair.get("answer", ""),
                    "metadata": {
                        "url": url,
                        "title": title,
                        "category": article.get("category", ""),
                        "date": article.get("date", ""),
                    },
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_qa += 1
                print(f"  OK  1 pair | {title[:50]}")

            out_f.flush()

    print(f"\n{'=' * 60}")
    print(f"Done.")
    print(f"  Documents processed : {len(articles) - total_skipped}")
    print(f"  Documents skipped   : {total_skipped}")
    print(f"  Total QA pairs      : {total_qa}")
    print(f"  Output file         : {args.output}")


if __name__ == "__main__":
    main()