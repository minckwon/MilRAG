"""
RAG pipeline for Korean military QA dataset.

Pipeline
--------
1. Embed all article contents in article/combined.json with BGE-M3 → FAISS index
   (saved as milrag.index + milrag_meta.json for reuse)
2. For each QA record in qa_dataset.jsonl:
   a. Embed the question → retrieve top-3 article docs
   b. Check retrieval hit (ground-truth URL in top-3)
   c. Build context from retrieved contents → generate answer with EXAONE-4.5-33B
3. Write per-record results to rag_results.jsonl

Output schema per line
----------------------
{
  "question":            str,
  "ground_truth_answer": str,
  "generated_answer":    str,
  "metadata":            {url, title, category, date},
  "retrieval": {
    "retrieved_docs": [{content, metadata, score, rank}, ...],
    "hit":            bool,    # ground-truth URL found in top-3
    "hit_rank":       int|null
  }
}
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT      = Path(__file__).parent
QA_DATASET     = REPO_ROOT / "qa_dataset.jsonl"
CORPUS_PATH    = REPO_ROOT / "article" / "combined.json"
INDEX_PATH     = REPO_ROOT / "milrag.index"
META_PATH      = REPO_ROOT / "milrag_meta.json"
RESULTS_PATH   = REPO_ROOT / "eval" / "rag_results.jsonl"

EMBED_MODEL_ID   = "BAAI/bge-m3"
GEN_MODEL_ID     = "LGAI-EXAONE/EXAONE-4.0-32B"
TOP_K            = 3
BATCH_SIZE       = 1

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "당신은 군사 무기 및 방산 분야의 전문 어시스턴트입니다. "
    "주어진 참고 문서를 바탕으로 질문에 정확하고 간결하게 답변하세요. "
    "문서에 없는 내용은 추측하지 마세요."
)


def build_user_prompt(context: str, question: str) -> str:
    return (
        f"[참고 문서]\n{context}\n\n"
        f"[질문]\n{question}\n\n"
        "[답변]"
    )


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def build_index(corpus: list[dict], embed_model: SentenceTransformer) -> faiss.IndexFlatIP:
    docs = [r["content"] for r in corpus]
    print(f"\n[index] Encoding {len(docs)} documents ...")
    t0 = time.time()
    embeddings = embed_model.encode(
        docs,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    ).astype(np.float32)
    print(f"[index] Done in {time.time() - t0:.1f}s")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"[index] Saved → {INDEX_PATH}")
    return index


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open(encoding="utf-8") as f:
        records = json.load(f)
    print(f"[index] Loaded {index.ntotal} vectors from {INDEX_PATH}")
    return index, records


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    questions: list[str],
    embed_model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    index_records: list[dict],
) -> list[list[dict]]:
    q_embs = embed_model.encode(
        questions,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    ).astype(np.float32)

    scores, indices = index.search(q_embs, TOP_K)

    results = []
    for q_scores, q_indices in zip(scores, indices):
        hits = []
        for rank, (idx, score) in enumerate(zip(q_indices, q_scores), start=1):
            rec = index_records[idx]
            hits.append({
                "content":  rec["content"],
                "metadata": {
                    "url":      rec["url"],
                    "title":    rec["title"],
                    "category": rec["category"],
                    "date":     rec["date"],
                },
                "score":    float(score),
                "rank":     rank,
            })
        results.append(hits)
    return results


def check_hit(qa_record: dict, retrieved: list[dict]) -> tuple[bool, int | None]:
    source_url = qa_record["metadata"]["url"]
    for hit in retrieved:
        if hit["metadata"]["url"] == source_url:
            return True, hit["rank"]
    return False, None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_llm() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"\n[llm] Loading {GEN_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
    gt_answers: list[str] | None = None,
) -> list[str]:
    outputs = []
    for start in tqdm(range(0, len(prompts), BATCH_SIZE), desc="[gen] generating", unit="batch"):
        chunk = prompts[start : start + BATCH_SIZE]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        for i, gen_ids in enumerate(generated_ids):
            input_len = inputs["input_ids"].shape[1]
            new_ids = gen_ids[input_len:]
            answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            outputs.append(answer)
            idx = start + i
            print(f"\n{'─' * 60}")
            print(f"[{idx + 1}/{len(prompts)}]")
            if gt_answers is not None:
                print(f"  GT : {gt_answers[idx]}")
            print(f"  GEN: {answer}")
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load QA dataset (questions + ground-truth answers)
    print(f"\n[data] Loading {QA_DATASET} ...")
    qa_records: list[dict] = []
    with QA_DATASET.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_records.append(json.loads(line))
    print(f"[data] {len(qa_records)} QA records loaded")

    # 2. Embedding model (CPU to preserve GPU memory for generation)
    print(f"\n[embed] Loading {EMBED_MODEL_ID} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_ID, device="cpu")

    # 3. FAISS index — rebuild from corpus if missing, reuse if present
    if INDEX_PATH.exists():
        index, corpus = load_index()
    else:
        print(f"\n[corpus] Loading {CORPUS_PATH} ...")
        with CORPUS_PATH.open(encoding="utf-8") as f:
            corpus = json.load(f)
        print(f"[corpus] {len(corpus)} articles loaded")
        index = build_index(corpus, embed_model)

    # 4. Retrieve
    print(f"\n[retrieve] Retrieving top-{TOP_K} docs for {len(qa_records)} questions ...")
    all_retrieved = retrieve([r["question"] for r in qa_records], embed_model, index, corpus)

    retrieval_results = []
    for qa_record, retrieved in zip(qa_records, all_retrieved):
        hit, hit_rank = check_hit(qa_record, retrieved)
        retrieval_results.append({
            "retrieved_docs": retrieved,
            "hit":            hit,
            "hit_rank":       hit_rank,
        })

    hit_count = sum(r["hit"] for r in retrieval_results)
    print(f"[retrieve] hit@{TOP_K}: {hit_count}/{len(qa_records)} ({hit_count/len(qa_records)*100:.1f}%)")

    # 5. Generate
    model, tokenizer = load_llm()

    print(f"\n[gen] Building prompts ...")
    prompts: list[str] = []
    for qa_record, ret in zip(qa_records, retrieval_results):
        context = "\n\n".join(
            f"[문서 {d['rank']}] {d['content']}"
            for d in ret["retrieved_docs"]
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(context, qa_record["question"])},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    print(f"[gen] Generating {len(prompts)} answers ...")
    generated = batch_generate(model, tokenizer, prompts, gt_answers=[r["answer"] for r in qa_records])

    # 6. Write results
    print(f"\n[output] Writing {RESULTS_PATH} ...")
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for qa_record, ret, gen_ans in zip(qa_records, retrieval_results, generated):
            f.write(json.dumps({
                "question":            qa_record["question"],
                "ground_truth_answer": qa_record["answer"],
                "generated_answer":    gen_ans,
                "metadata":            qa_record["metadata"],
                "retrieval":           ret,
            }, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 50}")
    print(f"  QA records  : {len(qa_records)}")
    print(f"  Corpus docs : {len(corpus)}")
    print(f"  hit@{TOP_K}       : {hit_count}/{len(qa_records)} ({hit_count/len(qa_records)*100:.1f}%)")
    print(f"  Output      : {RESULTS_PATH}")


if __name__ == "__main__":
    main()