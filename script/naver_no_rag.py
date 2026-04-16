"""
No-RAG baseline QA pipeline for Korean military QA dataset.

Pipeline
--------
1. Load QA records from qa_dataset.jsonl
2. For each question, generate an answer using HyperCLOVAX-SEED-Think-32B
   with NO retrieval — relying solely on the model's internal knowledge
3. Write per-record results to eval/naver_no_rag_results.jsonl

Output schema per line
----------------------
{
  "question":            str,
  "ground_truth_answer": str,
  "generated_answer":    str,   # <think>...</think> 블록 제거된 최종 답변
  "metadata":            {url, title, category, date}
}
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import contextlib
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils as _modeling_utils

# transformers 5.x 에서 no_init_weights 가 제거됨 — 하위 호환 패치
if not hasattr(_modeling_utils, "no_init_weights"):
    @contextlib.contextmanager
    def _no_init_weights(_enable: bool = True):
        yield
    _modeling_utils.no_init_weights = _no_init_weights

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT    = Path(__file__).parent.parent
QA_DATASET   = REPO_ROOT / "qa_dataset.jsonl"
RESULTS_PATH = REPO_ROOT / "eval" / "naver_no_rag_results.jsonl"

GEN_MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"
BATCH_SIZE   = 1

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "당신은 군사 무기 및 방산 분야의 전문 어시스턴트입니다. "
    "보유한 지식을 바탕으로 질문에 정확하고 간결하게 답변하세요."
)


def build_user_prompt(question: str) -> str:
    return f"[질문]\n{question}\n\n[답변]"


# ---------------------------------------------------------------------------
# Thinking token 제거
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """<think>...</think> 블록을 제거하고 최종 답변만 반환."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def load_llm() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"\n[llm] Loading {GEN_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        GEN_MODEL_ID, trust_remote_code=True, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
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
            raw = tokenizer.decode(new_ids, skip_special_tokens=True)
            answer = strip_thinking(raw)
            outputs.append(answer)
            idx = start + i
            print(f"\n{'─' * 60}")
            print(f"[{idx + 1}/{len(prompts)}]")
            if gt_answers is not None:
                print(f"  GT  : {gt_answers[idx]}")
            print(f"  RAW : {raw[:200]}{'...' if len(raw) > 200 else ''}")
            print(f"  GEN : {answer}")
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load QA dataset
    print(f"\n[data] Loading {QA_DATASET} ...")
    qa_records: list[dict] = []
    with QA_DATASET.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_records.append(json.loads(line))
    print(f"[data] {len(qa_records)} QA records loaded")

    # 2. Load LLM
    model, tokenizer = load_llm()

    # 3. Build prompts (question only — no retrieved context)
    print(f"\n[gen] Building prompts ...")
    prompts: list[str] = []
    for qa_record in qa_records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(qa_record["question"])},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    # 4. Generate
    print(f"[gen] Generating {len(prompts)} answers ...")
    generated = batch_generate(
        model, tokenizer, prompts,
        gt_answers=[r["answer"] for r in qa_records],
    )

    # 5. Write results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[output] Writing {RESULTS_PATH} ...")
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for qa_record, gen_ans in zip(qa_records, generated):
            f.write(json.dumps({
                "question":            qa_record["question"],
                "ground_truth_answer": qa_record["answer"],
                "generated_answer":    gen_ans,
                "metadata":            qa_record["metadata"],
            }, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 50}")
    print(f"  QA records : {len(qa_records)}")
    print(f"  Output     : {RESULTS_PATH}")


if __name__ == "__main__":
    main()
