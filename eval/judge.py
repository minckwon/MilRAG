"""
LLM-as-Judge evaluation script using GPT.

- No-RAG mode : generation 성능만 평가 (0~5점)
- RAG mode    : generation 성능 (0~5점) + retrieval 성공 여부 (0 또는 1)

Usage
-----
  python eval/judge.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# ★ 직접 수정하는 설정값
# ---------------------------------------------------------------------------

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "OPENAI_API_KEY_REMOVED")   # 또는 직접 문자열 입력
MODEL           = "gpt-5.4-mini"

FILE_PATH       = Path(__file__).parent / "exaone_rag_results.jsonl"

NUM_RECORDS     = None          # 평가할 문서 수 (전체 평가 시 None으로 변경)

# ---------------------------------------------------------------------------
# 내부 설정
# ---------------------------------------------------------------------------

RETRY_WAIT  = 5
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_JUDGE_SYSTEM = """\
당신은 군사 무기 및 방산 분야 QA 평가 전문가입니다.
아래 채점 기준에 따라 생성된 답변을 0~5점으로 평가하세요.

[채점 기준]
5점: 정답과 완전히 일치하며, 핵심 정보가 누락 없이 정확하게 포함됨
4점: 대부분 정확하고, 사소한 표현 차이나 누락만 존재함
3점: 핵심 내용은 맞지만 일부 오류 또는 중요한 정보 누락이 있음
2점: 부분적으로 맞으나 중요한 오류 또는 다수의 누락이 존재함
1점: 관련은 있으나 대부분 틀리거나 핵심을 벗어난 답변
0점: 완전히 틀리거나 질문과 무관한 답변

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
{"score": <0~5 정수>, "reason": "<한 문장 이유>"}"""

ANSWER_JUDGE_USER = """\
[질문]
{question}

[정답]
{ground_truth}

[생성된 답변]
{generated}"""

# ---------------------------------------------------------------------------
# GPT call
# ---------------------------------------------------------------------------

def call_gpt(client: OpenAI, system: str, user: str) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  [retry {attempt}/{MAX_RETRIES}] {e} — waiting {RETRY_WAIT}s ...")
                time.sleep(RETRY_WAIT)
            else:
                raise


def evaluate_answer(client: OpenAI, record: dict) -> dict:
    result = call_gpt(
        client,
        ANSWER_JUDGE_SYSTEM,
        ANSWER_JUDGE_USER.format(
            question=record["question"],
            ground_truth=record["ground_truth_answer"],
            generated=record["generated_answer"],
        ),
    )
    return {"score": int(result["score"]), "reason": result["reason"]}


# ---------------------------------------------------------------------------
# Evaluate one file
# ---------------------------------------------------------------------------

def run(client: OpenAI, path: Path) -> None:
    if not path.exists():
        print(f"[skip] 파일 없음: {path}\n")
        return

    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if NUM_RECORDS is not None:
        records = records[:NUM_RECORDS]

    has_retrieval = "retrieval" in records[0] if records else False
    mode = "RAG" if has_retrieval else "No-RAG"

    print(f"\n{'=' * 60}")
    print(f"  파일   : {path.name}")
    print(f"  모드   : {mode}")
    print(f"  평가수 : {len(records)}")
    print(f"{'=' * 60}\n")

    answer_scores: list[int] = []
    retrieval_hits: list[int] = []      # 0 또는 1 (RAG 전용)

    for i, record in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] Q: {record['question']}")

        # Generation 평가
        ans_eval = evaluate_answer(client, record)
        answer_scores.append(ans_eval["score"])
        print(f"  GT  : {record['ground_truth_answer']}")
        print(f"  GEN : {record['generated_answer']}")
        print(f"  → Generation Score: {ans_eval['score']}/5  ({ans_eval['reason']})")

        # Retrieval 평가 (RAG 전용 — hit 여부를 0/1로 기록)
        if has_retrieval:
            hit = int(record["retrieval"]["hit"])          # True→1, False→0
            retrieval_hits.append(hit)
            hit_rank = record["retrieval"]["hit_rank"]
            hit_str = f"1 (rank {hit_rank})" if hit else "0 (miss)"
            print(f"  → Retrieval Hit  : {hit_str}")

        print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"{'─' * 60}")
    print(f"  SUMMARY — {mode} ({path.name})")
    print(f"{'─' * 60}")

    avg_ans = sum(answer_scores) / len(answer_scores) if answer_scores else 0.0
    dist = {s: answer_scores.count(s) for s in range(6)}
    print(f"  [Generation]")
    print(f"  평균 점수     : {avg_ans:.2f} / 5.00")
    print(f"  점수 분포     : " + "  ".join(f"{k}점:{v}건" for k, v in dist.items()))

    if retrieval_hits:
        hit_rate = sum(retrieval_hits) / len(retrieval_hits) * 100
        top_k = len(records[0]["retrieval"]["retrieved_docs"])
        print(f"\n  [Retrieval]")
        print(f"  hit@{top_k}         : {sum(retrieval_hits)}/{len(retrieval_hits)} ({hit_rate:.1f}%)")
        print(f"  (각 문서별 0/1)  : {retrieval_hits}")

    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not OPENAI_API_KEY:
        sys.exit("[error] OPENAI_API_KEY가 설정되지 않았습니다.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    run(client, FILE_PATH)


if __name__ == "__main__":
    main()
