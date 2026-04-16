# MilRAG — Korean Military Domain RAG Benchmark

국방 도메인 문서를 기반으로 **RAG(Retrieval-Augmented Generation) 시스템의 성능을 평가**하기 위한 파이프라인입니다.  
국방일보 무기백과 및 국방부 뉴스 기사를 크롤링하고, QA 데이터셋을 생성한 뒤, 여러 LLM의 RAG / No-RAG 성능을 비교합니다.

---

## Repository Structure

```
milrag/
├── crawling/
│   ├── crawl_weapon.py              # 국방일보 무기백과 크롤러
│   └── crawl_mnd.py                 # 국방부 국방일보 뉴스 크롤러
│
├── article/
│   ├── combined.json                # 전체 기사 통합본 (FAISS 인덱스 입력)
│   ├── weapon/                      # 무기백과 카테고리별 기사
│   │   ├── aircraft.json/csv        # 항공기 (49)
│   │   ├── command_communication.json/csv  # 지휘통신 (20)
│   │   ├── defense_robot.json/csv   # 국방로봇 (50)
│   │   ├── firepower.json/csv       # 화력 (93)
│   │   ├── guided_munition.json/csv # 유도무기 (32)
│   │   ├── maneuver.json/csv        # 기동 (54)
│   │   ├── naval_vessel.json/csv    # 함정 (31)
│   │   ├── protection.json/csv      # 방호 (42)
│   │   ├── RnD.json/csv             # 연구개발 (80)
│   │   └── surveillance_reconnaissance.json/csv  # 감시정찰 (21)
│   └── mnd/
│       └── mnd_news.json/csv        # 국방부 뉴스 (174)
│
├── generate_qa.py                   # QA 데이터셋 생성기 (Qwen2.5-32B, vLLM)
├── qa_dataset.jsonl                 # 생성된 QA 쌍 (636건)
│
├── milrag.index                     # FAISS 벡터 인덱스 (BGE-M3 임베딩)
├── milrag_meta.json                 # FAISS 인덱스 메타데이터
│
├── script/
│   ├── exaone_rag.py                # RAG 추론 — LGAI-EXAONE/EXAONE-4.0-32B
│   ├── exaone_no_rag.py             # No-RAG 추론 — LGAI-EXAONE/EXAONE-4.0-32B
│   ├── naver_rag.py                 # RAG 추론 — naver-hyperclovax/HyperCLOVAX-SEED-Think-32B
│   └── naver_no_rag.py              # No-RAG 추론 — naver-hyperclovax/HyperCLOVAX-SEED-Think-32B
│
└── eval/
    ├── judge.py                     # LLM-as-Judge 평가 스크립트 (GPT, 0~5점)
    ├── exaone_rag_results.jsonl     # EXAONE RAG 추론 결과
    ├── exaone_no_rag_results.jsonl  # EXAONE No-RAG 추론 결과
    ├── naver_rag_results.jsonl      # HyperCLOVAX RAG 추론 결과
    └── naver_no_rag_results.jsonl   # HyperCLOVAX No-RAG 추론 결과
```

---

## Dataset Statistics

### Weapon Encyclopedia (`article/weapon/`)

Source: [국방일보 무기백과](https://kookbang.dema.mil.kr/newsWeb/ATCE_CTGR_0080010000/list.do)

| Category | Description | Articles |
|---|---|---:|
| `aircraft` | 항공기 | 49 |
| `command_communication` | 지휘통신 | 20 |
| `defense_robot` | 국방로봇 | 50 |
| `firepower` | 화력 | 93 |
| `guided_munition` | 유도무기 | 32 |
| `maneuver` | 기동 | 54 |
| `naval_vessel` | 함정 | 31 |
| `protection` | 방호 | 42 |
| `RnD` | 연구개발 | 80 |
| `surveillance_reconnaissance` | 감시정찰 | 21 |
| **Total** | | **472** |

### MND News (`article/mnd/`)

Source: [국방부 국방일보](https://www.mnd.go.kr/cop/kookbang/kookbangIlboList.do)

| File | Articles |
|---|---:|
| `mnd_news.json` | 174 |

### Corpus (`article/combined.json`)

무기백과 + MND 뉴스 전체 통합본. FAISS 인덱스 구축에 사용됩니다.

| File | Articles |
|---|---:|
| `combined.json` | 638 |

### QA Dataset (`qa_dataset.jsonl`)

| Generator Model | QA Pairs |
|---|---:|
| Qwen/Qwen2.5-32B-Instruct | 636 |

---

## Article Schema

```json
{
  "url": "https://kookbang.dema.mil.kr/newsWeb/...",
  "category": "aircraft",
  "title": "P-8A 해상초계기",
  "date": "2023-03-14",
  "content": "P-8 포세이돈(Poseidon) 해상초계기는 ..."
}
```

---

## QA Dataset Schema

```json
{
  "question": "FFG 충남함의 복합센서마스트(ISM)에는 어떤 장비들이 설치되어 있습니까?",
  "answer": "복합센서마스트에는 다기능 위상배열레이더(MFR) 4대와 적외선탐지추적장비(IRST) 4대가 각 면에 통합 설치되어 있습니다.",
  "metadata": {
    "url": "https://kookbang.dema.mil.kr/newsWeb/...",
    "title": "FFG 충남함 마스트, 기존 함정과 어떻게 다를까",
    "category": "RnD",
    "date": ""
  }
}
```

---

## Pipeline

### 1. Crawling

```bash
python crawling/crawl_weapon.py   # 무기백과 크롤링 → article/weapon/
python crawling/crawl_mnd.py      # 국방부 뉴스 크롤링 → article/mnd/
```

### 2. QA Dataset Generation

```bash
python generate_qa.py                     # 전체 실행
python generate_qa.py --max-docs 5        # 테스트 (5개 문서)
python generate_qa.py --qa-per-doc 2      # 문서당 QA 쌍 수 조정
python generate_qa.py --resume            # 중단된 실행 재개
```

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | `article/weapon/` | 입력 JSON 디렉토리 |
| `--output` | `qa_dataset.jsonl` | 출력 JSONL 경로 |
| `--qa-per-doc` | `1` | 문서당 QA 쌍 수 |
| `--tensor-parallel` | `2` | GPU 수 |
| `--batch-size` | `16` | vLLM 배치 크기 |
| `--max-docs` | — | 최대 문서 수 (테스트용) |
| `--resume` | — | 이미 처리된 URL 건너뜀 |

### 3. Inference

RAG와 No-RAG 두 가지 설정으로 각 모델을 실행합니다.  
RAG는 **BGE-M3** 임베딩 + **FAISS** 인덱스로 top-3 문서를 검색한 뒤 LLM에 전달합니다.

```bash
# EXAONE-4.0-32B
python script/exaone_rag.py        # RAG    → eval/exaone_rag_results.jsonl
python script/exaone_no_rag.py     # No-RAG → eval/exaone_no_rag_results.jsonl

# HyperCLOVAX-SEED-Think-32B
python script/naver_rag.py         # RAG    → eval/naver_rag_results.jsonl
python script/naver_no_rag.py      # No-RAG → eval/naver_no_rag_results.jsonl
```

#### Inference Result Schema

**RAG** (`*_rag_results.jsonl`):
```json
{
  "question":            "질문",
  "ground_truth_answer": "정답",
  "generated_answer":    "모델 생성 답변",
  "metadata":            {"url": "...", "title": "...", "category": "...", "date": "..."},
  "retrieval": {
    "retrieved_docs": [{"content": "...", "metadata": {...}, "score": 0.91, "rank": 1}],
    "hit":      true,
    "hit_rank": 1
  }
}
```

**No-RAG** (`*_no_rag_results.jsonl`):
```json
{
  "question":            "질문",
  "ground_truth_answer": "정답",
  "generated_answer":    "모델 생성 답변",
  "metadata":            {"url": "...", "title": "...", "category": "...", "date": "..."}
}
```

### 4. Evaluation (LLM-as-Judge)

GPT를 judge로 사용하여 생성 품질을 평가합니다.

```bash
# 평가할 파일을 judge.py 상단 FILE_PATH 변수에 지정 후 실행
OPENAI_API_KEY=sk-... python eval/judge.py
```

| 평가 항목 | 적용 | 척도 |
|---|---|---|
| Generation 품질 | RAG / No-RAG 모두 | 0 ~ 5점 (GPT judge) |
| Retrieval 성공 여부 | RAG만 | 0 또는 1 (hit@3 기반) |

`eval/judge.py` 상단에서 수정 가능한 설정값:

```python
FILE_PATH    = Path(__file__).parent / "exaone_rag_results.jsonl"  # 평가 대상 파일
NUM_RECORDS  = 1      # 평가 문서 수 (전체: None)
MODEL        = "gpt-5.4-mini"
```

---

## Models

| Model | Type | Parameters |
|---|---|---|
| [LGAI-EXAONE/EXAONE-4.0-32B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B) | General LLM | 32B |
| [naver-hyperclovax/HyperCLOVAX-SEED-Think-32B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B) | Thinking LLM | 32B |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Embedding | — |
| [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | QA 생성용 | 32B |

---

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU (80GB+ VRAM 권장, 40GB 최소)
- conda

### Install

```bash
conda activate roka
pip install transformers torch faiss-gpu sentence-transformers tqdm
pip install openai          # eval/judge.py 사용 시
pip install vllm            # generate_qa.py 사용 시
```

---

## Data Source

| Source | URL |
|---|---|
| 국방일보 무기백과 | https://kookbang.dema.mil.kr/newsWeb/ATCE_CTGR_0080010000/list.do |
| 국방부 국방일보 | https://www.mnd.go.kr/cop/kookbang/kookbangIlboList.do |
