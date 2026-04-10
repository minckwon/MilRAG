# MilRAG — Korean Military Document RAG Dataset

A dataset pipeline for building a Retrieval-Augmented Generation (RAG) system on Korean military documents.  
Articles are crawled from [국방일보 (Korea Defense Daily)](https://kookbang.dema.mil.kr) and the [Ministry of National Defense (MND)](https://www.mnd.go.kr), then used to generate QA pairs via **Qwen2.5-32B-Instruct** running locally on GPU.

---

## Repository Structure

```
milrag/
├── crawling/
│   ├── crawl_weapon.py          # Crawler for 국방일보 무기백과 (Weapons Encyclopedia)
│   └── crawl_mnd.py             # Crawler for MND 국방일보 news articles
├── article/
│   ├── weapon/                  # Weapon encyclopedia articles (by category)
│   │   ├── aircraft.json/csv
│   │   ├── command_communication.json/csv
│   │   ├── defense_robot.json/csv
│   │   ├── firepower.json/csv
│   │   ├── guided_munition.json/csv
│   │   ├── maneuver.json/csv
│   │   ├── naval_vessel.json/csv
│   │   ├── protection.json/csv
│   │   ├── RnD.json/csv
│   │   └── surveillance_reconnaissance.json/csv
│   └── mnd/
│       └── mnd_news.json/csv    # MND news articles (latest 300)
├── generate_qa.py               # QA dataset generator (local vLLM inference)
└── qa_dataset.jsonl             # Generated QA pairs (output)
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
| `mnd_news.json` | 300 |

### QA Dataset (`qa_dataset.jsonl`)

| Model | QA Pairs |
|---|---:|
| Qwen/Qwen2.5-32B-Instruct | 1,237 |

---

## Article Schema

Each article JSON file is a list of objects with the following fields:

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

Each line in `qa_dataset.jsonl` is a QA pair with a reference to its source article:

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

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with 40GB+ VRAM (80GB recommended for Qwen2.5-32B)
- conda

### Create environment

```bash
conda activate roka
pip install vllm
```

### Crawling

```bash
# Crawl weapon encyclopedia articles
python crawling/crawl_weapon.py

# Crawl MND news articles
python crawling/crawl_mnd.py
```

Outputs are saved under `article/weapon/` and `article/mnd/` as both `.json` and `.csv`.

### Generate QA dataset

```bash
# Full run — all 472 weapon articles, 5 QA pairs each
python generate_qa.py

# Test on 5 documents first
python generate_qa.py --max-docs 5

# Adjust QA pairs per document
python generate_qa.py --qa-per-doc 10

# Resume an interrupted run
python generate_qa.py --resume

# Single GPU
python generate_qa.py --tensor-parallel 1
```

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | `article/weapon/` | Source JSON directory |
| `--output` | `qa_dataset.jsonl` | Output JSONL path |
| `--qa-per-doc` | `1` | QA pairs per article |
| `--tensor-parallel` | `2` | Number of GPUs |
| `--batch-size` | `16` | Prompts per vLLM batch |
| `--max-docs` | — | Limit documents (for testing) |
| `--resume` | — | Skip already-processed URLs |

---

## Data Source

| Source | URL |
|---|---|
| 국방일보 무기백과 | https://kookbang.dema.mil.kr/newsWeb/ATCE_CTGR_0080010000/list.do |
| 국방부 국방일보 | https://www.mnd.go.kr/cop/kookbang/kookbangIlboList.do |
