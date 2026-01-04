# SkillSight — NLP Dataset Generator

Synthetic dataset generator for the academic project **Explicit–Implicit Skill Extraction**.

The system generates labeled examples from free-text job descriptions and assigns a score per skill:ס
- `0` = NONE
- `0.5` = IMPLICIT
- `1.0` = EXPLICIT

Repository:
- https://github.com/RoniF24/SkillSight

---

## What this project does

This project creates synthetic training data in two steps:

1) **Create Plans**  
   Defines which skills appear in each example and whether each one is explicit/implicit/none.

2) **Generate Dataset**  
   Uses the plans to produce `job_description` + `skills` labels via OpenAI (**gpt-4o-mini**) or another backend.

---

## Project structure

Common folders/files you will work with:

- `src/`
  - `src/sampler/` — creates plans (`make_plans.py`)
  - `src/generator/` — generates dataset from plans (`generate_dataset.py`)
- `data/` — **final dataset outputs used by the project**
  - `data/synthetic_dataset.jsonl`
  - `data/synthetic_dataset_extra.jsonl`
  - `data/plans/` — plans files (e.g., `plans_v1.jsonl`)
- `outputs/` — run artifacts/logs (not the final dataset)
  - `outputs/full/`
  - `outputs/slim/`

---

## Output

Generated files during runs may be written to:
- `outputs/full/*.jsonl`  (full record)
- `outputs/slim/*.jsonl`  (lighter version)

**Final datasets (for submission / training) are in:**
- `data/synthetic_dataset.jsonl`
- `data/synthetic_dataset_extra.jsonl`

---

## Prerequisites

- Windows 10/11
- Python 3.10+ (added to PATH)
- Git installed

---

## Quick Start - Using Pre-trained Models

### 1. Clone and Setup

```bash
git clone https://github.com/RoniF24/SkillSight.git
cd SkillSight
setup_env.bat
```

This will:
- Create virtual environment
- Install dependencies
- Download models from Hugging Face (~2GB)

### 2. Analyze Text

```bash
python analyze_resume.py --text "I have 5 years of experience with Python and Django" --model deberta
```

For more details, see [MODELS_GUIDE.md](MODELS_GUIDE.md).

**Models hosted at:**
- https://huggingface.co/YonatanEl/skillsight-deberta-v3
- https://huggingface.co/YonatanEl/skillsight-roberta-base
- https://huggingface.co/YonatanEl/skillsight-deberta-v3-onepass

---

## Run locally

### Windows

1) Clone the repo

git clone https://github.com/RoniF24/SkillSight.git  
cd SkillSight

2) Install dependencies (venv + requirements)

setup_env.bat

3) Set OpenAI API key (required for OpenAI backend)

Temporary (only for this terminal):

set OPENAI_API_KEY=PASTE_YOUR_KEY_HERE

Permanent (open a NEW terminal after this):

setx OPENAI_API_KEY "PASTE_YOUR_KEY_HERE"

Verify:

echo %OPENAI_API_KEY%

---

## Generate data (Plans → Dataset)

### Step 1 — Create Plans (choose SEED + amount)

Edit:
- `src/sampler/make_plans.py`

Set:
- `SEED` (choose a different number per teammate to avoid duplicates)
- `PLANS_TOTAL` (use `2` for a quick test, then `1000` for the full run)

Run:

python src/sampler/make_plans.py

This creates/updates the plans file under:
- `data/plans/plans_v1.jsonl`

---

### Step 2 — Generate Dataset from Plans

Recommended: run a small test first:

Generate 2 examples:

python src/generator/generate_dataset.py --backend openai --model gpt-4o-mini --n 2 --temperature 0 --show_text

Generate 1000 examples:

python src/generator/generate_dataset.py --backend openai --model gpt-4o-mini --n 1000 --temperature 0

After a successful run, verify your **final dataset files** under `data/`:
- `data/synthetic_dataset.jsonl`
- `data/synthetic_dataset_extra.jsonl`

---

## Common issues

### HTTP 400 Bad Request
- Model name must be exactly: `gpt-4o-mini`
- `OPENAI_API_KEY` must be set (not empty)

### Generated 0 rows
- Usually API calls failed (missing/invalid key or model name)
- Try a small run first (`--n 2`) and check printed `[FAIL] ...` lines

---

## Team tips

To avoid generating identical plans:
- each teammate should use a different `SEED`
- optionally keep separate plan files per seed (e.g., `plans_seed23.jsonl`)
