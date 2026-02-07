# 🚢 Titanic ML Workflow — Leak-Safe Feature Engineering (Kaggle-Ready)

A production-minded workflow on the classic Titanic dataset: **clean data contracts**, **leak-safe feature engineering**, **reproducible modeling**, and a **one-command submission generator**.

**Full case study:** `CASE_STUDY.md`

---

## ✅ What this repo provides
- **Leak-safe feature engineering**
  - `Title` from `Name` (rare titles grouped)
  - `FamilySize` and `IsAlone`
  - `CabinDeck` from `Cabin` (unknown → `U`)
  - Train-only categorical mappings (unknowns in test handled safely)
- **Reproducible modeling**
  - CPU-friendly baseline with fixed seed
  - Optional `--fast` mode for quick iteration
- **Decision policy**
  - Validation-tuned probability threshold (F1-optimized by default; easy to swap objective)
- **One-command Kaggle submission**
  - `python scripts/make_submission.py` → generates `submission.csv`
- **Artifacts for auditability**
  - Exports `artifacts/model.joblib`, `artifacts/metadata.json`, `artifacts/metrics.json`

---

## 🧩 Case Study (1-minute read)
**Goal:** build a strong baseline while showcasing a *real* ML workflow: leak-safe preprocessing, explicit decisions, and reproducible outputs.

**Key decisions**
- Fit all imputers/mappings on **train only**; apply the same transforms to test (unknown categories handled explicitly).
- Use a **CPU-friendly** model for fast iteration and low operational cost.
- Export a **threshold policy** and reuse it consistently during inference/submission generation.
- Export **artifacts + run metadata** so results are auditable and repeatable.

**Outcome**
- A portable pipeline that runs locally or on Kaggle, produces `submission.csv` in one command, and leaves a clean trail of artifacts for review.

---

## 📂 Dataset
This repository does **not** include the Titanic CSVs.

### Local (recommended)
1) Download the Kaggle Titanic competition files.
2) Place them here:
- `data/raw/train.csv`
- `data/raw/test.csv`

See: `data/raw/README.md`

### Kaggle
The submission script also supports Kaggle’s default input paths:
- `/kaggle/input/titanic/train.csv`
- `/kaggle/input/titanic/test.csv`

---

## 🛠️ Environment
- **Python**: 3.10–3.12

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Start (generate `submission.csv`)
```bash
git clone https://github.com/tarekmasryo/titanic-ml-workflow
cd titanic-ml-workflow
pip install -r requirements.txt

# Ensure data/raw/train.csv and data/raw/test.csv exist (or run on Kaggle)
python scripts/make_submission.py --data-dir data/raw --out submission.csv
```

Fast iteration mode (fewer trees):
```bash
python scripts/make_submission.py --data-dir data/raw --out submission.csv --fast
```

After running, you’ll also get:
- `artifacts/model.joblib`
- `artifacts/metadata.json`
- `artifacts/metrics.json`

---

## 📓 Notebook
The notebook (`Titanic - Advanced Feature Engineering.ipynb`) walks through:
- auditing + cleaning
- feature engineering decisions
- model training + evaluation
- interpretability checks

---

## 🔍 Notes on Methodology
- **No leakage**: encoders/mappings are fit on train only; unknown categories in test are handled explicitly.
- **Threshold policy**: selected on a validation split (default objective: F1).
- **Interpretability**: feature importance + permutation importance (when applicable).

---

## 📜 License
MIT (code) — dataset subject to Kaggle competition terms.
