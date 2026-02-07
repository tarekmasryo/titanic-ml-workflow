# Case Study — Titanic ML Workflow (Portfolio-Grade Baseline)

## Context
The Titanic dataset is widely used as an introductory benchmark, which means the model itself is rarely the differentiator. The differentiator is **workflow quality**: leak-safe preprocessing, reproducibility, and an auditable path from data → model → submission.

This repository is designed to demonstrate a **production-minded ML workflow** on a familiar dataset: it runs on CPU, requires minimal dependencies, and produces a Kaggle `submission.csv` in one command.

## Goals
- Build a solid baseline with **feature engineering that reflects real-world thinking**.
- Ensure **no leakage** during preprocessing and validation.
- Make the workflow **repeatable** (fixed seeds, deterministic transforms where possible).
- Produce outputs that are **usable and auditable** beyond a notebook.

## Key Decisions

### 1) Leak-safe feature engineering
The workflow creates a small set of high-signal engineered features while enforcing a strict rule:
> Any mapping, imputer, or encoder is fit on the training split only, then applied to the test split.

Examples of engineered signals:
- `Title` extracted from passenger names (rare titles grouped)
- `FamilySize` and `IsAlone` to represent social context
- `CabinDeck` derived from cabin identifiers (unknown mapped to `U`)

Categorical features are mapped using **train-only dictionaries**, and unknown categories in test are handled explicitly rather than silently leaking information.

### 2) CPU-friendly model choice
For this dataset, the goal is not “deep learning because it’s trendy,” but a model that:
- runs quickly on CPU,
- provides stable results across runs,
- and supports probability-based decision policies.

A RandomForest baseline satisfies these constraints and stays easy to explain and maintain.

### 3) Decision policy via thresholding
Instead of treating classification as a fixed 0.5 cutoff, this repo exports an explicit **threshold policy** tuned on a validation split (F1-optimized by default). This makes trade-offs concrete:
- lower threshold → higher recall (more positives)
- higher threshold → higher precision (fewer false positives)

For Kaggle, you can swap the objective to accuracy if desired, but the main point is demonstrating **explicit decision control**.

## Evaluation
A validation split is used to:
- compute a compact metric snapshot (accuracy, F1, ROC-AUC),
- select the operating threshold,
- and produce an auditable record of the run.

The metrics for a given run are exported to `artifacts/metrics.json`.

## Artifacts (Audit Trail)
After generating a submission, the repo exports:
- `artifacts/model.joblib` — trained model
- `artifacts/metadata.json` — configuration, threshold, seed, and train-only mappings
- `artifacts/metrics.json` — validation snapshot

This turns the workflow into something you can review, reproduce, and explain—without relying on notebook state.

## How to Run
1) Install:
- `pip install -r requirements.txt`

2) Put data in:
- `data/raw/train.csv`
- `data/raw/test.csv`

3) Generate a submission:
- `python scripts/make_submission.py --data-dir data/raw --out submission.csv`

A `--fast` flag is available for quick iteration.

## Limitations
- Titanic is a toy dataset; this is a workflow showcase, not a production deployment claim.
- Leaderboard scores vary across splits and randomness; the emphasis here is **methodology and auditability**.

## Next Steps
If turning this into a stronger “systems” demo, a natural extension would be:
- add a smoke test that runs the submission script on a tiny synthetic dataset,
- add a simple model comparison table (logistic regression vs RF vs boosting),
- and document a cost-aware decision policy (FP vs FN costs) for threshold selection.
