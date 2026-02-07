#!/usr/bin/env python3
"""
Generate a Kaggle Titanic submission from the classic ML workflow.

- Leak-safe feature engineering (train-only mappings)
- RandomForest model (CPU-friendly)
- F1-tuned threshold (exported & reused)

Usage:
  python scripts/make_submission.py --data-dir data/raw --out submission.csv
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


def _resolve_paths(data_dir: Path) -> Tuple[Path, Path]:
    """Resolve train/test CSV paths from a local folder or Kaggle default path."""
    local_train = data_dir / "train.csv"
    local_test = data_dir / "test.csv"
    if local_train.exists() and local_test.exists():
        return local_train, local_test

    # Kaggle fallback (competition input)
    kaggle_train = Path("/kaggle/input/titanic/train.csv")
    kaggle_test = Path("/kaggle/input/titanic/test.csv")
    if kaggle_train.exists() and kaggle_test.exists():
        return kaggle_train, kaggle_test

    raise FileNotFoundError(
        "Could not find 'train.csv' and 'test.csv'.\n"
        f"Tried:\n  - {local_train}\n  - {local_test}\n"
        f"  - {kaggle_train}\n  - {kaggle_test}\n\n"
        "Fix: place the Kaggle Titanic files under the provided data directory."
    )


def _extract_title(name: str) -> str:
    import re
    m = re.search(r" ([A-Za-z]+)\.", str(name))
    return m.group(1) if m else "Unknown"


def _normalize_titles(s: pd.Series) -> pd.Series:
    s = s.fillna("Unknown").astype(str)
    # Canonicalize common variants
    s = s.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    rare = {
        "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
        "Sir", "Jonkheer", "Dona"
    }
    s = s.apply(lambda t: "Rare" if t in rare else t)
    return s


def _fit_age_imputer(train_df: pd.DataFrame) -> Dict[Tuple[str, str, int], float]:
    # Median age by (Title, Sex, Pclass)
    grp = train_df.groupby(["Title", "Sex", "Pclass"])["Age"].median()
    return {k: float(v) for k, v in grp.dropna().items()}


def _impute_age(df: pd.DataFrame, age_map: Dict[Tuple[str, str, int], float], global_median: float) -> pd.DataFrame:
    def _row_age(r):
        if pd.notna(r["Age"]):
            return r["Age"]
        key = (r["Title"], r["Sex"], int(r["Pclass"]))
        return age_map.get(key, global_median)

    df["Age"] = df.apply(_row_age, axis=1)
    return df


def _prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    # Basic cleanup
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Ensure required columns exist
    required = {"PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name", "Cabin", "Ticket"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Train is missing columns: {sorted(missing)}")

    # Titles
    train_df["Title"] = train_df["Name"].apply(_extract_title)
    test_df["Title"] = test_df["Name"].apply(_extract_title)
    train_df["Title"] = _normalize_titles(train_df["Title"])
    test_df["Title"] = _normalize_titles(test_df["Title"])

    # Family features
    for df in (train_df, test_df):
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # CabinDeck
    for df in (train_df, test_df):
        deck = df["Cabin"].fillna("U").astype(str).str[0]
        df["CabinDeck"] = deck.replace({"n": "U"})  # defensive
        df["CabinDeck"] = df["CabinDeck"].fillna("U")

    # Embarked
    embarked_mode = train_df["Embarked"].dropna().mode()
    embarked_mode = embarked_mode.iloc[0] if len(embarked_mode) else "S"
    train_df["Embarked"] = train_df["Embarked"].fillna(embarked_mode)
    test_df["Embarked"] = test_df["Embarked"].fillna(embarked_mode)

    # Fare
    # Use median fare by Pclass from train, with global fallback
    fare_by_pclass = train_df.groupby("Pclass")["Fare"].median().to_dict()
    global_fare = float(train_df["Fare"].median())
    def _fill_fare(df):
        def _row_fare(r):
            if pd.notna(r["Fare"]):
                return r["Fare"]
            return float(fare_by_pclass.get(int(r["Pclass"]), global_fare))
        df["Fare"] = df.apply(_row_fare, axis=1)
        return df
    train_df = _fill_fare(train_df)
    test_df = _fill_fare(test_df)

    # Age
    age_map = _fit_age_imputer(train_df)
    global_age = float(train_df["Age"].median())
    train_df = _impute_age(train_df, age_map, global_age)
    test_df = _impute_age(test_df, age_map, global_age)

    # Drop non-feature columns
    drop_cols = ["Cabin", "PassengerId", "Name", "Ticket"]
    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    # Leak-safe encoding: fit mappings on train only
    cat_cols = ["Sex", "Embarked", "Title", "CabinDeck"]
    mappings: Dict[str, Dict[str, int]] = {}
    for col in cat_cols:
        vals = pd.Series(X_train[col].astype(str).unique()).sort_values()
        mapping = {v: i for i, v in enumerate(vals.tolist())}
        mappings[col] = mapping
        X_train[col] = X_train[col].astype(str).map(mapping).astype(int)
        X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1).astype(int)

    return X_train, X_test, mappings


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_th = 0.5
    best_f1 = -1.0
    for th in thresholds:
        pred = (proba >= th).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    return best_th


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/raw", help="Folder containing train.csv and test.csv")
    p.add_argument("--out", type=str, default="submission.csv", help="Output submission CSV path")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--fast", action="store_true", help="Faster run (fewer trees)")
    args = p.parse_args()

    rng = int(args.seed)
    np.random.seed(rng)

    data_dir = Path(args.data_dir)
    train_path, test_path = _resolve_paths(data_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    passenger_ids = test_df["PassengerId"].copy()

    X_all, X_test, mappings = _prepare_features(train_df, test_df)
    y_all = train_df["Survived"].astype(int).values

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all, y_all, test_size=0.2, random_state=rng, stratify=y_all
    )

    n_estimators = 300 if args.fast else 600
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=rng,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
    )
    model.fit(X_tr, y_tr)

    proba_va = model.predict_proba(X_va)[:, 1]
    th = _best_threshold(y_va, proba_va)

    # Simple validation snapshot
    pred_va = (proba_va >= th).astype(int)
    val_metrics = {
        "acc": float(accuracy_score(y_va, pred_va)),
        "f1": float(f1_score(y_va, pred_va)),
        "roc_auc": float(roc_auc_score(y_va, proba_va)),
        "threshold": float(th),
        "n_estimators": int(n_estimators),
        "seed": int(rng),
    }

    # Fit final model on all data
    model.fit(X_all, y_all)
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= th).astype(int)

    out_path = Path(args.out)
    submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": pred_test.astype(int)})
    submission.to_csv(out_path, index=False)

    # Export artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "model.joblib")
    meta = {
        "model": "RandomForestClassifier",
        "threshold": float(th),
        "seed": int(rng),
        "n_estimators": int(n_estimators),
        "mappings": mappings,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Artifacts: {artifacts_dir}/model.joblib, metadata.json, metrics.json")


if __name__ == "__main__":
    main()
