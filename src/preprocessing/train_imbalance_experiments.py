from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

# Project root is two levels above src/preprocessing/
ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "cic_binary_train.csv"
OUT_DIR = ROOT / "experiments" / "imbalance_experiments"

SEED = 42
VAL_FRACTION = 0.20              # group-level validation split from the training data
TRAIN_SAMPLE_SIZE = 300_000      # sampled from the inner training portion only
SMOTE_RATIO = 0.5                # minority/majority ratio after oversampling
EXPECTED_FEATURES = 78

LABEL_COL = "normalized_label"
GROUP_COL = "group_id"
METADATA_COLS = ["source_file", "source_row", "group_id", LABEL_COL, "binary_label"]
BENIGN_LABEL = "BENIGN"


def load_split(path: Path):
    """Return features (float32), binary targets (ATTACK=1), attack family labels, group IDs, feature names."""
    df = pd.read_csv(path, low_memory=False)

    family = df[LABEL_COL].astype(str).str.strip().str.upper().to_numpy()
    if BENIGN_LABEL not in set(family):
        raise ValueError(f"No BENIGN rows found in {path.name}.")
    y = (family != BENIGN_LABEL).astype(np.int8)
    groups = df[GROUP_COL].to_numpy()

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    if len(feature_cols) != EXPECTED_FEATURES:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_FEATURES} model features, got {len(feature_cols)}."
        )

    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32)
        .to_numpy()
    )
    return X, y, family, groups, feature_cols


def build_model(kind: str, class_weight: str | None, use_smote: bool):
    """Every step inside the pipeline is fit on the training fold only."""
    steps = [("impute", SimpleImputer(strategy="median"))]

    if kind == "lr":
        steps.append(("scale", StandardScaler()))
        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs", class_weight=class_weight, random_state=SEED
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    if use_smote:
        # Runs only during fit, so validation rows are never oversampled
        steps.append(
            ("smote", SMOTE(sampling_strategy=SMOTE_RATIO, k_neighbors=5, random_state=SEED))
        )

    steps.append(("clf", clf))
    return ImbPipeline(steps)


CONFIGS = [
    # (display name, model kind, class_weight, use SMOTE)
    ("LR baseline", "lr", None, False),
    ("LR class_weight=balanced", "lr", "balanced", False),
    ("LR SMOTE", "lr", None, True),
    ("RF baseline", "rf", None, False),
    ("RF class_weight=balanced_subsample", "rf", "balanced_subsample", False),
    ("RF SMOTE", "rf", None, True),
]


def evaluate(name, model, X_val, y_val, fam_val, fit_seconds: float):
    start = time.perf_counter()
    y_pred = model.predict(X_val)
    predict_seconds = time.perf_counter() - start

    scores = model.predict_proba(X_val)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    metrics = {
        "config": name,
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_f1": f1_score(y_val, y_pred, average="macro"),
        "attack_precision": precision_score(y_val, y_pred, zero_division=0),
        "attack_recall": recall_score(y_val, y_pred, zero_division=0),
        "false_positive_rate": fpr,
        "pr_auc": average_precision_score(y_val, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fit_seconds": round(fit_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
    }

    # Per-family recall: share of each attack family flagged as ATTACK
    family_rows = []
    for fam in sorted(set(fam_val) - {BENIGN_LABEL}):
        mask = fam_val == fam
        family_rows.append({
            "config": name,
            "family": fam,
            "support": int(mask.sum()),
            "detected": int(y_pred[mask].sum()),
            "recall": float(y_pred[mask].mean()),
        })

    print(f"  macro_f1={metrics['macro_f1']:.4f} attack_recall={metrics['attack_recall']:.4f} "
          f"FPR={fpr:.4f} PR-AUC={metrics['pr_auc']:.4f} fit={fit_seconds:.1f}s")
    return metrics, family_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    X, y, fam, groups, feature_cols = load_split(TRAIN_PATH)

    # Group-level split: no group_id appears in both inner training and validation
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    inner_idx, val_idx = next(gss.split(X, y, groups))
    overlap = np.intersect1d(groups[inner_idx], groups[val_idx])
    if overlap.size:
        raise RuntimeError(f"Group leakage detected: {overlap.size} shared group IDs.")

    # Stratified sample from the inner training portion
    sample_idx, _ = train_test_split(
        inner_idx,
        train_size=TRAIN_SAMPLE_SIZE,
        stratify=y[inner_idx],
        random_state=SEED,
    )

    X_tr, y_tr = X[sample_idx], y[sample_idx]
    X_val, y_val, fam_val = X[val_idx], y[val_idx], fam[val_idx]
    del X, y, fam, groups

    print(f"Inner train sample: {X_tr.shape}, ATTACK share: {y_tr.mean():.4f}")
    print(f"Validation set:     {X_val.shape}, ATTACK share: {y_val.mean():.4f}")

    results, family_results = [], []
    for name, kind, class_weight, use_smote in CONFIGS:
        print(f"\n=== {name} ===")
        model = build_model(kind, class_weight, use_smote)

        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - start

        metrics, fam_rows = evaluate(name, model, X_val, y_val, fam_val, fit_seconds)
        results.append(metrics)
        family_results.extend(fam_rows)

    summary = pd.DataFrame(results)
    family_df = pd.DataFrame(family_results)

    summary.to_csv(OUT_DIR / "validation_metrics.csv", index=False)
    family_df.to_csv(OUT_DIR / "per_family_recall.csv", index=False)

    pivot = family_df.pivot_table(index=["family", "support"], columns="config", values="recall")
    pivot.to_csv(OUT_DIR / "per_family_recall_pivot.csv")

    config = {
        "seed": SEED,
        "train_path": str(TRAIN_PATH),
        "validation": "GroupShuffleSplit on group_id (test_size=0.20); test CSV not used",
        "inner_train_sample_size": TRAIN_SAMPLE_SIZE,
        "validation_rows": int(len(y_val)),
        "validation_attack_share": float(y_val.mean()),
        "num_features": len(feature_cols),
        "excluded_columns": METADATA_COLS,
        "imputation": "median, fit inside each pipeline on training fold only",
        "smote": {"sampling_strategy": SMOTE_RATIO, "k_neighbors": 5,
                  "note": "binary SMOTE; interpolates across all attack families"},
        "decision_threshold": "default predict() threshold; probabilities not recalibrated",
        "configs": [c[0] for c in CONFIGS],
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n=== Validation summary ===")
    print(summary[["config", "macro_f1", "attack_recall", "false_positive_rate",
                   "pr_auc", "fit_seconds"]].to_string(index=False))
    print("\n=== Per-family recall (validation) ===")
    print(pivot.round(4).to_string())
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()