
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import imblearn
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "cic_binary_train.csv"

RUN_ID = "rf_balanced_subsample_shap_20261030"
OUT_DIR = ROOT / "experiments" / "shap_analysis" / RUN_ID
MODEL_DIR = ROOT / "src" / "models"

SEED = 42
VAL_FRACTION = 0.20
TRAIN_SAMPLE_SIZE = 300_000

EXPECTED_FEATURES = 78
LABEL_COL = "normalized_label"
GROUP_COL = "group_id"
BENIGN_LABEL = "BENIGN"

METADATA_COLS = [
    "source_file",
    "source_row",
    "group_id",
    LABEL_COL,
    "binary_label",
]


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_data(path: Path):
    print("Loading CIC binary training data...")

    df = pd.read_csv(path, low_memory=False)

    family = (
        df[LABEL_COL]
        .astype(str)
        .str.strip()
        .str.upper()
        .to_numpy()
    )

    if BENIGN_LABEL not in set(family):
        raise ValueError("BENIGN label not found.")

    y = (family != BENIGN_LABEL).astype(np.int8)
    groups = df[GROUP_COL].to_numpy()

    feature_cols = [
        col for col in df.columns
        if col not in METADATA_COLS
    ]

    if len(feature_cols) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"found {len(feature_cols)}."
        )

    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32)
        .to_numpy()
    )

    return X, y, family, groups, feature_cols


# --------------------------------------------------
# MODEL
# --------------------------------------------------

def build_model():

    return ImbPipeline([
        (
            "impute",
            SimpleImputer(strategy="median"),
        ),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
    ])


# --------------------------------------------------
# DATASET HASH
# --------------------------------------------------

def calculate_sha256(path: Path) -> str:

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    # Load dataset
    X, y, family, groups, feature_cols = load_data(TRAIN_PATH)

    print(f"Dataset shape: {X.shape}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Attack share: {y.mean():.6f}")

    # Group-level train/validation split
    print("\nCreating group-level validation split...")

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_FRACTION,
        random_state=SEED,
    )

    inner_idx, val_idx = next(
        gss.split(X, y, groups)
    )

    overlap = np.intersect1d(
        groups[inner_idx],
        groups[val_idx],
    )

    if overlap.size:
        raise RuntimeError(
            f"Group leakage detected: {overlap.size} groups."
        )

    # Stratified training sample
    print("Selecting stratified training sample...")

    sample_idx, _ = train_test_split(
        inner_idx,
        train_size=TRAIN_SAMPLE_SIZE,
        stratify=y[inner_idx],
        random_state=SEED,
    )

    X_train = X[sample_idx]
    y_train = y[sample_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    family_val = family[val_idx]

    # Verify split
    train_groups = set(groups[sample_idx])
    validation_groups = set(groups[val_idx])

    if train_groups.intersection(validation_groups):
        raise RuntimeError(
            "Training and validation groups overlap."
        )

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Training attack share: {y_train.mean():.6f}")
    print(f"Validation attack share: {y_val.mean():.6f}")
    print(f"Training groups: {len(train_groups):,}")
    print(f"Validation groups: {len(validation_groups):,}")

    # Save exact split indices
    np.save(
        OUT_DIR / "training_sample_indices.npy",
        sample_idx,
    )

    np.save(
        OUT_DIR / "validation_indices.npy",
        val_idx,
    )

    # Train
    print("\nTraining Random Forest...")

    model = build_model()

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    # Save model
    model_path = MODEL_DIR / f"{RUN_ID}.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved: {model_path}")
    print(f"Training time: {fit_seconds:.2f} seconds")

    # Validation performance
    print("\nEvaluating validation set...")

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "macro_f1": float(
            f1_score(y_val, y_pred, average="macro")
        ),
        "attack_precision": float(
            precision_score(y_val, y_pred, zero_division=0)
        ),
        "attack_recall": float(
            recall_score(y_val, y_pred, zero_division=0)
        ),
        "false_positive_rate": float(fpr),
        "average_precision": float(
            average_precision_score(y_val, y_prob)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fit_seconds": round(fit_seconds, 3),
    }

    print(json.dumps(metrics, indent=2))

    report = classification_report(
        y_val,
        y_pred,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    )

    (OUT_DIR / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    (OUT_DIR / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Per-family binary detection recall
    family_results = []

    for name in sorted(set(family_val) - {BENIGN_LABEL}):

        mask = family_val == name

        family_results.append({
            "family": name,
            "support": int(mask.sum()),
            "detected": int(y_pred[mask].sum()),
            "recall": float(y_pred[mask].mean()),
        })

    pd.DataFrame(family_results).to_csv(
        OUT_DIR / "per_family_recall.csv",
        index=False,
    )

    # Save reproducibility metadata
    config = {
        "run_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "New reproducible SHAP model run; "
            "not the original historical Phase 3 fitted model."
        ),
        "dataset_path": str(TRAIN_PATH),
        "dataset_sha256": calculate_sha256(TRAIN_PATH),
        "seed": SEED,
        "split": {
            "method": "GroupShuffleSplit",
            "group_column": GROUP_COL,
            "validation_fraction": VAL_FRACTION,
            "training_sample_method": "stratified train_test_split",
            "training_sample_size": TRAIN_SAMPLE_SIZE,
            "training_rows": int(len(sample_idx)),
            "validation_rows": int(len(val_idx)),
            "training_attack_share": float(y_train.mean()),
            "validation_attack_share": float(y_val.mean()),
            "group_overlap": 0,
        },
        "features": feature_cols,
        "excluded_columns": METADATA_COLS,
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": SEED,
        },
        "preprocessing": {
            "imputation": "median",
            "imputer_fit_on": "training_sample_only",
        },
        "packages": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
            "shap": shap.__version__,
        },
        "model_path": str(model_path),
        "fit_seconds": round(fit_seconds, 3),
        "total_script_seconds": round(
            time.perf_counter() - start_time, 3
        ),
    }

    (OUT_DIR / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    # Save feature names separately for SHAP
    (OUT_DIR / "feature_names.json").write_text(
        json.dumps(feature_cols, indent=2),
        encoding="utf-8",
    )

    print("\nRun completed successfully.")
    print(f"Results directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
