from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Project root is two levels above src/preprocessing/
ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "cic_binary_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "cic_binary_test.csv"
OUT_DIR = ROOT / "experiments" / "binary_baselines"

SEED = 42
TRAIN_SAMPLE_SIZE = 300_000
LABEL_COL = "normalized_label"
METADATA_COLS = ["source_file", "source_row", "group_id", LABEL_COL, "binary_label"]
EXPECTED_FEATURES = 78
BENIGN_LABEL = "BENIGN"
CLASS_NAMES = ["BENIGN", "ATTACK"]


def load_xy(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load a split. Returns features, binary targets (ATTACK=1), and feature names.

    normalized_label holds fine-grained labels (BENIGN, DDOS, PORTSCAN, ...).
    Every non-BENIGN label is mapped to the binary ATTACK class.
    """
    df = pd.read_csv(path, low_memory=False)

    labels = df[LABEL_COL].astype(str).str.strip().str.upper()
    if BENIGN_LABEL not in set(labels.unique()):
        raise ValueError(f"No BENIGN rows found in {path.name}; check the label column.")

    print(f"  Label counts in {path.name}:")
    for label, count in labels.value_counts().items():
        print(f"    {label:<30} {count:>10,}")

    y = (labels != BENIGN_LABEL).astype(np.int8).to_numpy()


    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    if len(feature_cols) != EXPECTED_FEATURES:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_FEATURES} model features, "
            f"got {len(feature_cols)}: {feature_cols}"
        )
    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)  # infinities become missing values
        .astype(np.float32)
    )
    return X, y, feature_cols


def evaluate(name, model, X_test, y_test, fit_seconds: float) -> dict:
    # Time only the hard predictions, which is the latency that matters for deployment
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_seconds = time.perf_counter() - start

    # Attack scores for PR-AUC (not included in the timing above)
    scores = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    negatives = fp + tn
    fpr = fp / negatives if negatives else 0.0

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "attack_precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "attack_recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "false_positive_rate": fpr,
        "pr_auc": average_precision_score(y_test, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fit_seconds": round(fit_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "latency_us_per_row": round(predict_seconds / len(y_test) * 1e6, 4),
    }

    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    safe_name = name.lower().replace(" ", "_")
    (OUT_DIR / f"{safe_name}_classification_report.txt").write_text(report, encoding="utf-8")

    print(f"\n=== {name} ===")
    print(report)
    print(f"PR-AUC: {metrics['pr_auc']:.4f} | FPR: {fpr:.4f} | fit: {fit_seconds:.1f}s")
    return metrics


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training split...")
    X_train_full, y_train_full, train_features = load_xy(TRAIN_PATH)

    # Stratified sample of the training split. The test set is not involved here.
    X_train, _, y_train, _ = train_test_split(
        X_train_full,
        y_train_full,
        train_size=TRAIN_SAMPLE_SIZE,
        stratify=y_train_full,
        random_state=SEED,
    )
    del X_train_full, y_train_full  # free memory

    print("\nLoading test split...")
    X_test, y_test, test_features = load_xy(TEST_PATH)

    if train_features != test_features:
        raise ValueError("Train and test feature columns do not match.")

    print(f"\nTrain sample: {X_train.shape}, ATTACK share: {y_train.mean():.4f}")
    print(f"Test set:     {X_test.shape}, ATTACK share: {y_test.mean():.4f}")

    # Imputers are inside each pipeline, so they are fit on the training sample only.
    # StandardScaler is used only for Logistic Regression.
    models = {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
    }

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        results.append(evaluate(name, model, X_test, y_test, fit_seconds))

    summary = pd.DataFrame(results)
    summary.to_csv(OUT_DIR / "baseline_metrics.csv", index=False)

    config = {
        "seed": SEED,
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "train_sample_size": TRAIN_SAMPLE_SIZE,
        "train_sample_attack_share": float(y_train.mean()),
        "test_rows": int(len(y_test)),
        "test_attack_share": float(y_test.mean()),
        "num_features": len(train_features),
        "excluded_columns": METADATA_COLS,
        "label_mapping": "BENIGN -> 0, all other labels -> 1 (ATTACK)",
        "class_weighting": None,
        "imputation": "median, fit on training sample only",
        "models": {
            "Logistic Regression": "SimpleImputer(median) + StandardScaler + LogisticRegression(max_iter=1000)",
            "Random Forest": "SimpleImputer(median) + RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2)",
        },
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(summary[["model", "accuracy", "macro_f1", "attack_precision",
                   "attack_recall", "false_positive_rate", "pr_auc",
                   "fit_seconds", "latency_us_per_row"]].to_string(index=False))
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
