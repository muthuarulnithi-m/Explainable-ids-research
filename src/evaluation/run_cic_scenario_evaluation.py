"""
Phase 6.1 - CIC scenario-based evaluation (main result "A").

Evaluates the three saved multiclass XGBoost models on cic_binary_test.csv,
separately for each capture file (scenario). Nothing is trained, refitted
or tuned. No output is written if any check fails.

Run from the project root:
    python src\\evaluation\\run_cic_scenario_evaluation.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED / "cic_binary_train.csv"
TEST_PATH = PROCESSED / "cic_binary_test.csv"

MODEL_DIR = ROOT / "experiments" / "multiclass_cic"
IMPUTER_PATH = MODEL_DIR / "multiclass_imputer.joblib"
ENCODER_PATH = MODEL_DIR / "multiclass_label_encoder.joblib"
VALIDATION_POSITIONS = MODEL_DIR / "validation_positions.npy"
MODELS = {
    "xgboost_baseline": MODEL_DIR / "multiclass_xgboost.json",
    "xgboost_balanced": MODEL_DIR / "imbalance" / "xgboost_balanced.json",
    "xgboost_sqrt_balanced": MODEL_DIR / "imbalance" / "xgboost_sqrt_balanced.json",
}

OUTPUT_DIR = ROOT / "experiments" / "generalization" / "cic_scenario"

EXPECTED_FEATURES = 78
EXPECTED_CLASSES = 15
EXPECTED_TRAIN_ROWS = 2_255_853
EXPECTED_TEST_ROWS = 567_870
METADATA_COLUMNS = ["source_file", "source_row", "group_id",
                    "normalized_label", "binary_label"]
LOW_SUPPORT_THRESHOLD = 30
POOLED = "ALL_TEST_DAYS"

# Recorded in the Phase 5 summary (rounded to six decimals).
RECORDED_VALIDATION_MACRO_F1 = {
    "xgboost_baseline": 0.787566,
    "xgboost_balanced": 0.790589,
    "xgboost_sqrt_balanced": 0.795563,
}
REPRODUCTION_TOLERANCE = 1e-5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"\nFAILED: {message}\nNo output was written.")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validation_rows(feature_names: list[str]) -> pd.DataFrame:
    """Read only the saved validation rows from the training file, in chunks."""
    positions = np.load(VALIDATION_POSITIONS)
    keep = np.zeros(EXPECTED_TRAIN_ROWS, dtype=bool)
    keep[positions] = True

    parts, offset = [], 0
    reader = pd.read_csv(
        TRAIN_PATH,
        usecols=feature_names + ["normalized_label"],
        chunksize=200_000,
    )
    for chunk in reader:
        n = len(chunk)
        parts.append(chunk.loc[keep[offset:offset + n]])
        offset += n

    require(offset == EXPECTED_TRAIN_ROWS,
            f"training file has {offset} rows, expected {EXPECTED_TRAIN_ROWS}")
    validation = pd.concat(parts, ignore_index=True)
    require(len(validation) == len(positions),
            f"loaded {len(validation)} validation rows, expected {len(positions)}")
    return validation


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray):
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    summary = {
        "rows": int(len(y_true)),
        "classes_in_truth": int(len(set(y_true))),
        "classes_in_union": len(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
    }
    per_class = pd.DataFrame({
        "class": labels,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return summary, per_class, labels, cm


def main() -> None:
    print("=" * 70)
    print("PHASE 6.1 - CIC SCENARIO-BASED EVALUATION (main result A)")
    print("=" * 70)

    # 1. Files and output folder
    required_files = [TRAIN_PATH, TEST_PATH, IMPUTER_PATH, ENCODER_PATH,
                      VALIDATION_POSITIONS, *MODELS.values()]
    for path in required_files:
        require(path.exists(), f"missing file: {path}")
    require(not (OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir())),
            f"output folder already has content, refusing to overwrite: {OUTPUT_DIR}")
    print("[PASS] required files exist; output folder is new or empty")

    # 2. Saved preprocessing and label encoder
    imputer = joblib.load(IMPUTER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    require(len(getattr(imputer, "statistics_", [])) == EXPECTED_FEATURES,
            f"imputer statistics do not cover {EXPECTED_FEATURES} features")
    if hasattr(imputer, "feature_names_in_"):
        feature_names = list(imputer.feature_names_in_)
        order_source = "imputer.feature_names_in_"
    else:
        feature_names = [c for c in pd.read_csv(TEST_PATH, nrows=0).columns
                         if c not in METADATA_COLUMNS]
        order_source = ("non-metadata columns in file order; the imputer stored no "
                        "feature names, so the reproduction check verifies the order")
    require(len(feature_names) == EXPECTED_FEATURES,
            f"found {len(feature_names)} features, expected {EXPECTED_FEATURES}")
    require(len(encoder.classes_) == EXPECTED_CLASSES,
            f"encoder has {len(encoder.classes_)} classes, expected {EXPECTED_CLASSES}")
    print(f"[PASS] imputer and label encoder: {len(feature_names)} features, "
          f"{len(encoder.classes_)} classes; order source: {order_source}")

    # 3. Test file: row count, schema, infinities, labels
    test = pd.read_csv(TEST_PATH)
    require(len(test) == EXPECTED_TEST_ROWS,
            f"test file has {len(test)} rows, expected {EXPECTED_TEST_ROWS}")
    missing = [c for c in feature_names if c not in test.columns]
    require(not missing, f"test file lacks features: {missing[:5]}")
    x_raw = test[feature_names].astype(float)
    infinite_cells = int(np.isinf(x_raw.to_numpy()).sum())
    require(infinite_cells == 0,
            f"{infinite_cells} infinite feature values in the test file")
    nan_cells = int(x_raw.isna().sum().sum())
    unknown_labels = set(test["normalized_label"].unique()) - set(encoder.classes_)
    require(not unknown_labels, f"labels unknown to the encoder: {unknown_labels}")
    print(f"[PASS] test file: {len(test):,} rows, {len(feature_names)} features, "
          f"{nan_cells:,} NaN cells (imputed), 0 infinite, all labels known")

    # 4. Group disjointness from the training file
    train_groups = set(pd.read_csv(TRAIN_PATH, usecols=["group_id"])["group_id"].unique())
    shared = train_groups & set(test["group_id"].unique())
    require(not shared, f"{len(shared)} group_id values shared between train and test")
    print("[PASS] zero group_id overlap between training and test files")

    # 5. Transform only, using the saved imputer
    x_test = pd.DataFrame(imputer.transform(x_raw), columns=feature_names)
    y_true = test["normalized_label"].to_numpy()
    scenarios = test["source_file"].str.replace(
        ".pcap_ISCX.csv", "", regex=False).to_numpy()

    # 6. Validation rows for the reproduction check
    validation = load_validation_rows(feature_names)
    x_val = pd.DataFrame(imputer.transform(validation[feature_names].astype(float)),
                         columns=feature_names)
    y_val = validation["normalized_label"].to_numpy()
    print(f"[PASS] validation rows reloaded for reproduction check: {len(validation):,}")

    summary_rows, class_frames = [], []
    sanity, model_meta = {}, {}
    confusion_dir = OUTPUT_DIR / "confusion_matrices"

    for name, path in MODELS.items():
        model = xgboost.XGBClassifier()
        model.load_model(str(path))

        booster_features = model.get_booster().feature_names
        if booster_features is None:
            feature_status = "feature names not stored in model file; order checked via imputer only"
        else:
            require(list(booster_features) == feature_names,
                    f"{name}: stored feature names or order differ from the imputer")
            feature_status = "stored feature names match imputer order"

        # Reproduction check: recorded validation macro F1 must be reproduced.
        val_pred = encoder.inverse_transform(np.asarray(model.predict(x_val)).astype(int))
        val_f1 = f1_score(y_val, val_pred, labels=list(encoder.classes_),
                          average="macro", zero_division=0)
        recorded = RECORDED_VALIDATION_MACRO_F1[name]
        require(abs(val_f1 - recorded) <= REPRODUCTION_TOLERANCE,
                f"{name}: reproduced validation macro F1 {val_f1:.6f}, "
                f"recorded {recorded:.6f}")
        sanity[name] = {"reproduced": round(float(val_f1), 6), "recorded": recorded}
        print(f"[PASS] {name}: validation macro F1 reproduced "
              f"({val_f1:.6f} vs recorded {recorded:.6f}); {feature_status}")

        # Test predictions (default argmax, no threshold tuning)
        pred_codes = np.asarray(model.predict(x_test)).astype(int)
        require(pred_codes.min() >= 0 and pred_codes.max() < EXPECTED_CLASSES,
                f"{name}: prediction codes outside the encoder range")
        y_pred = encoder.inverse_transform(pred_codes)

        for scenario in [*sorted(set(scenarios)), POOLED]:
            mask = (np.ones(len(y_true), dtype=bool) if scenario == POOLED
                    else scenarios == scenario)
            summary, per_class, labels, cm = metrics_block(y_true[mask], y_pred[mask])
            summary_rows.append({"model": name, "scenario": scenario, **summary})

            per_class.insert(0, "scenario", scenario)
            per_class.insert(0, "model", name)
            per_class["low_support"] = per_class["support"] < LOW_SUPPORT_THRESHOLD
            class_frames.append(per_class)

            confusion_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(
                confusion_dir / f"{name}__{scenario}.csv")

        model_meta[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_of(path),
            "feature_status": feature_status,
        }

    # 7. Write outputs (only reached if every check passed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    distribution = (
        pd.DataFrame({"scenario": scenarios, "class": y_true})
        .groupby(["scenario", "class"]).size().reset_index(name="test_rows")
    )
    distribution["low_support"] = distribution["test_rows"] < LOW_SUPPORT_THRESHOLD
    distribution.to_csv(OUTPUT_DIR / "scenario_distribution.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "scenario_metrics.csv", index=False)
    pd.concat(class_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "per_class_scenario_metrics.csv", index=False)

    config = {
        "phase": "6.1 CIC scenario-based evaluation (main result A)",
        "protocol": ("Saved multiclass XGBoost models evaluated per capture file on "
                     "cic_binary_test.csv. No training, refitting or threshold tuning."),
        "scenario_definition": "one capture file per scenario, '.pcap_ISCX.csv' removed",
        "prediction_rule": "XGBoost default argmax over class probabilities",
        "metric_definition": ("accuracy; macro and weighted F1 by sklearn over the union of "
                              "classes present in y_true or y_pred within each scenario"),
        "pooled_row": f"'{POOLED}' pools all test rows; reference only, not a scenario result",
        "low_support_threshold": LOW_SUPPORT_THRESHOLD,
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "features": feature_names,
        "feature_order_source": order_source,
        "classes": [str(c) for c in encoder.classes_],
        "input_files": {
            "test": {"path": str(TEST_PATH.relative_to(ROOT)),
                     "sha256": sha256_of(TEST_PATH), "rows": len(test)},
            "train": {"path": str(TRAIN_PATH.relative_to(ROOT)),
                      "sha256": sha256_of(TRAIN_PATH), "rows": EXPECTED_TRAIN_ROWS},
        },
        "imputer": {"path": str(IMPUTER_PATH.relative_to(ROOT)),
                    "sha256": sha256_of(IMPUTER_PATH)},
        "label_encoder": {"path": str(ENCODER_PATH.relative_to(ROOT)),
                          "sha256": sha256_of(ENCODER_PATH)},
        "models": model_meta,
        "reproduction_check": sanity,
        "random_seed": "none used in evaluation; training seed 42 is recorded in multiclass_config.json",
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "Each capture file is a scenario; several attack classes occur in only one file.",
            "Models were trained on all capture days; this is per-day evaluation, not scenario-independent training.",
            "Monday contains benign traffic only and gives false-positive information only.",
            "cic_binary_test.csv was used to compare binary LR and RF baselines; it is not a clean holdout for binary work.",
            "Classes below the low-support threshold have unreliable per-class metrics.",
        ],
    }
    with open(OUTPUT_DIR / "scenario_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)

    print("\nSCENARIO METRICS")
    print(pd.DataFrame(summary_rows).round(6).to_string(index=False))
    print(f"\nPython {sys.version.split()[0]} | Output: {OUTPUT_DIR}")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
