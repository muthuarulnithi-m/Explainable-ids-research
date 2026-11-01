
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "cic_binary_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "cic_binary_test.csv"
BASELINE_DIR = ROOT / "experiments" / "binary_baselines"
OUTPUT_DIR = ROOT / "experiments" / "attack_family_evaluation"

CONFIG_PATH = BASELINE_DIR / "run_config.json"
SEED = 42


METADATA_COLUMNS = {
    "source_file",
    "source_row",
    "group_id",
    "normalized_label",
    "binary_label",
}


def binary_target(series: pd.Series) -> np.ndarray:
    """Convert saved binary labels into 0=BENIGN and 1=ATTACK."""
    values = series.astype(str).str.strip().str.upper()

    mapping = {"BENIGN": 0, "ATTACK": 1}
    unknown = sorted(set(values.unique()) - set(mapping))

    if unknown:
        raise ValueError(f"Unexpected binary labels: {unknown}")

    return values.map(mapping).to_numpy(dtype=np.int8)


def load_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Convert features to numeric values and represent infinities as missing."""
    features = frame[feature_columns].copy()

    for column in feature_columns:
        features[column] = pd.to_numeric(
            features[column], errors="coerce"
        )

    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    return features.astype(np.float32)


def build_models() -> dict[str, Pipeline]:
    """Use the model settings recorded in the baseline configuration."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=20,
                        min_samples_leaf=2,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Expected processed train/test CSVs were not found."
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Baseline configuration not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        baseline_config = json.load(file)

    sample_size = int(baseline_config["train_sample_size"])
    seed = int(baseline_config["seed"])

    print("Loading saved training and test datasets...")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    required_columns = METADATA_COLUMNS
    for dataset_name, frame in [
        ("training", train_df),
        ("test", test_df),
    ]:
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{dataset_name} dataset is missing columns: "
                f"{sorted(missing)}"
            )

    feature_columns = [
        column
        for column in train_df.columns
        if column not in METADATA_COLUMNS
    ]

    if feature_columns != [
        column for column in test_df.columns
        if column not in METADATA_COLUMNS
    ]:
        raise ValueError(
            "Training and test feature columns differ or are ordered "
            "differently. Stop and inspect the schemas."
        )

    expected_features = int(baseline_config["num_features"])
    if len(feature_columns) != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, "
            f"found {len(feature_columns)}."
        )

    y_all = binary_target(train_df["binary_label"])
    y_test = binary_target(test_df["binary_label"])

    if sample_size > len(train_df):
        raise ValueError("Configured sample exceeds training row count.")

    # Reproduce a deterministic stratified training sample.
    sample_indices, _ = train_test_split(
        np.arange(len(train_df)),
        train_size=sample_size,
        random_state=seed,
        stratify=y_all,
    )

    sample_df = train_df.iloc[sample_indices]
    y_train = y_all[sample_indices]

    # Verify the sample size and label proportion against the saved config.
    actual_attack_share = float(np.mean(y_train))
    expected_attack_share = float(
        baseline_config["train_sample_attack_share"]
    )

    print(f"Training sample rows: {len(sample_df):,}")
    print(f"Training sample attack share: {actual_attack_share:.6f}")
    print(f"Recorded attack share: {expected_attack_share:.6f}")

    if not np.isclose(
        actual_attack_share, expected_attack_share, atol=1e-6
    ):
        raise ValueError(
            "The reconstructed training sample's attack share differs "
            "from run_config.json. Check the original baseline script "
            "before interpreting results."
        )

    X_train = load_features(sample_df, feature_columns)
    X_test = load_features(test_df, feature_columns)

    # Keep the true multiclass attack names for analysis only.
    # They are never included as model features.
    if test_df["normalized_label"].isna().any():
        raise ValueError("Test data contains missing normalized labels.")

    attack_names = test_df["normalized_label"].astype(str).str.strip()
    if (attack_names == "").any():
        raise ValueError("Test data contains blank normalized labels.")

    models = build_models()
    metric_rows = []
    family_rows = []
    run_details = {
        "seed": seed,
        "train_sample_size": sample_size,
        "test_rows": len(test_df),
        "num_features": len(feature_columns),
        "training_sample_attack_share": actual_attack_share,
        "test_attack_share": float(np.mean(y_test)),
        "sampling_method": "Stratified train_test_split",
        "test_set_used_for_training_or_tuning": False,
        "attack_family_evaluation": (
            "Binary predictions grouped by normalized_label"
        ),
        "feature_columns": feature_columns,
        "models": {},
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        start_fit = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start_fit

        print(f"Evaluating {model_name} on the full test set...")

        start_predict = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_seconds = time.perf_counter() - start_predict

        # Both models expose predict_proba; use attack probability for PR-AUC.
        attack_probabilities = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "attack_precision": float(
                precision_score(y_test, y_pred, zero_division=0)
            ),
            "attack_recall": float(
                recall_score(y_test, y_pred, zero_division=0)
            ),
            "false_positive_rate": float(fpr),
            "pr_auc_average_precision": float(
                average_precision_score(y_test, attack_probabilities)
            ),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "fit_seconds": float(fit_seconds),
            "prediction_seconds": float(predict_seconds),
            "prediction_ms_per_row": float(
                predict_seconds / len(X_test) * 1000
            ),
        }
        metric_rows.append(metrics)

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
            digits=6,
        )

        report_path = (
            OUTPUT_DIR
            / f"{model_name.lower().replace(' ', '_')}_binary_report.txt"
        )
        with report_path.open("w", encoding="utf-8") as file:
            file.write(f"Model: {model_name}\n")
            file.write(f"Fit time (seconds): {fit_seconds:.4f}\n")
            file.write(f"Prediction time (seconds): {predict_seconds:.4f}\n")
            file.write(
                f"Prediction latency (ms/row): "
                f"{metrics['prediction_ms_per_row']:.8f}\n\n"
            )
            file.write("Confusion matrix (rows=true, columns=predicted):\n")
            file.write(f"{confusion_matrix(y_test, y_pred, labels=[0, 1])}\n\n")
            file.write(report)

        # Evaluate attack-family recall using the binary predictions.
        family_frame = pd.DataFrame(
            {
                "normalized_label": attack_names.to_numpy(),
                "binary_true": y_test,
                "binary_pred": y_pred,
            }
        )

        for label, group in family_frame.groupby(
            "normalized_label", sort=True
        ):
            if label.upper() == "BENIGN":
                benign_total = len(group)
                false_alarms = int((group["binary_pred"] == 1).sum())

                family_rows.append(
                    {
                        "model": model_name,
                        "label": label,
                        "category": "BENIGN",
                        "test_rows": benign_total,
                        "detected_as_attack": false_alarms,
                        "missed_as_benign": benign_total - false_alarms,
                        "detection_rate": np.nan,
                        "false_positive_rate": (
                            false_alarms / benign_total
                            if benign_total else np.nan
                        ),
                    }
                )
            else:
                total = len(group)
                detected = int((group["binary_pred"] == 1).sum())

                family_rows.append(
                    {
                        "model": model_name,
                        "label": label,
                        "category": "ATTACK",
                        "test_rows": total,
                        "detected_as_attack": detected,
                        "missed_as_benign": total - detected,
                        "detection_rate": detected / total if total else np.nan,
                        "false_positive_rate": np.nan,
                    }
                )

        run_details["models"][model_name] = {
            "settings": baseline_config["models"][model_name],
            "class_weighting": None,
            "fit_seconds": float(fit_seconds),
            "prediction_seconds": float(predict_seconds),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }

        print(
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f} | "
            f"Attack recall={metrics['attack_recall']:.4f}"
        )

    pd.DataFrame(metric_rows).to_csv(
        OUTPUT_DIR / "binary_metrics.csv", index=False
    )
    pd.DataFrame(family_rows).to_csv(
        OUTPUT_DIR / "attack_family_detection_metrics.csv", index=False
    )

    with (OUTPUT_DIR / "run_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(run_details, file, indent=2)

    summary_path = OUTPUT_DIR / "summary.txt"
    family_results = pd.DataFrame(family_rows)

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("BINARY BASELINE: ATTACK-FAMILY DETECTION\n")
        file.write("=" * 55 + "\n\n")
        file.write(
            "Each attack family's detection rate is the proportion of "
            "that family's test rows predicted as ATTACK by the binary model.\n"
        )
        file.write(
            "This is not multiclass classification. The model does not "
            "predict an attack-family name.\n\n"
        )

        for model_name in models:
            file.write(f"\n{model_name}\n")
            file.write("-" * len(model_name) + "\n")
            subset = family_results[
                family_results["model"] == model_name
            ].sort_values(["category", "label"])

            file.write(subset.to_string(index=False))
            file.write("\n")

    print("\nCompleted. New files saved under:")
    print(OUTPUT_DIR)
    print("- binary_metrics.csv")
    print("- attack_family_detection_metrics.csv")
    print("- summary.txt")
    print("- run_config.json")
    print("- logistic_regression_binary_report.txt")
    print("- random_forest_binary_report.txt")


if __name__ == "__main__":
    main()
