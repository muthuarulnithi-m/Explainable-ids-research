
from pathlib import Path
from collections import Counter
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data" / "processed" / "cic_binary_train.csv"
TEST_FILE = ROOT / "data" / "processed" / "cic_binary_test.csv"

CONFIG_FILE = (
    ROOT
    / "experiments"
    / "generalization"
    / "temporal"
    / "cutoff3_experiment_config.json"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "generalization"
    / "temporal"
    / "cutoff3"
)

MODEL_DIR = OUTPUT_DIR / "models"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
REPORT_DIR = OUTPUT_DIR / "reports"
CONFIG_DIR = OUTPUT_DIR / "configs"

for directory in [
    MODEL_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    CONFUSION_DIR,
    REPORT_DIR,
    CONFIG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000
TRAIN_SAMPLE_SIZE = 300_000
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5

METADATA_COLUMNS = {
    "source_file",
    "source_row",
    "group_id",
    "normalized_label",
    "binary_label",
}

# These must match the frozen configuration.
CAPTURE_ORDER = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

TRAIN_FILES = set(CAPTURE_ORDER[:3])
EVALUATION_FILES = set(CAPTURE_ORDER[3:])

LABEL_TO_INT = {
    "BENIGN": 0,
    "ATTACK": 1,
}

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn"
)


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs():
    for path in [TRAIN_FILE, TEST_FILE, CONFIG_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    if config["split"]["cutoff"] != 3:
        raise ValueError("Configuration cutoff is not 3.")

    if config["training_performed"]:
        raise ValueError(
            "Configuration indicates training was already performed. "
            "Review it before starting another run."
        )

    if config["threshold_tuning_performed"]:
        raise ValueError(
            "Unexpected threshold tuning status in frozen configuration."
        )

    if config["dataset"]["feature_count"] != 78:
        raise ValueError("Expected 78 model features.")

    train_columns = pd.read_csv(TRAIN_FILE, nrows=0).columns.tolist()
    test_columns = pd.read_csv(TEST_FILE, nrows=0).columns.tolist()

    if train_columns != test_columns:
        raise ValueError("Train/test schemas differ.")

    features = [
        column for column in train_columns
        if column not in METADATA_COLUMNS
    ]

    if features != config["dataset"]["feature_columns"]:
        raise ValueError("Feature names or feature order changed.")

    schema_hash = hashlib.sha256(
        "\n".join(features).encode("utf-8")
    ).hexdigest()

    if schema_hash != config["dataset"]["feature_schema_sha256"]:
        raise ValueError("Feature schema SHA256 mismatch.")

    return config, train_columns, features


# ============================================================
# DATA HELPERS
# ============================================================

def validate_labels(frame):
    labels = set(frame["binary_label"].dropna().astype(str).unique())

    if not labels.issubset(LABEL_TO_INT):
        raise ValueError(f"Unexpected binary labels: {labels}")

    benign_mask = frame["normalized_label"].astype(str).eq("BENIGN")
    binary_benign_mask = frame["binary_label"].astype(str).eq("BENIGN")

    if not np.array_equal(
        benign_mask.to_numpy(),
        binary_benign_mask.to_numpy()
    ):
        raise ValueError(
            "normalized_label and binary_label are inconsistent."
        )


def feature_hashes(frame, feature_columns):
    values = frame[feature_columns].astype("float64")

    return pd.util.hash_pandas_object(
        values,
        index=False,
        categorize=False
    ).to_numpy(dtype="uint64")


def read_chunks(path, columns):
    return pd.read_csv(
        path,
        usecols=columns,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )


# ============================================================
# PASS 1: FULL CUTOFF-3 TRAINING FINGERPRINTS
# ============================================================

def build_training_fingerprints(columns, features):
    print("\nPASS 1: Building full training fingerprint set")

    training_hashes = set()
    label_counts = Counter()
    training_rows = 0

    for path in [TRAIN_FILE, TEST_FILE]:
        print(f"\nReading {path.name}")

        for chunk in read_chunks(path, columns):
            selected = chunk[
                chunk["source_file"].isin(TRAIN_FILES)
            ]

            if selected.empty:
                continue

            validate_labels(selected)

            hashes = feature_hashes(selected, features)

            training_hashes.update(
                int(value) for value in hashes
            )

            counts = selected["binary_label"].value_counts()

            for label, count in counts.items():
                label_counts[str(label)] += int(count)

            training_rows += len(selected)

    if training_rows != 1_663_220:
        raise ValueError(
            f"Unexpected training row count: {training_rows:,}"
        )

    if not training_hashes:
        raise ValueError("Training fingerprint set is empty.")

    print(f"\nTraining rows: {training_rows:,}")
    print(f"Unique training fingerprints: {len(training_hashes):,}")
    print(f"Training labels: {dict(label_counts)}")

    return training_hashes, training_rows, label_counts


# ============================================================
# PASS 2: DETERMINISTIC STRATIFIED RESERVOIR SAMPLE
# ============================================================

def allocate_sample_sizes(label_counts):
    total = sum(label_counts.values())

    allocation = {
        label: int(round(
            TRAIN_SAMPLE_SIZE * count / total
        ))
        for label, count in label_counts.items()
    }

    # Correct rounding while preserving the requested total.
    difference = TRAIN_SAMPLE_SIZE - sum(allocation.values())

    if difference:
        largest_class = max(
            label_counts,
            key=label_counts.get
        )
        allocation[largest_class] += difference

    for label, size in allocation.items():
        if size > label_counts[label]:
            raise ValueError(
                f"Requested sample exceeds available rows for {label}."
            )

    return allocation


def create_training_sample(columns, features, label_counts):
    print("\nPASS 2: Creating stratified training sample")

    allocation = allocate_sample_sizes(label_counts)

    print(f"Sample allocation: {allocation}")

    rng = np.random.default_rng(RANDOM_STATE)

    reservoirs = {
        label: pd.DataFrame()
        for label in allocation
    }

    priorities = {
        label: np.empty(0, dtype=np.float64)
        for label in allocation
    }

    for path in [TRAIN_FILE, TEST_FILE]:
        print(f"\nSampling from {path.name}")

        for chunk in read_chunks(path, columns):
            selected = chunk[
                chunk["source_file"].isin(TRAIN_FILES)
            ]

            if selected.empty:
                continue

            validate_labels(selected)

            for label, sample_size in allocation.items():
                if sample_size == 0:
                    continue

                class_rows = selected[
                    selected["binary_label"].astype(str) == label
                ]

                if class_rows.empty:
                    continue

                new_priorities = rng.random(len(class_rows))

                combined = pd.concat(
                    [reservoirs[label], class_rows],
                    ignore_index=True
                )

                combined_priorities = np.concatenate([
                    priorities[label],
                    new_priorities,
                ])

                if len(combined) > sample_size:
                    keep = np.argpartition(
                        combined_priorities,
                        -sample_size
                    )[-sample_size:]
                else:
                    keep = np.arange(len(combined))

                reservoirs[label] = (
                    combined.iloc[keep]
                    .reset_index(drop=True)
                )

                priorities[label] = combined_priorities[keep]

    sample = pd.concat(
        [reservoirs[label] for label in allocation],
        ignore_index=True
    )

    if len(sample) != TRAIN_SAMPLE_SIZE:
        raise ValueError(
            f"Expected {TRAIN_SAMPLE_SIZE:,} sampled rows, "
            f"got {len(sample):,}."
        )

    # Deterministic shuffle after class-wise sampling.
    sample = sample.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    sample_labels = sample["binary_label"].astype(str)

    if set(sample_labels.unique()) != {"BENIGN", "ATTACK"}:
        raise ValueError("Training sample does not contain both classes.")

    sample_metadata = sample[
        [
            "source_file",
            "source_row",
            "group_id",
            "normalized_label",
            "binary_label",
        ]
    ].copy()

    sample_metadata.to_csv(
        CONFIG_DIR / "training_sample_metadata.csv",
        index=False
    )

    sample_metadata.groupby(
        ["source_file", "binary_label"]
    ).size().reset_index(
        name="rows"
    ).to_csv(
        CONFIG_DIR / "training_sample_distribution.csv",
        index=False
    )

    X_train = sample[features].astype("float64")
    y_train = sample["binary_label"].map(LABEL_TO_INT).astype("int8")

    print(f"\nFinal sample size: {len(sample):,}")
    print(f"Sample class distribution: {y_train.value_counts().to_dict()}")

    return X_train, y_train


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models():
    logistic_regression = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight=None,
            random_state=RANDOM_STATE,
        )),
    ])

    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    return {
        "logistic_regression": logistic_regression,
        "random_forest_balanced_subsample": random_forest,
    }


# ============================================================
# MODEL TRAINING
# ============================================================

def train_models(X_train, y_train):
    models = build_models()

    print("\nTRAINING MODELS")
    print("=" * 65)

    for name, model in models.items():
        print(f"\nTraining: {name}")

        model.fit(X_train, y_train)

        model_path = MODEL_DIR / f"{name}.joblib"

        joblib.dump(model, model_path, compress=3)

        print(f"Saved model: {model_path}")

    return models


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= DEFAULT_THRESHOLD
    ).astype("int8")

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "rows": int(len(y_true)),
        "threshold": DEFAULT_THRESHOLD,
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "attack_precision": float(
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "attack_recall": float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "attack_f1": float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "false_positive_rate": float(
            fp / (fp + tn) if (fp + tn) else 0.0
        ),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0,
    )

    return metrics, cm, report, predictions


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(name, model, columns, features, training_hashes):
    print("\n" + "=" * 65)
    print(f"EVALUATING: {name}")
    print("=" * 65)

    all_true = []
    all_probabilities = []
    all_overlap = []
    all_capture_files = []
    all_attack_families = []
    all_source_rows = []

    prediction_path = PREDICTION_DIR / f"{name}_predictions.csv"

    if prediction_path.exists():
        prediction_path.unlink()

    first_write = True

    for path in [TRAIN_FILE, TEST_FILE]:
        print(f"\nEvaluating rows from {path.name}")

        for chunk in read_chunks(path, columns):
            selected = chunk[
                chunk["source_file"].isin(EVALUATION_FILES)
            ].copy()

            if selected.empty:
                continue

            validate_labels(selected)

            hashes = feature_hashes(selected, features)

            overlap_mask = np.fromiter(
                (
                    int(value) in training_hashes
                    for value in hashes
                ),
                dtype=bool,
                count=len(hashes),
            )

            X = selected[features].astype("float64")

            probabilities = model.predict_proba(X)[:, 1]

            y_true = (
                selected["binary_label"]
                .map(LABEL_TO_INT)
                .to_numpy(dtype="int8")
            )

            predictions = (
                probabilities >= DEFAULT_THRESHOLD
            ).astype("int8")

            prediction_frame = pd.DataFrame({
                "source_file": selected["source_file"].to_numpy(),
                "source_row": selected["source_row"].to_numpy(),
                "normalized_label": (
                    selected["normalized_label"].to_numpy()
                ),
                "binary_label": selected["binary_label"].to_numpy(),
                "y_true": y_true,
                "attack_probability": probabilities,
                "prediction": predictions,
                "fingerprint_overlap": overlap_mask,
            })

            prediction_frame.to_csv(
                prediction_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            first_write = False

            all_true.append(y_true)
            all_probabilities.append(probabilities)
            all_overlap.append(overlap_mask)

            all_capture_files.extend(
                selected["source_file"].astype(str).tolist()
            )

            all_attack_families.extend(
                selected["normalized_label"].astype(str).tolist()
            )

            all_source_rows.extend(
                selected["source_row"].tolist()
            )

    if not all_true:
        raise ValueError("No evaluation rows were found.")

    y_true = np.concatenate(all_true)
    probabilities = np.concatenate(all_probabilities)
    overlap = np.concatenate(all_overlap)

    capture_files = np.asarray(all_capture_files, dtype=object)
    attack_families = np.asarray(all_attack_families, dtype=object)

    if len(y_true) != 1_160_503:
        raise ValueError(
            f"Unexpected natural evaluation rows: {len(y_true):,}"
        )

    if int(overlap.sum()) != 59_210:
        raise ValueError(
            f"Unexpected overlapping rows: {int(overlap.sum()):,}"
        )

    conditions = {
        "natural_evaluation": np.ones(len(y_true), dtype=bool),
        "overlap_excluded_evaluation": ~overlap,
    }

    results = {}

    for condition_name, mask in conditions.items():
        condition_true = y_true[mask]
        condition_probabilities = probabilities[mask]

        metrics, cm, report, predictions = calculate_metrics(
            condition_true,
            condition_probabilities
        )

        results[condition_name] = metrics

        # Metrics JSON
        with open(
            METRIC_DIR / f"{name}_{condition_name}.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(metrics, file, indent=4)

        # Confusion matrix CSV
        pd.DataFrame(
            cm,
            index=["Actual_BENIGN", "Actual_ATTACK"],
            columns=["Predicted_BENIGN", "Predicted_ATTACK"],
        ).to_csv(
            CONFUSION_DIR / f"{name}_{condition_name}.csv"
        )

        # Classification report CSV
        pd.DataFrame(report).transpose().to_csv(
            REPORT_DIR / f"{name}_{condition_name}_classification_report.csv"
        )

                # Attack-family detection metrics.
        # This is a binary classifier, not a multiclass attack-family classifier.
        # Therefore, family-specific precision and F1 are not reported.
        family_rows = []

        selected_families = attack_families[mask]

        for family in sorted(set(selected_families)):
            family_mask = selected_families == family

            family_true = condition_true[family_mask]
            family_predictions = predictions[family_mask]

            is_attack_family = np.any(family_true == 1)

            detected_attacks = int(
                np.sum(
                    (family_true == 1) &
                    (family_predictions == 1)
                )
            )

            predicted_attacks = int(
                np.sum(family_predictions == 1)
            )

            family_rows.append({
                "attack_family": family,
                "support": int(family_mask.sum()),
                "detected_attacks": detected_attacks,
                "predicted_attacks": predicted_attacks,
                "attack_recall": (
                    float(detected_attacks / np.sum(family_true == 1))
                    if is_attack_family else None
                ),
            })

        pd.DataFrame(family_rows).to_csv(
            METRIC_DIR / f"{name}_{condition_name}_family_metrics.csv",
            index=False,
        )

        print(f"\n{condition_name}")
        print(f"Rows: {metrics['rows']:,}")
        print(f"Accuracy: {metrics['accuracy']:.6f}")
        print(f"Macro F1: {metrics['macro_f1']:.6f}")
        print(f"Attack precision: {metrics['attack_precision']:.6f}")
        print(f"Attack recall: {metrics['attack_recall']:.6f}")
        print(f"Attack F1: {metrics['attack_f1']:.6f}")
        print(f"FPR: {metrics['false_positive_rate']:.6f}")
        print(f"AP: {metrics['average_precision']:.6f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.6f}")

    print(f"\nPredictions saved: {prediction_path}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 6.2 - CUTOFF 3 GENERALIZATION EXPERIMENT")
    print("=" * 70)

    config, columns, features = validate_inputs()

    print("\nConfiguration validated.")
    print(f"Features: {len(features)}")
    print(f"Training scenarios: {len(TRAIN_FILES)}")
    print(f"Evaluation scenarios: {len(EVALUATION_FILES)}")
    print(f"Training sample size: {TRAIN_SAMPLE_SIZE:,}")

    training_hashes, training_rows, label_counts = (
        build_training_fingerprints(columns, features)
    )

    X_train, y_train = create_training_sample(
        columns,
        features,
        label_counts,
    )

    models = train_models(X_train, y_train)

    all_results = {}

    for name, model in models.items():
        all_results[name] = evaluate_model(
            name,
            model,
            columns,
            features,
            training_hashes,
        )

    # Save experiment run details.
    run_config = {
        "experiment_id": config["experiment_id"],
        "experiment_version": config["experiment_version"],
        "cutoff": 3,
        "training_rows": training_rows,
        "training_unique_fingerprints": len(training_hashes),
        "training_sample_size": len(X_train),
        "training_sample_seed": RANDOM_STATE,
        "training_sample_method": (
            "Deterministic class-stratified reservoir sampling"
        ),
        "evaluation_rows": 1_160_503,
        "overlap_rows": 59_210,
        "overlap_excluded_rows": 1_101_293,
        "feature_count": len(features),
        "threshold": DEFAULT_THRESHOLD,
        "threshold_tuning": False,
        "models": list(models.keys()),
        "preprocessing_fit_on_training_sample_only": True,
        "evaluation_data_used_for_threshold_tuning": False,
        "results": all_results,
    }

    with open(
        CONFIG_DIR / "run_config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(run_config, file, indent=4)

    # Compact comparison table.
    comparison_rows = []

    for model_name, conditions in all_results.items():
        for condition_name, metrics in conditions.items():
            comparison_rows.append({
                "model": model_name,
                "evaluation_condition": condition_name,
                **metrics,
            })

    pd.DataFrame(comparison_rows).to_csv(
        METRIC_DIR / "cutoff3_model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("PHASE 6.2 TRAINING AND EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Threshold tuning: NOT performed")
    print("Evaluation-based threshold selection: NOT performed")


if __name__ == "__main__":
    main()
