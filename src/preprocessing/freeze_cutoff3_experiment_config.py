
from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data" / "processed" / "cic_binary_train.csv"
TEST_FILE = ROOT / "data" / "processed" / "cic_binary_test.csv"

TEMPORAL_DIR = (
    ROOT / "experiments" / "generalization" / "temporal"
)

OVERLAP_DIR = TEMPORAL_DIR / "cutoff3_overlap_distribution"

OUTPUT_FILE = TEMPORAL_DIR / "cutoff3_experiment_config.json"

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

METADATA_COLUMNS = {
    "source_file",
    "source_row",
    "group_id",
    "normalized_label",
    "binary_label",
}


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def main():
    require_file(TRAIN_FILE)
    require_file(TEST_FILE)

    require_file(OVERLAP_DIR / "audit_summary.json")
    require_file(OVERLAP_DIR / "overlap_by_binary_label.csv")
    require_file(OVERLAP_DIR / "non_overlap_by_binary_label.csv")
    require_file(OVERLAP_DIR / "overlap_by_attack_family.csv")
    require_file(OVERLAP_DIR / "non_overlap_by_attack_family.csv")

    train_columns = pd.read_csv(TRAIN_FILE, nrows=0).columns.tolist()
    test_columns = pd.read_csv(TEST_FILE, nrows=0).columns.tolist()

    if train_columns != test_columns:
        raise ValueError("Train/test column order or schema differs.")

    feature_columns = [
        column for column in train_columns
        if column not in METADATA_COLUMNS
    ]

    if len(feature_columns) != 78:
        raise ValueError(
            f"Expected 78 features, found {len(feature_columns)}."
        )

    schema_text = "\n".join(feature_columns)
    schema_hash = hashlib.sha256(
        schema_text.encode("utf-8")
    ).hexdigest()

    with open(
        OVERLAP_DIR / "audit_summary.json",
        "r",
        encoding="utf-8"
    ) as file:
        overlap_summary = json.load(file)

    if overlap_summary["cutoff"] != 3:
        raise ValueError("Overlap audit is not for cutoff 3.")

    if overlap_summary["feature_count"] != 78:
        raise ValueError("Overlap audit feature count is inconsistent.")

    if overlap_summary["evaluation_rows"] != 1_160_503:
        raise ValueError("Unexpected evaluation row count.")

    if overlap_summary["overlap_rows"] != 59_210:
        raise ValueError("Unexpected overlap row count.")

    if overlap_summary["non_overlap_rows"] != 1_101_293:
        raise ValueError("Unexpected non-overlap row count.")

    overlap_binary = pd.read_csv(
        OVERLAP_DIR / "overlap_by_binary_label.csv"
    )
    non_overlap_binary = pd.read_csv(
        OVERLAP_DIR / "non_overlap_by_binary_label.csv"
    )

    overlap_labels = dict(zip(
        overlap_binary["binary_label"].astype(str),
        overlap_binary["rows"].astype(int)
    ))

    non_overlap_labels = dict(zip(
        non_overlap_binary["binary_label"].astype(str),
        non_overlap_binary["rows"].astype(int)
    ))

    if overlap_labels != {"BENIGN": 59_210}:
        raise ValueError(
            f"Unexpected overlap label distribution: {overlap_labels}"
        )

    expected_non_overlap = {
        "BENIGN": 811_213,
        "ATTACK": 290_080,
    }

    if non_overlap_labels != expected_non_overlap:
        raise ValueError(
            "Unexpected non-overlap label distribution: "
            f"{non_overlap_labels}"
        )

    config = {
        "experiment_id": "CIC_CUTOFF3_CAPTURE_SCENARIO_ORDERED",
        "experiment_version": "1.0",
        "status": "FROZEN_BEFORE_TRAINING",

        "research_question": (
            "How well does a binary intrusion detector trained on "
            "earlier CIC-IDS2017 capture scenarios detect attack "
            "families absent from its training scenarios?"
        ),

        "dataset": {
            "name": "CIC-IDS2017",
            "training_files": "cic_binary_train.csv",
            "evaluation_files": "cic_binary_test.csv",
            "feature_count": 78,
            "feature_columns": feature_columns,
            "feature_schema_sha256": schema_hash,
            "metadata_excluded_from_model": sorted(
                METADATA_COLUMNS
            ),
        },

        "split": {
            "method": "capture_scenario_order",
            "cutoff": 3,
            "cutoff_description": "After Wednesday",
            "capture_order": CAPTURE_ORDER,
            "training_capture_files": CAPTURE_ORDER[:3],
            "evaluation_capture_files": CAPTURE_ORDER[3:],
            "is_true_temporal_validation": False,
            "description": (
                "Capture-scenario order is used because reliable "
                "timestamps are unavailable. This is not claimed "
                "to be true temporal validation."
            ),
        },

        "labels": {
            "target_column": "binary_label",
            "benign_value": "BENIGN",
            "attack_value": "ATTACK",
            "attack_family_column": "normalized_label",
        },

        "evaluation_conditions": {
            "primary": {
                "name": "natural_evaluation",
                "include_all_evaluation_rows": True,
                "expected_rows": 1_160_503,
                "expected_benign": 870_423,
                "expected_attack": 290_080,
            },
            "sensitivity": {
                "name": "overlap_excluded_evaluation",
                "exclude_evaluation_fingerprints_seen_in_training": True,
                "expected_rows": 1_101_293,
                "expected_benign": 811_213,
                "expected_attack": 290_080,
                "expected_excluded_rows": 59_210,
                "excluded_label_distribution": {
                    "BENIGN": 59_210,
                    "ATTACK": 0,
                },
            },
        },

        "overlap_audit": {
            "method": "pandas.hash_pandas_object",
            "hash_width_bits": 64,
            "feature_count": 78,
            "training_unique_fingerprints": 1_527_133,
            "evaluation_overlapping_rows": 59_210,
            "evaluation_overlap_percentage": 5.1021,
            "shared_groups": 23_658,
            "shared_group_and_fingerprint_counts_are_not_independent": True,
            "limitation": (
                "Fingerprint matching is a 64-bit hash-based audit, "
                "not a mathematical proof of exact row equality."
            ),
        },

        "metrics": [
            "accuracy",
            "macro_f1",
            "attack_precision",
            "attack_recall",
            "attack_f1",
            "false_positive_rate",
            "average_precision",
        ],

        "threshold_policy": {
            "threshold_tuning": False,
            "policy": (
                "Use the previously specified default binary "
                "classification threshold. Do not optimize it on "
                "the evaluation data."
            ),
        },

        "training_policy": {
            "training_data": "Rows from the first three capture files",
            "evaluation_data": "Rows from the remaining five capture files",
            "fit_preprocessing_on_training_only": True,
            "evaluation_data_used_for_model_selection": False,
        },

        "limitations": [
            (
                "This is capture-scenario-ordered evaluation, "
                "not true temporal validation."
            ),
            (
                "The binary test dataset was previously used for "
                "baseline model comparison; this is not a completely "
                "untouched independent holdout."
            ),
            (
                "Evaluation attack families are absent from the "
                "training capture scenarios; this is binary detection "
                "of unseen attack families, not multiclass recognition "
                "of unseen families."
            ),
            (
                "Fingerprint overlap was identified using 64-bit hashes."
            ),
            (
                "Overlap exclusion removes BENIGN rows and changes "
                "the evaluation class distribution."
            ),
            (
                "Rare attack-family supports must be reported and "
                "interpreted cautiously."
            ),
        ],

        "training_performed": False,
        "threshold_tuning_performed": False,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=4)

    print("=" * 65)
    print("CUTOFF-3 EXPERIMENT CONFIGURATION FROZEN")
    print("=" * 65)
    print(f"Feature count: {len(feature_columns)}")
    print(f"Feature schema SHA256: {schema_hash}")
    print("Cutoff: After Wednesday")
    print("Natural evaluation rows: 1,160,503")
    print("Overlap-excluded rows: 1,101,293")
    print("Overlap rows excluded: 59,210")
    print("\nSaved:")
    print(OUTPUT_FILE)
    print("\nModel training: NOT performed")
    print("Threshold tuning: NOT performed")


if __name__ == "__main__":
    main()
