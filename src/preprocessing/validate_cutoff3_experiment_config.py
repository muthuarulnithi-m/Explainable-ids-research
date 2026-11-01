
from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = (
    ROOT
    / "experiments"
    / "generalization"
    / "temporal"
    / "cutoff3_experiment_config.json"
)

TRAIN_FILE = ROOT / "data" / "processed" / "cic_binary_train.csv"
TEST_FILE = ROOT / "data" / "processed" / "cic_binary_test.csv"


def main():
    print("=" * 65)
    print("CUTOFF-3 CONFIGURATION VALIDATION")
    print("=" * 65)

    for path in [CONFIG_FILE, TRAIN_FILE, TEST_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    train_columns = pd.read_csv(TRAIN_FILE, nrows=0).columns.tolist()
    test_columns = pd.read_csv(TEST_FILE, nrows=0).columns.tolist()

    if train_columns != test_columns:
        raise AssertionError("Train/test schemas differ.")

    metadata = set(
        config["dataset"]["metadata_excluded_from_model"]
    )

    actual_features = [
        column for column in train_columns
        if column not in metadata
    ]

    expected_features = config["dataset"]["feature_columns"]

    assert actual_features == expected_features, (
        "Feature schema or feature order has changed."
    )

    actual_hash = hashlib.sha256(
        "\n".join(actual_features).encode("utf-8")
    ).hexdigest()

    assert actual_hash == config["dataset"]["feature_schema_sha256"], (
        "Feature schema SHA256 mismatch."
    )

    assert len(actual_features) == 78
    assert config["split"]["cutoff"] == 3
    assert config["split"]["training_capture_files"] == (
        config["split"]["capture_order"][:3]
    )
    assert config["split"]["evaluation_capture_files"] == (
        config["split"]["capture_order"][3:]
    )

    assert config["labels"]["benign_value"] == "BENIGN"
    assert config["labels"]["attack_value"] == "ATTACK"

    assert (
        config["evaluation_conditions"]["primary"]["expected_rows"]
        == 1_160_503
    )

    assert (
        config["evaluation_conditions"]["sensitivity"]["expected_rows"]
        == 1_101_293
    )

    assert config["training_performed"] is False
    assert config["threshold_tuning_performed"] is False

    print("PASS: Dataset files exist")
    print("PASS: Train/test schemas match")
    print("PASS: All 78 feature names and their order match")
    print("PASS: Feature schema SHA256 matches")
    print("PASS: Cutoff and capture-file order are consistent")
    print("PASS: Binary label mapping is consistent")
    print("PASS: Evaluation row counts match audited results")
    print("PASS: No training or threshold tuning recorded")

    print("\nCONFIGURATION VALIDATION: PASSED")
    print("Model training: NOT performed")


if __name__ == "__main__":
    main()
