
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "cic_binary_train.csv"
OUTPUT_DIR = ROOT / "experiments" / "multiclass_cic"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

TARGET_COLUMN = "normalized_label"
GROUP_COLUMN = "group_id"

METADATA_COLUMNS = [
    "source_file",
    "source_row",
    "group_id",
    "normalized_label",
    "binary_label",
]

# --------------------------------------------------
# Main preparation
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("=" * 65)
    print("MULTICLASS CIC DATA PREPARATION")
    print("=" * 65)

    # Read only the columns required for group splitting.
    print("\n[1/6] Reading target and group columns...")

    data = pd.read_csv(
        DATA_PATH,
        usecols=[TARGET_COLUMN, GROUP_COLUMN],
    )

    if data.empty:
        raise ValueError("The training dataset is empty.")

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing labels.")

    if data[GROUP_COLUMN].isna().any():
        raise ValueError("Group column contains missing values.")

    print(f"Total rows: {len(data):,}")
    print(f"Unique groups: {data[GROUP_COLUMN].nunique():,}")

    # --------------------------------------------------
    # Group-level split
    # --------------------------------------------------

    print("\n[2/6] Creating group-level training/validation split...")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    train_positions, validation_positions = next(
        splitter.split(
            data,
            groups=data[GROUP_COLUMN],
        )
    )

    train_data = data.iloc[train_positions]
    validation_data = data.iloc[validation_positions]

    # --------------------------------------------------
    # Leakage and class checks
    # --------------------------------------------------

    print("\n[3/6] Checking group leakage and class coverage...")

    train_groups = set(train_data[GROUP_COLUMN].unique())
    validation_groups = set(validation_data[GROUP_COLUMN].unique())

    shared_groups = train_groups.intersection(validation_groups)

    if shared_groups:
        raise RuntimeError(
            f"Group leakage detected: {len(shared_groups)} shared groups."
        )

    all_classes = set(data[TARGET_COLUMN].unique())
    train_classes = set(train_data[TARGET_COLUMN].unique())
    validation_classes = set(validation_data[TARGET_COLUMN].unique())

    missing_train = all_classes - train_classes
    missing_validation = all_classes - validation_classes

    if missing_train or missing_validation:
        raise RuntimeError(
            "Class coverage failure.\n"
            f"Missing from training: {sorted(missing_train)}\n"
            f"Missing from validation: {sorted(missing_validation)}"
        )

    # --------------------------------------------------
    # Save split positions
    # --------------------------------------------------

    print("\n[4/6] Saving split position arrays...")

    np.save(
        OUTPUT_DIR / "train_positions.npy",
        train_positions,
    )

    np.save(
        OUTPUT_DIR / "validation_positions.npy",
        validation_positions,
    )

    # --------------------------------------------------
    # Save class distributions
    # --------------------------------------------------

    print("\n[5/6] Saving class distributions...")

    train_counts = (
        train_data[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    validation_counts = (
        validation_data[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    distribution = pd.DataFrame({
        "class_name": sorted(all_classes),
    })

    distribution["train_count"] = (
        distribution["class_name"]
        .map(train_counts)
        .fillna(0)
        .astype(int)
    )

    distribution["validation_count"] = (
        distribution["class_name"]
        .map(validation_counts)
        .fillna(0)
        .astype(int)
    )

    distribution.to_csv(
        OUTPUT_DIR / "class_distribution.csv",
        index=False,
    )

    # --------------------------------------------------
    # Save reproducibility configuration
    # --------------------------------------------------

    config = {
        "experiment": "CIC multiclass intrusion detection",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "metadata_columns_excluded": METADATA_COLUMNS,
        "feature_count": 78,
        "class_count": len(all_classes),
        "classes": sorted(all_classes),
        "total_rows": int(len(data)),
        "training_rows": int(len(train_positions)),
        "validation_rows": int(len(validation_positions)),
        "validation_fraction": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "split_method": "GroupShuffleSplit",
        "shared_groups": len(shared_groups),
        "training_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "train_positions_file": "train_positions.npy",
        "validation_positions_file": "validation_positions.npy",
        "missing_classes_train": sorted(missing_train),
        "missing_classes_validation": sorted(missing_validation),
    }

    with open(
        OUTPUT_DIR / "multiclass_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=4)

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n[6/6] Preparation completed.")

    print("\nCLASS DISTRIBUTION")
    print(distribution.to_string(index=False))

    print("\nSPLIT SUMMARY")
    print(f"Training rows:      {len(train_positions):,}")
    print(f"Validation rows:    {len(validation_positions):,}")
    print(f"Training groups:    {len(train_groups):,}")
    print(f"Validation groups:  {len(validation_groups):,}")
    print(f"Shared groups:      {len(shared_groups)}")
    print(f"Classes:            {len(all_classes)}")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
