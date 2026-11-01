
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data/processed/cic_binary_train.csv"
TEST_FILE = ROOT / "data/processed/cic_binary_test.csv"

PREDICTION_DIR = (
    ROOT / "experiments/generalization/temporal/cutoff3/predictions"
)

OUTPUT_DIR = (
    ROOT / "experiments/generalization/temporal/cutoff3/group_audit"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["source_file", "source_row"]

PREDICTION_FILES = {
    "logistic_regression": (
        PREDICTION_DIR / "logistic_regression_predictions.csv"
    ),
    "random_forest": (
        PREDICTION_DIR / "random_forest_balanced_subsample_predictions.csv"
    ),
}


def load_metadata(path):
    return pd.read_csv(
        path,
        usecols=[
            "source_file",
            "source_row",
            "group_id",
            "binary_label",
        ],
        dtype={
            "source_file": "string",
            "group_id": "string",
            "binary_label": "string",
        },
    )


def main():
    print("Loading original metadata...")

    metadata = pd.concat(
        [
            load_metadata(TRAIN_FILE),
            load_metadata(TEST_FILE),
        ],
        ignore_index=True,
    )

    metadata["source_row"] = pd.to_numeric(
        metadata["source_row"], errors="raise"
    ).astype("int64")

    print(f"Metadata rows: {len(metadata):,}")

    duplicate_keys = metadata.duplicated(KEYS, keep=False).sum()
    print(f"Duplicate metadata keys: {duplicate_keys:,}")

    if duplicate_keys:
        raise ValueError(
            "source_file + source_row is not unique in combined metadata."
        )

    if metadata["group_id"].isna().any():
        raise ValueError("Missing group_id values found.")

    metadata = metadata[
        KEYS + ["group_id", "binary_label"]
    ]

    summary = []

    for model_name, prediction_path in PREDICTION_FILES.items():
        print(f"\nChecking {model_name}...")

        predictions = pd.read_csv(
            prediction_path,
            dtype={"source_file": "string"},
        )

        predictions["source_row"] = pd.to_numeric(
            predictions["source_row"], errors="raise"
        ).astype("int64")

        duplicate_prediction_keys = predictions.duplicated(
            KEYS, keep=False
        ).sum()

        if duplicate_prediction_keys:
            raise ValueError(
                f"{model_name}: duplicate prediction keys found."
            )

        merged = predictions.merge(
            metadata,
            on=KEYS,
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        unmatched = int((merged["_merge"] != "both").sum())

        if unmatched:
            raise ValueError(
                f"{model_name}: {unmatched:,} predictions "
                "could not be mapped to group_id."
            )

        label_mismatches = int(
            (
                merged["binary_label_x"]
                != merged["binary_label_y"]
            ).sum()
        )

        if label_mismatches:
            raise ValueError(
                f"{model_name}: binary label mismatches found."
            )

        merged = merged.drop(columns=["_merge"])
        merged = merged.rename(
            columns={
                "group_id": "group_id",
                "binary_label_y": "metadata_binary_label",
            }
        )

        output_file = OUTPUT_DIR / (
            f"{model_name}_predictions_with_groups.csv"
        )

        merged.to_csv(output_file, index=False)

        group_count = merged["group_id"].nunique()
        largest_group = int(
            merged.groupby("group_id").size().max()
        )

        print(f"Prediction rows: {len(merged):,}")
        print(f"Mapped rows: {len(merged):,}")
        print(f"Unmatched rows: {unmatched:,}")
        print(f"Label mismatches: {label_mismatches:,}")
        print(f"Unique evaluation groups: {group_count:,}")
        print(f"Largest evaluation group: {largest_group:,}")
        print(f"Saved: {output_file}")

        summary.append({
            "model": model_name,
            "prediction_rows": len(merged),
            "unique_groups": group_count,
            "largest_group_size": largest_group,
            "unmatched_rows": unmatched,
            "label_mismatches": label_mismatches,
        })

    summary_df = pd.DataFrame(summary)
    summary_path = OUTPUT_DIR / "group_mapping_audit_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if summary_df["prediction_rows"].nunique() != 1:
        raise ValueError("Model prediction row counts differ.")

    if summary_df["unique_groups"].nunique() != 1:
        raise ValueError("Model evaluation group counts differ.")

    print("\nGROUP MAPPING AUDIT PASSED")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
