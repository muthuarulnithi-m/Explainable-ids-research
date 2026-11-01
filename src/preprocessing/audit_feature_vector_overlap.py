
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILES = [
    ROOT / "data" / "processed" / "cic_binary_train.csv",
    ROOT / "data" / "processed" / "cic_binary_test.csv",
]

OUTPUT_DIR = ROOT / "experiments" / "generalization" / "temporal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

CHUNK_SIZE = 100_000


def get_feature_columns(path):
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    features = [c for c in columns if c not in METADATA_COLUMNS]

    if len(features) != 78:
        raise ValueError(
            f"Expected 78 model features, found {len(features)}."
        )

    sample = pd.read_csv(
        path,
        nrows=5,
        usecols=features,
    )

    non_numeric = sample.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Non-numeric model features found: {non_numeric}"
        )

    return features


def main():
    print("PHASE 6.2 — FEATURE VECTOR OVERLAP AUDIT")
    print("=" * 65)

    for path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(path)

    features = get_feature_columns(INPUT_FILES[0])

    other_features = get_feature_columns(INPUT_FILES[1])

    if features != other_features:
        raise ValueError(
            "Train and test feature columns or their order differ."
        )

    print("Model features:", len(features))
    print("Chunk size:", f"{CHUNK_SIZE:,}")
    print("Fingerprint method: pandas 64-bit row hash")
    print("Feature values are read as float64.")

    usecols = ["source_file"] + features
    dtype_map = {column: "float64" for column in features}
    dtype_map["source_file"] = "string"

    # Store compact fingerprints separately for each capture file.
    fingerprints = {
        filename: []
        for filename in CAPTURE_ORDER
    }

    total_rows = 0

    for path in INPUT_FILES:
        print(f"\nReading: {path.name}")

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                path,
                usecols=usecols,
                dtype=dtype_map,
                chunksize=CHUNK_SIZE,
            ),
            start=1,
        ):
            unknown_files = (
                set(chunk["source_file"].dropna().unique())
                - set(CAPTURE_ORDER)
            )

            if unknown_files:
                raise ValueError(
                    f"Unexpected capture files: {unknown_files}"
                )

            if chunk["source_file"].isna().any():
                raise ValueError("Missing source_file values found.")

            for filename, part in chunk.groupby(
                "source_file",
                sort=False,
            ):
                row_hashes = pd.util.hash_pandas_object(
                    part[features],
                    index=False,
                    categorize=False,
                ).to_numpy(dtype=np.uint64)

                fingerprints[filename].append(row_hashes)

            total_rows += len(chunk)

            print(
                f"  Chunk {chunk_number}: "
                f"{len(chunk):,} rows processed"
            )

    # Merge chunk fingerprints into one array per capture file.
    file_hashes = {}

    for filename in CAPTURE_ORDER:
        if not fingerprints[filename]:
            raise ValueError(
                f"No rows found for capture file: {filename}"
            )

        file_hashes[filename] = np.concatenate(
            fingerprints[filename]
        )

        print(
            f"{filename}: "
            f"{len(file_hashes[filename]):,} fingerprints"
        )

    print("\nTotal rows fingerprinted:", f"{total_rows:,}")

    # Evaluate every candidate cutoff.
    results = []

    for cut in range(1, len(CAPTURE_ORDER)):
        train_files = CAPTURE_ORDER[:cut]
        eval_files = CAPTURE_ORDER[cut:]

        train_hashes = np.concatenate(
            [file_hashes[name] for name in train_files]
        )

        eval_hashes = np.concatenate(
            [file_hashes[name] for name in eval_files]
        )

        unique_train = np.unique(train_hashes)
        unique_eval = np.unique(eval_hashes)

        # Evaluation rows whose fingerprint exists in training.
        eval_row_overlap = np.isin(
            eval_hashes,
            unique_train,
            assume_unique=False,
        )

        # Unique evaluation fingerprints also present in training.
        unique_overlap = np.intersect1d(
            unique_eval,
            unique_train,
            assume_unique=True,
        )

        overlapping_eval_rows = int(eval_row_overlap.sum())
        unique_overlapping_vectors = len(unique_overlap)

        eval_rows = len(eval_hashes)

        row_overlap_pct = (
            overlapping_eval_rows / eval_rows * 100
            if eval_rows else 0.0
        )

        unique_vector_overlap_pct = (
            unique_overlapping_vectors / len(unique_eval) * 100
            if len(unique_eval) else 0.0
        )

        row = {
            "cutoff_after": train_files[-1],
            "training_rows": len(train_hashes),
            "evaluation_rows": eval_rows,
            "unique_training_vectors": len(unique_train),
            "unique_evaluation_vectors": len(unique_eval),
            "overlapping_evaluation_rows": overlapping_eval_rows,
            "evaluation_row_overlap_pct": row_overlap_pct,
            "unique_overlapping_vectors": unique_overlapping_vectors,
            "unique_evaluation_vector_overlap_pct": (
                unique_vector_overlap_pct
            ),
        }

        results.append(row)

        print(f"\nCutoff {cut}: after {train_files[-1]}")
        print(
            f"  Training rows: {len(train_hashes):,}"
        )
        print(
            f"  Evaluation rows: {eval_rows:,}"
        )
        print(
            f"  Evaluation rows with matching training vector: "
            f"{overlapping_eval_rows:,} ({row_overlap_pct:.4f}%)"
        )
        print(
            f"  Unique overlapping evaluation vectors: "
            f"{unique_overlapping_vectors:,}"
        )
        print(
            f"  Unique evaluation-vector overlap: "
            f"{unique_vector_overlap_pct:.4f}%"
        )

        # Release large temporary arrays before next cutoff.
        del train_hashes
        del eval_hashes
        del unique_train
        del unique_eval
        del eval_row_overlap
        del unique_overlap

    result_df = pd.DataFrame(results)

    output_path = (
        OUTPUT_DIR / "capture_day_feature_vector_overlap.csv"
    )

    result_df.to_csv(output_path, index=False)

    config = {
        "experiment": "CIC capture-day feature-vector overlap audit",
        "capture_order": CAPTURE_ORDER,
        "feature_count": len(features),
        "feature_columns": features,
        "feature_dtype": "float64",
        "fingerprint_method": (
            "pandas.util.hash_pandas_object, 64-bit row fingerprints"
        ),
        "comparison": (
            "Evaluation row fingerprints compared with all training "
            "row fingerprints for each candidate capture-file cutoff."
        ),
        "hash_collision_limitation": (
            "64-bit fingerprints are probabilistic identifiers, not "
            "a mathematical proof of feature-vector equality. "
            "A matching fingerprint is treated as a matching vector "
            "for this screening audit."
        ),
        "total_rows": total_rows,
        "chunk_size": CHUNK_SIZE,
        "model_training_performed": False,
        "threshold_tuning_performed": False,
    }

    config_path = (
        OUTPUT_DIR / "feature_vector_overlap_config.json"
    )

    config_path.write_text(
        json.dumps(config, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("FEATURE VECTOR OVERLAP AUDIT COMPLETE")
    print("Saved:", output_path)
    print("Saved:", config_path)
    print("Model training: NOT performed")
    print("Threshold tuning: NOT performed")


if __name__ == "__main__":
    main()
