
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FILES = [
    ROOT / "data" / "processed" / "cic_binary_train.csv",
    ROOT / "data" / "processed" / "cic_binary_test.csv",
]

OUTPUT_DIR = ROOT / "experiments" / "generalization" / "temporal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA = {
    "source_file",
    "source_row",
    "group_id",
    "normalized_label",
    "binary_label",
}

CHUNK_SIZE = 100_000


def main():
    print("GROUP ID vs FEATURE FINGERPRINT IDENTITY AUDIT")
    print("=" * 65)

    header = pd.read_csv(FILES[0], nrows=0)
    features = [c for c in header.columns if c not in METADATA]

    if len(features) != 78:
        raise ValueError(f"Expected 78 features, got {len(features)}")

    pair_frames = []

    for path in FILES:
        print(f"\nProcessing {path.name}")

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                path,
                usecols=["group_id"] + features,
                chunksize=CHUNK_SIZE,
            ),
            start=1,
        ):
            if chunk["group_id"].isna().any():
                raise ValueError("Missing group_id values found.")

            hashes = pd.util.hash_pandas_object(
                chunk[features].astype("float64"),
                index=False,
                categorize=False,
            ).to_numpy(dtype="uint64")

            pairs = pd.DataFrame({
                "group_id": chunk["group_id"].to_numpy(),
                "feature_hash": hashes,
            })

            pair_frames.append(pairs.drop_duplicates())

            print(
                f"  Chunk {chunk_number}: "
                f"{len(chunk):,} rows"
            )

    print("\nCombining unique group/hash relationships...")

    pairs = pd.concat(pair_frames, ignore_index=True).drop_duplicates()

    # Does one group_id map to multiple feature fingerprints?
    hashes_per_group = pairs.groupby("group_id")[
        "feature_hash"
    ].nunique()

    # Does one feature fingerprint map to multiple group IDs?
    groups_per_hash = pairs.groupby("feature_hash")[
        "group_id"
    ].nunique()

    multi_hash_groups = hashes_per_group[hashes_per_group > 1]
    multi_group_hashes = groups_per_hash[groups_per_hash > 1]

    print("\nRESULTS")
    print("-" * 65)
    print("Unique group/hash pairs:", f"{len(pairs):,}")
    print("Unique group IDs:", f"{len(hashes_per_group):,}")
    print("Unique feature hashes:", f"{len(groups_per_hash):,}")

    print(
        "Groups mapping to multiple feature hashes:",
        f"{len(multi_hash_groups):,}",
    )

    print(
        "Feature hashes mapping to multiple group IDs:",
        f"{len(multi_group_hashes):,}",
    )

    if len(multi_hash_groups) == 0 and len(multi_group_hashes) == 0:
        print("\nObserved relationship: one-to-one")
    elif len(multi_hash_groups) == 0:
        print("\nObserved relationship: each group maps to one hash,")
        print("but some hashes map to multiple groups.")
    elif len(multi_group_hashes) == 0:
        print("\nObserved relationship: each hash maps to one group,")
        print("but some groups map to multiple hashes.")
    else:
        print("\nObserved relationship: neither mapping is one-to-one.")

    output_path = OUTPUT_DIR / "group_feature_identity_summary.csv"

    summary = pd.DataFrame([
        {
            "unique_group_hash_pairs": len(pairs),
            "unique_group_ids": len(hashes_per_group),
            "unique_feature_hashes": len(groups_per_hash),
            "groups_with_multiple_hashes": len(multi_hash_groups),
            "hashes_with_multiple_groups": len(multi_group_hashes),
        }
    ])

    summary.to_csv(output_path, index=False)

    print("\nSaved:", output_path)
    print("Model training: NOT performed")
    print("Threshold tuning: NOT performed")


if __name__ == "__main__":
    main()
