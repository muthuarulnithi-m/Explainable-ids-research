from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data" / "processed" / "cic_binary_train.csv"
TEST_FILE = ROOT / "data" / "processed" / "cic_binary_test.csv"

OUTPUT_DIR = ROOT / "experiments" / "generalization" / "temporal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Documented capture-scenario order.
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


def load_metadata(path):
    required = [
        "source_file",
        "group_id",
        "normalized_label",
        "binary_label",
    ]
    return pd.read_csv(path, usecols=required)


def main():
    print("PHASE 6.2 — CAPTURE-DAY CUT-OFF AUDIT")
    print("=" * 60)

    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError("Required processed CIC CSV file is missing.")

    print("Loading metadata columns...")
    train = load_metadata(TRAIN_FILE)
    test = load_metadata(TEST_FILE)

    # Validate binary target labels before combining the metadata.
    for name, df in [
        ("train", train),
        ("test", test),
    ]:
        unexpected = set(df["binary_label"].dropna().unique()) - {
            "BENIGN",
            "ATTACK",
        }
        if unexpected:
            raise ValueError(
                f"Unexpected binary labels in {name}: {unexpected}"
            )

    df = pd.concat([train, test], ignore_index=True)

    print("Total rows:", len(df))
    print("Unique groups:", df["group_id"].nunique())

    actual_files = set(df["source_file"].dropna().unique())
    expected_files = set(CAPTURE_ORDER)
    missing = expected_files - actual_files
    unexpected = actual_files - expected_files

    if missing or unexpected:
        raise ValueError(
            f"Capture-file mismatch. Missing={missing}; "
            f"Unexpected={unexpected}"
        )

    order_map = {
        filename: index
        for index, filename in enumerate(CAPTURE_ORDER)
    }
    df["capture_order"] = df["source_file"].map(order_map)

    file_distribution = (
        df.groupby(["capture_order", "source_file", "normalized_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("capture_order")
    )
    file_distribution.to_csv(
        OUTPUT_DIR / "capture_file_class_distribution.csv",
        index=False,
    )

    group_file_counts = df.groupby("group_id")["source_file"].nunique()
    spanning_groups = set(group_file_counts[group_file_counts > 1].index)
    print("Groups spanning multiple capture files:", len(spanning_groups))

    # Binary class distribution by file (retained as an audit calculation).
    binary_distribution = (
        df.groupby(["capture_order", "source_file", "binary_label"])
        .size()
        .unstack(fill_value=0)
    )

    results = []

    for cut in range(1, len(CAPTURE_ORDER)):
        train_files = CAPTURE_ORDER[:cut]
        eval_files = CAPTURE_ORDER[cut:]

        train_part = df[df["source_file"].isin(train_files)]
        eval_part = df[df["source_file"].isin(eval_files)]

        train_groups = set(train_part["group_id"].unique())
        eval_groups = set(eval_part["group_id"].unique())
        shared_groups = train_groups.intersection(eval_groups)

        train_counts = train_part["binary_label"].value_counts()
        eval_counts = eval_part["binary_label"].value_counts()

        train_attack_families = sorted(
            train_part.loc[
                train_part["normalized_label"] != "BENIGN",
                "normalized_label",
            ].unique().tolist()
        )
        eval_attack_families = sorted(
            eval_part.loc[
                eval_part["normalized_label"] != "BENIGN",
                "normalized_label",
            ].unique().tolist()
        )

        crossing_groups = group_file_counts.index[
            group_file_counts.index.isin(shared_groups)
        ]

        row = {
            "cutoff_after": train_files[-1],
            "training_files": " ; ".join(train_files),
            "evaluation_files": " ; ".join(eval_files),
            "training_rows": len(train_part),
            "evaluation_rows": len(eval_part),
            "training_benign": int(train_counts.get("BENIGN", 0)),
            "training_attack": int(train_counts.get("ATTACK", 0)),
            "evaluation_benign": int(eval_counts.get("BENIGN", 0)),
            "evaluation_attack": int(eval_counts.get("ATTACK", 0)),
            "training_attack_families": len(train_attack_families),
            "evaluation_attack_families": len(eval_attack_families),
            "shared_group_count": len(shared_groups),
            "crossing_group_count": len(crossing_groups),
            "evaluation_attack_family_names": " ; ".join(eval_attack_families),
        }
        results.append(row)

        print(f"\nCut-off {cut}: after {train_files[-1]}")
        print(
            f"  Train rows={len(train_part):,}, "
            f"evaluation rows={len(eval_part):,}"
        )
        print(
            f"  Train BENIGN={row['training_benign']:,}, "
            f"ATTACK={row['training_attack']:,}"
        )
        print(
            f"  Eval BENIGN={row['evaluation_benign']:,}, "
            f"ATTACK={row['evaluation_attack']:,}"
        )
        print(f"  Shared groups={len(shared_groups):,}")
        print(f"  Attack families in evaluation={len(eval_attack_families)}")

    result_df = pd.DataFrame(results)
    result_path = OUTPUT_DIR / "capture_day_candidate_cutoffs.csv"
    result_df.to_csv(result_path, index=False)

    config = {
        "experiment": "CIC capture-day ordered binary evaluation",
        "capture_order": CAPTURE_ORDER,
        "ordering_basis": "Documented capture scenario order, not timestamps",
        "temporal_claim": (
            "Approximate capture-day ordered generalization; "
            "not true timestamp-based temporal validation"
        ),
        "source_files": [
            str(TRAIN_FILE.relative_to(ROOT)),
            str(TEST_FILE.relative_to(ROOT)),
        ],
        "rows": len(df),
        "unique_groups": int(df["group_id"].nunique()),
        "groups_spanning_capture_files": len(spanning_groups),
        "candidate_cutoffs": len(results),
        "feature_vector_overlap": (
            "Not calculated in this metadata-only audit. "
            "Requires exact 78-feature row comparison."
        ),
        "model_training_performed": False,
        "threshold_tuning_performed": False,
    }

    config_path = OUTPUT_DIR / "capture_day_audit_config.json"
    config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("Saved:", result_path)
    print("Saved:", config_path)
    print("Model training: NOT performed")
    print("Threshold tuning: NOT performed")


if __name__ == "__main__":
    main()
