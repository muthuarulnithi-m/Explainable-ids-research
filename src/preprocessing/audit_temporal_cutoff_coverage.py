
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FILES = [
    ROOT / "data" / "processed" / "cic_binary_train.csv",
    ROOT / "data" / "processed" / "cic_binary_test.csv",
]

OUTPUT_DIR = ROOT / "experiments" / "generalization" / "temporal"

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

CUTOFF = 3


def main():
    print("PHASE 6.2 — CUTOFF COVERAGE AUDIT")
    print("=" * 65)

    frames = []

    for path in FILES:
        df = pd.read_csv(
            path,
            usecols=[
                "source_file",
                "group_id",
                "normalized_label",
                "binary_label",
            ],
        )
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    train_files = set(CAPTURE_ORDER[:CUTOFF])
    eval_files = set(CAPTURE_ORDER[CUTOFF:])

    train = df[df["source_file"].isin(train_files)].copy()
    evaluation = df[df["source_file"].isin(eval_files)].copy()

    train_families = set(
        train.loc[
            train["normalized_label"] != "BENIGN",
            "normalized_label",
        ].unique()
    )

    eval_families = set(
        evaluation.loc[
            evaluation["normalized_label"] != "BENIGN",
            "normalized_label",
        ].unique()
    )

    unseen = eval_families - train_families
    seen = eval_families.intersection(train_families)

    print("\nSelected cutoff: after", CAPTURE_ORDER[CUTOFF - 1])

    print("\nTraining attack families:")
    for family in sorted(train_families):
        count = int(
            (train["normalized_label"] == family).sum()
        )
        print(f"  {family}: {count:,}")

    print("\nEvaluation attack families:")
    for family in sorted(eval_families):
        count = int(
            (evaluation["normalized_label"] == family).sum()
        )
        status = "SEEN" if family in train_families else "UNSEEN"
        print(f"  {family}: {count:,} [{status}]")

    print("\nSeen evaluation families:", len(seen))
    print("Unseen evaluation families:", len(unseen))

    print("\nUnseen families:")
    for family in sorted(unseen):
        print(" ", family)

    # Check whether one group maps to multiple distinct feature identities
    # indirectly through its labels and capture-file membership.
    group_summary = (
        df.groupby("group_id")
        .agg(
            capture_files=("source_file", "nunique"),
            labels=("normalized_label", "nunique"),
            rows=("group_id", "size"),
        )
    )

    print("\nGroup identity diagnostics:")
    print("Total groups:", len(group_summary))
    print(
        "Groups spanning capture files:",
        int((group_summary["capture_files"] > 1).sum()),
    )
    print(
        "Groups containing multiple labels:",
        int((group_summary["labels"] > 1).sum()),
    )

    print("\nGroup size distribution:")
    print(group_summary["rows"].describe().to_string())

    print("\nModel training: NOT performed")
    print("Threshold tuning: NOT performed")


if __name__ == "__main__":
    main()
