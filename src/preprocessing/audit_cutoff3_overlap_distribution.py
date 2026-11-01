
from pathlib import Path
from collections import Counter
import json

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data" / "processed" / "cic_binary_train.csv"
TEST_FILE = ROOT / "data" / "processed" / "cic_binary_test.csv"

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "generalization"
    / "temporal"
    / "cutoff3_overlap_distribution"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

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

TRAIN_FILES = set(CAPTURE_ORDER[:CUTOFF])
EVALUATION_FILES = set(CAPTURE_ORDER[CUTOFF:])

METADATA_COLUMNS = {
    "source_file",
    "source_row",
    "group_id",
    "normalized_label",
    "binary_label",
}


# ============================================================
# VALIDATION
# ============================================================

for path in [TRAIN_FILE, TEST_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

if CUTOFF <= 0 or CUTOFF >= len(CAPTURE_ORDER):
    raise ValueError("Invalid cutoff configuration.")


# ============================================================
# FEATURE SCHEMA
# ============================================================

print("=" * 65)
print("CUTOFF 3 OVERLAP DISTRIBUTION AUDIT")
print("=" * 65)

columns = pd.read_csv(TRAIN_FILE, nrows=0).columns.tolist()

if set(columns) != set(pd.read_csv(TEST_FILE, nrows=0).columns):
    raise ValueError("Train/test column schemas do not match.")

feature_columns = [
    column for column in columns
    if column not in METADATA_COLUMNS
]

if len(feature_columns) != 78:
    raise ValueError(
        f"Expected 78 features, found {len(feature_columns)}."
    )

required_columns = METADATA_COLUMNS

missing = required_columns - set(columns)

if missing:
    raise ValueError(f"Missing metadata columns: {sorted(missing)}")

print(f"Feature columns: {len(feature_columns)}")
print(f"Training capture files: {len(TRAIN_FILES)}")
print(f"Evaluation capture files: {len(EVALUATION_FILES)}")


# ============================================================
# HASH FUNCTION
# Must match the previous feature-overlap audit.
# ============================================================

def calculate_hashes(frame):
    features = frame[feature_columns].astype("float64")

    return pd.util.hash_pandas_object(
        features,
        index=False,
        categorize=False
    ).to_numpy(dtype="uint64")


# ============================================================
# PASS 1: BUILD TRAINING FINGERPRINT SET
# ============================================================

training_hashes = set()
training_rows = 0

print("\nPASS 1: Building cutoff-3 training fingerprints")

for file_path in [TRAIN_FILE, TEST_FILE]:

    print(f"\nReading {file_path.name}")

    for chunk in pd.read_csv(
        file_path,
        usecols=columns,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        selected = chunk[
            chunk["source_file"].isin(TRAIN_FILES)
        ]

        if selected.empty:
            continue

        hashes = calculate_hashes(selected)

        training_hashes.update(int(value) for value in hashes)

        training_rows += len(selected)

print("\nTraining fingerprint audit complete.")
print(f"Training rows: {training_rows:,}")
print(f"Unique training fingerprints: {len(training_hashes):,}")


# ============================================================
# PASS 2: CLASSIFY EVALUATION ROWS
# ============================================================

overlap_counts = Counter()
non_overlap_counts = Counter()

overlap_by_binary = Counter()
non_overlap_by_binary = Counter()

overlap_by_file = Counter()
non_overlap_by_file = Counter()

overlap_by_family = Counter()
non_overlap_by_family = Counter()

evaluation_rows = 0
overlap_rows = 0
non_overlap_rows = 0

print("\nPASS 2: Auditing cutoff-3 evaluation rows")

for file_path in [TRAIN_FILE, TEST_FILE]:

    print(f"\nReading {file_path.name}")

    for chunk in pd.read_csv(
        file_path,
        usecols=columns,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        selected = chunk[
            chunk["source_file"].isin(EVALUATION_FILES)
        ].copy()

        if selected.empty:
            continue

        hashes = calculate_hashes(selected)

        is_overlap = [
            int(value) in training_hashes
            for value in hashes
        ]

        selected["overlap_status"] = is_overlap

        evaluation_rows += len(selected)

        overlap_rows += sum(is_overlap)
        non_overlap_rows += len(is_overlap) - sum(is_overlap)

        # ----------------------------------------------------
        # Binary label distribution
        # ----------------------------------------------------

        for label, count in selected[
            selected["overlap_status"]
        ]["binary_label"].value_counts().items():

            overlap_by_binary[str(label)] += int(count)

        for label, count in selected[
            ~selected["overlap_status"]
        ]["binary_label"].value_counts().items():

            non_overlap_by_binary[str(label)] += int(count)

        # ----------------------------------------------------
        # Capture file distribution
        # ----------------------------------------------------

        overlap_frame = selected[selected["overlap_status"]]
        non_overlap_frame = selected[~selected["overlap_status"]]

        for (source_file, label), count in (
            overlap_frame.groupby(
                ["source_file", "binary_label"]
            ).size().items()
        ):
            overlap_by_file[(str(source_file), str(label))] += int(count)

        for (source_file, label), count in (
            non_overlap_frame.groupby(
                ["source_file", "binary_label"]
            ).size().items()
        ):
            non_overlap_by_file[(str(source_file), str(label))] += int(count)

        # ----------------------------------------------------
        # Attack family distribution
        # ----------------------------------------------------

        for (family, label), count in (
            overlap_frame.groupby(
                ["normalized_label", "binary_label"]
            ).size().items()
        ):
            overlap_by_family[(str(family), str(label))] += int(count)

        for (family, label), count in (
            non_overlap_frame.groupby(
                ["normalized_label", "binary_label"]
            ).size().items()
        ):
            non_overlap_by_family[(str(family), str(label))] += int(count)

print("\nEvaluation fingerprint audit complete.")


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_counter(counter, columns, filename):
    rows = []

    for key, count in counter.items():
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(zip(columns, key))
        row["rows"] = count
        rows.append(row)

    frame = pd.DataFrame(rows)

    if not frame.empty:
        frame = frame.sort_values(
            "rows",
            ascending=False
        )

    frame.to_csv(OUTPUT_DIR / filename, index=False)

    return frame


# ============================================================
# SAVE REPORTS
# ============================================================

save_counter(
    overlap_by_binary,
    ["binary_label"],
    "overlap_by_binary_label.csv"
)

save_counter(
    non_overlap_by_binary,
    ["binary_label"],
    "non_overlap_by_binary_label.csv"
)

save_counter(
    overlap_by_file,
    ["source_file", "binary_label"],
    "overlap_by_capture_file.csv"
)

save_counter(
    non_overlap_by_file,
    ["source_file", "binary_label"],
    "non_overlap_by_capture_file.csv"
)

save_counter(
    overlap_by_family,
    ["normalized_label", "binary_label"],
    "overlap_by_attack_family.csv"
)

save_counter(
    non_overlap_by_family,
    ["normalized_label", "binary_label"],
    "non_overlap_by_attack_family.csv"
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "cutoff": CUTOFF,
    "cutoff_description": "After Wednesday",
    "training_capture_files": CAPTURE_ORDER[:CUTOFF],
    "evaluation_capture_files": CAPTURE_ORDER[CUTOFF:],
    "feature_count": len(feature_columns),
    "training_rows": training_rows,
    "unique_training_fingerprints": len(training_hashes),
    "evaluation_rows": evaluation_rows,
    "overlap_rows": overlap_rows,
    "non_overlap_rows": non_overlap_rows,
    "overlap_percentage": (
        overlap_rows / evaluation_rows * 100
        if evaluation_rows else 0
    ),
    "non_overlap_percentage": (
        non_overlap_rows / evaluation_rows * 100
        if evaluation_rows else 0
    ),
    "hash_method": "pandas.hash_pandas_object uint64",
    "model_training_performed": False,
    "threshold_tuning_performed": False,
}

with open(
    OUTPUT_DIR / "audit_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)


# ============================================================
# FINAL CONSOLE REPORT
# ============================================================

print("\n" + "=" * 65)
print("FINAL CUTOFF-3 OVERLAP SUMMARY")
print("=" * 65)

print(f"Evaluation rows:       {evaluation_rows:,}")
print(f"Overlapping rows:      {overlap_rows:,}")
print(f"Non-overlapping rows:  {non_overlap_rows:,}")

print(
    f"Overlap percentage:    "
    f"{summary['overlap_percentage']:.4f}%"
)

print(
    f"Non-overlap percentage: "
    f"{summary['non_overlap_percentage']:.4f}%"
)

print("\nOVERLAP BY BINARY LABEL")

for label, count in sorted(overlap_by_binary.items()):
    print(f"{label:12s}: {count:,}")

print("\nNON-OVERLAP BY BINARY LABEL")

for label, count in sorted(non_overlap_by_binary.items()):
    print(f"{label:12s}: {count:,}")

print("\nATTACK FAMILY SUPPORT AFTER OVERLAP EXCLUSION")

for (family, label), count in sorted(
    non_overlap_by_family.items()
):
    if label == "ATTACK":
        print(f"{family:40s}: {count:,}")

print("\nReports saved to:")
print(OUTPUT_DIR)

print("\nModel training: NOT performed")
print("Threshold tuning: NOT performed")
