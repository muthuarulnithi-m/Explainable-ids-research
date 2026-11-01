from __future__ import annotations

import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
TRAIN_PATH = PROCESSED_DIR / "cic_binary_train.csv"
TEST_PATH = PROCESSED_DIR / "cic_binary_test.csv"
REPORT_PATH = REPORT_DIR / "binary_integrity_report.txt"
AUDIT_DB_PATH = REPORT_DIR / "binary_integrity_groups.sqlite"

CHUNK_SIZE = 25_000
METADATA_COLUMNS = [
    "source_file",
    "source_row",
    "group_id",
    "normalized_label",
    "binary_label",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    for path in (TRAIN_PATH, TEST_PATH):
        if not path.is_file():
            fail(f"Required dataset file not found: {path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if AUDIT_DB_PATH.exists():
        AUDIT_DB_PATH.unlink()

    train_header = list(pd.read_csv(TRAIN_PATH, nrows=0).columns)
    test_header = list(pd.read_csv(TEST_PATH, nrows=0).columns)

    if train_header != test_header:
        fail("Train/test column schemas differ or have different column order.")
    if train_header[: len(METADATA_COLUMNS)] != METADATA_COLUMNS:
        fail(
            "Unexpected metadata columns/order. Expected first columns: "
            + ", ".join(METADATA_COLUMNS)
        )
    if len(train_header) - len(METADATA_COLUMNS) != 78:
        fail(
            f"Expected 78 model features, found "
            f"{len(train_header) - len(METADATA_COLUMNS)}."
        )

    feature_columns = train_header[len(METADATA_COLUMNS):]
    forbidden = {
        "label", "binary_label", "normalized_label", "attack_cat", "target"
    }
    leaked = [c for c in feature_columns if c.strip().lower() in forbidden]
    if leaked:
        fail(f"Target-like columns found among model features: {leaked}")

    conn = sqlite3.connect(AUDIT_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE group_presence (
            group_id TEXT PRIMARY KEY,
            split_mask INTEGER NOT NULL
        )
        """
    )
    conn.commit()

    row_counts = Counter()
    class_counts = {"train": Counter(), "test": Counter()}
    multiclass_counts = {"train": Counter(), "test": Counter()}
    missing_counts = Counter()
    nonnumeric_counts = Counter()
    nonfinite_counts = Counter()
    invalid_binary_rows = Counter()
    blank_group_ids = Counter()
    blank_source_rows = Counter()
    duplicate_rows_within_split = Counter()
    numeric_feature_check_rows = Counter()
    errors = []

    try:
        for split_name, path, bit in (
            ("train", TRAIN_PATH, 1),
            ("test", TEST_PATH, 2),
        ):
            print(f"[{split_name.upper()}] Checking {path.name} ...", flush=True)
            for chunk in pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            ):
                if list(chunk.columns) != train_header:
                    fail(f"{path.name}: column schema changed during reading.")

                n = len(chunk)
                row_counts[split_name] += n
                class_counts[split_name].update(chunk["binary_label"].str.strip())
                multiclass_counts[split_name].update(
                    chunk["normalized_label"].str.strip()
                )

                labels = chunk["binary_label"].str.strip().str.upper()
                normalized = chunk["normalized_label"].str.strip().str.upper()
                expected_binary = np.where(normalized == "BENIGN", "BENIGN", "ATTACK")
                invalid = labels.to_numpy() != expected_binary
                invalid_binary_rows[split_name] += int(invalid.sum())

                gids = chunk["group_id"].str.strip()
                blank_group_ids[split_name] += int((gids == "").sum())
                blank_source_rows[split_name] += int(
                    (chunk["source_file"].str.strip() == "").sum()
                )

                unique_gids = set(gids[gids != ""].tolist())
                conn.executemany(
                    """
                    INSERT INTO group_presence(group_id, split_mask)
                    VALUES (?, ?)
                    ON CONFLICT(group_id) DO UPDATE SET
                        split_mask = group_presence.split_mask | excluded.split_mask
                    """,
                    [(gid, bit) for gid in unique_gids],
                )
                conn.commit()

                # Count duplicate group occurrences within each split as diagnostic
                # information only; duplicates are expected when flows share features.
                nonblank_gid_count = int((gids != "").sum())
                duplicate_rows_within_split[split_name] += nonblank_gid_count - len(unique_gids)

                for column in feature_columns:
                    values = chunk[column].str.strip()
                    blank = values.eq("") | values.str.lower().isin(
                        ["nan", "none", "null"]
                    )
                    numeric = pd.to_numeric(values.mask(blank), errors="coerce")
                    missing_counts[column] += int(blank.sum())
                    nonnumeric_counts[column] += int((~blank & numeric.isna()).sum())
                    arr = numeric.to_numpy(dtype="float64", na_value=np.nan)
                    nonfinite_counts[column] += int(np.isinf(arr).sum())
                numeric_feature_check_rows[split_name] += n

        overlap_groups = conn.execute(
            "SELECT COUNT(*) FROM group_presence WHERE split_mask = 3"
        ).fetchone()[0]
        unique_groups = conn.execute(
            "SELECT COUNT(*) FROM group_presence"
        ).fetchone()[0]

    finally:
        conn.close()

    checks = {
        "Train split contains rows": row_counts["train"] > 0,
        "Test split contains rows": row_counts["test"] > 0,
        "Both classes present in train": (
            class_counts["train"]["BENIGN"] > 0
            and class_counts["train"]["ATTACK"] > 0
        ),
        "Both classes present in test": (
            class_counts["test"]["BENIGN"] > 0
            and class_counts["test"]["ATTACK"] > 0
        ),
        "No train/test group_id overlap": overlap_groups == 0,
        "No invalid binary-label mappings": (
            invalid_binary_rows["train"] == 0
            and invalid_binary_rows["test"] == 0
        ),
        "No blank group IDs": (
            blank_group_ids["train"] == 0 and blank_group_ids["test"] == 0
        ),
        "No blank source_file values": (
            blank_source_rows["train"] == 0 and blank_source_rows["test"] == 0
        ),
    }

    report = [
        "CIC-IDS2017 binary dataset integrity report",
        "===========================================",
        f"Train file: {TRAIN_PATH}",
        f"Test file: {TEST_PATH}",
        f"Train rows: {row_counts['train']:,}",
        f"Test rows: {row_counts['test']:,}",
        f"Total rows: {row_counts['train'] + row_counts['test']:,}",
        f"Model feature count: {len(feature_columns)}",
        f"Feature schema matches: {train_header == test_header}",
        f"Distinct group IDs across both splits: {unique_groups:,}",
        f"Group IDs present in both splits: {overlap_groups:,}",
        "",
        "Binary class counts",
        "-------------------",
        f"Train: {dict(class_counts['train'])}",
        f"Test: {dict(class_counts['test'])}",
        "",
        "Normalized multiclass label counts (train)",
        "------------------------------------------",
        str(dict(multiclass_counts["train"])),
        "",
        "Normalized multiclass label counts (test)",
        "-----------------------------------------",
        str(dict(multiclass_counts["test"])),
        "",
        "Data quality",
        "------------",
        f"Invalid binary-label mappings: train={invalid_binary_rows['train']:,}, "
        f"test={invalid_binary_rows['test']:,}",
        f"Blank group IDs: train={blank_group_ids['train']:,}, "
        f"test={blank_group_ids['test']:,}",
        f"Blank source_file values: train={blank_source_rows['train']:,}, "
        f"test={blank_source_rows['test']:,}",
        f"Repeated rows sharing a group ID within each split (diagnostic only): "
        f"train={duplicate_rows_within_split['train']:,}, "
        f"test={duplicate_rows_within_split['test']:,}",
        "",
        "Features with missing/non-numeric/non-finite values",
        "---------------------------------------------------",
    ]

    problematic = []
    for column in feature_columns:
        missing = missing_counts[column]
        nonnumeric = nonnumeric_counts[column]
        nonfinite = nonfinite_counts[column]
        if missing or nonnumeric or nonfinite:
            problematic.append((column, missing, nonnumeric, nonfinite))
    if problematic:
        for column, missing, nonnumeric, nonfinite in sorted(
            problematic, key=lambda x: (-(x[1] + x[2] + x[3]), x[0])
        ):
            report.append(
                f"{column}: missing={missing:,}, "
                f"non_numeric={nonnumeric:,}, infinite={nonfinite:,}"
            )
    else:
        report.append("No missing, non-numeric, or infinite feature values detected.")

    report.extend(["", "Integrity checks", "----------------"])
    for name, passed in checks.items():
        report.append(f"{'PASS' if passed else 'FAIL'}: {name}")

    overall = all(checks.values())
    report.extend(
        [
            "",
            f"OVERALL: {'PASS' if overall else 'FAIL'}",
            "",
            "Notes",
            "-----",
            "Missing feature values are reported, not imputed by this script.",
            "Repeated group IDs within a split are reported as a diagnostic; they are not "
            "automatically errors.",
            "This verifies the saved split files and group_id separation. The group IDs "
            "themselves are 128-bit fingerprints and therefore have a theoretical "
            "collision risk.",
            "This random group-level holdout does not establish temporal or cross-dataset "
            "generalization.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
