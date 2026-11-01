from __future__ import annotations

import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "cic_ids2017"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
DB_PATH = REPORT_DIR / "binary_pipeline_groups.sqlite"
REPORT_PATH = REPORT_DIR / "binary_pipeline_report.txt"

CHUNK_SIZE = 50_000
TEST_PERCENT = 20

# Flow/capture identifiers are excluded. Destination Port is intentionally retained.
EXCLUDED_FEATURES = {"flow id", "source ip", "source port", "destination ip", "timestamp"}

LABEL_ALIASES = {
    "web attack \ufffd brute force": "Web Attack - Brute Force",
    "web attack \ufffd xss": "Web Attack - XSS",
    "web attack \ufffd sql injection": "Web Attack - SQL Injection",
    "web attack - brute force": "Web Attack - Brute Force",
    "web attack - xss": "Web Attack - XSS",
    "web attack - sql injection": "Web Attack - SQL Injection",
}
HASH_KEY_A = "0123456789abcdef"
HASH_KEY_B = "fedcba9876543210"


def normalize_label(value: object) -> str:
    label = str(value).strip()
    return LABEL_ALIASES.get(label.lower(), label)


def binary_label(label: str) -> str:
    return "BENIGN" if label.strip().upper() == "BENIGN" else "ATTACK"


def get_schema(columns: list[str]) -> tuple[str, list[str]]:
    cleaned = [str(c).strip() for c in columns]
    labels = [c for c in cleaned if c.lower() == "label"]
    if len(labels) != 1:
        raise ValueError(f"Expected exactly one Label column; found {labels!r}")
    label_col = labels[0]
    features = [
        c for c in cleaned
        if c != label_col and c.strip().lower() not in EXCLUDED_FEATURES
    ]
    if not features:
        raise ValueError("No feature columns remain after excluding identifiers.")
    return label_col, features


def read_chunks(path: Path, features: list[str], label_col: str):
    wanted = set(features + [label_col])
    # A callable handles CIC headers such as ' Label' with leading whitespace.
    for chunk in pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
        usecols=lambda column: str(column).strip() in wanted,
    ):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        if label_col not in chunk.columns:
            raise ValueError(f"{path.name}: target column {label_col!r} not found.")
        yield chunk


def canonical_features(chunk: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    numeric = chunk[features].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.astype("float64")


def feature_fingerprints(numeric_features: pd.DataFrame) -> list[str]:
    # Two independent 64-bit hashes form a 128-bit group fingerprint.
    # Theoretical collisions are possible, but extremely unlikely.
    h1 = hash_pandas_object(
        numeric_features, index=False, hash_key=HASH_KEY_A
    ).to_numpy(dtype="uint64", copy=False)
    h2 = hash_pandas_object(
        numeric_features, index=False, hash_key=HASH_KEY_B
    ).to_numpy(dtype="uint64", copy=False)
    return [f"{int(a):016x}{int(b):016x}" for a, b in zip(h1, h2)]


def setup_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE feature_groups (
            gid TEXT PRIMARY KEY,
            label_mask INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            split TEXT
        )
        """
    )


def pass_one(files, features, label_col, conn):
    total_rows = 0
    label_counts = Counter()

    for path in files:
        print(f"[Pass 1/2] Auditing {path.name} ...", flush=True)
        file_rows = 0

        for chunk in read_chunks(path, features, label_col):
            labels = chunk[label_col].map(normalize_label)
            binaries = labels.map(binary_label)
            numeric = canonical_features(chunk, features)
            gids = feature_fingerprints(numeric)
            masks = binaries.map({"BENIGN": 1, "ATTACK": 2}).to_numpy(dtype="int64")

            label_counts.update(labels.tolist())
            total_rows += len(chunk)
            file_rows += len(chunk)

            # Collapse duplicate fingerprints within this chunk before SQLite upsert.
            per_group = {}
            for gid, mask in zip(gids, masks):
                if gid not in per_group:
                    per_group[gid] = [int(mask), 1]
                else:
                    per_group[gid][0] |= int(mask)
                    per_group[gid][1] += 1

            conn.executemany(
                """
                INSERT INTO feature_groups(gid, label_mask, row_count)
                VALUES (?, ?, ?)
                ON CONFLICT(gid) DO UPDATE SET
                    label_mask = feature_groups.label_mask | excluded.label_mask,
                    row_count = feature_groups.row_count + excluded.row_count
                """,
                [(gid, v[0], v[1]) for gid, v in per_group.items()],
            )
            conn.commit()

        print(f"    rows scanned: {file_rows:,}", flush=True)

    return total_rows, label_counts


def assign_group_split(conn):
    # Deterministic hash-bucket split at group level, so identical feature fingerprints
    # cannot be assigned to different splits.
    cursor = conn.execute("SELECT gid FROM feature_groups WHERE label_mask IN (1, 2)")
    batch = []
    train_groups = 0
    test_groups = 0

    while True:
        rows = cursor.fetchmany(50_000)
        if not rows:
            break
        for (gid,) in rows:
            bucket = int(gid[-8:], 16) % 100
            split = "test" if bucket < TEST_PERCENT else "train"
            if split == "train":
                train_groups += 1
            else:
                test_groups += 1
            batch.append((split, gid))

        conn.executemany("UPDATE feature_groups SET split=? WHERE gid=?", batch)
        conn.commit()
        batch.clear()

    return train_groups, test_groups


def pass_two(files, features, label_col, conn):
    train_path = PROCESSED_DIR / "cic_binary_train.csv"
    test_path = PROCESSED_DIR / "cic_binary_test.csv"
    for output in (train_path, test_path):
        if output.exists():
            output.unlink()

    row_counts = Counter()
    class_counts = {"train": Counter(), "test": Counter()}
    excluded_ambiguous = 0
    first_write = {"train": True, "test": True}
    conn.execute("CREATE TEMP TABLE chunk_ids (gid TEXT PRIMARY KEY)")

    for path in files:
        print(f"[Pass 2/2] Writing splits from {path.name} ...", flush=True)
        row_number = 0

        for chunk in read_chunks(path, features, label_col):
            n = len(chunk)
            labels = chunk[label_col].map(normalize_label)
            numeric = canonical_features(chunk, features)
            gids = feature_fingerprints(numeric)

            conn.execute("DELETE FROM chunk_ids")
            conn.executemany(
                "INSERT OR IGNORE INTO chunk_ids(gid) VALUES (?)",
                [(gid,) for gid in set(gids)],
            )

            group_info = {
                gid: (label_mask, split)
                for gid, label_mask, split in conn.execute(
                    """
                    SELECT g.gid, g.label_mask, g.split
                    FROM feature_groups g
                    JOIN chunk_ids c ON c.gid = g.gid
                    """
                ).fetchall()
            }

            masks = np.fromiter(
                (group_info[gid][0] for gid in gids), dtype=np.int8
            )
            splits = [group_info[gid][1] for gid in gids]

            keep = np.isin(masks, [1, 2])
            excluded_ambiguous += int((~keep).sum())

            for split_name in ("train", "test"):
                selected = keep & np.fromiter(
                    (split == split_name for split in splits), dtype=bool, count=n
                )
                if not selected.any():
                    continue

                selected_labels = labels.loc[selected]
                out = numeric.loc[selected].copy()
                out.insert(0, "group_id", np.asarray(gids, dtype=object)[selected])
                out.insert(
                    0, "source_row",
                    np.arange(row_number + 2, row_number + n + 2)[selected],
                )
                out.insert(0, "source_file", path.name)
                out["normalized_label"] = selected_labels.to_numpy()
                out["binary_label"] = selected_labels.map(binary_label).to_numpy()
                out = out[
                    ["source_file", "source_row", "group_id", "normalized_label",
                     "binary_label"] + features
                ]

                output = train_path if split_name == "train" else test_path
                out.to_csv(
                    output,
                    mode="w" if first_write[split_name] else "a",
                    header=first_write[split_name],
                    index=False,
                )
                first_write[split_name] = False
                row_counts[split_name] += len(out)
                class_counts[split_name].update(out["binary_label"].tolist())

            row_number += n

    if first_write["train"] or first_write["test"]:
        raise RuntimeError("A split CSV is empty; stop and inspect group/class coverage.")

    return train_path, test_path, row_counts, class_counts, excluded_ambiguous


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {RAW_DIR}")
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    headers = []
    for path in files:
        columns = [str(c).strip() for c in pd.read_csv(path, nrows=0).columns]
        headers.append((path, columns))

    label_col, features = get_schema(headers[0][1])
    expected = set(features + [label_col])
    for path, columns in headers[1:]:
        other_label, other_features = get_schema(columns)
        if other_label != label_col or set(other_features + [other_label]) != expected:
            raise ValueError(f"Feature schema mismatch in {path.name}")

    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)

    try:
        setup_db(conn)
        total_rows, label_counts = pass_one(files, features, label_col, conn)

        group_summary = {
            int(mask): (int(groups), int(rows))
            for mask, groups, rows in conn.execute(
                "SELECT label_mask, COUNT(*), SUM(row_count) "
                "FROM feature_groups GROUP BY label_mask"
            ).fetchall()
        }
        consistent_groups = sum(group_summary.get(mask, (0, 0))[0] for mask in (1, 2))
        consistent_rows = sum(group_summary.get(mask, (0, 0))[1] for mask in (1, 2))
        ambiguous_groups, ambiguous_rows = group_summary.get(3, (0, 0))

        train_groups, test_groups = assign_group_split(conn)
        train_path, test_path, row_counts, class_counts, excluded_rows = pass_two(
            files, features, label_col, conn
        )

        # Each fingerprint has exactly one split assignment in SQLite, and pass two
        # writes rows using that assignment. Verify class coverage in both output files.
        for split_name in ("train", "test"):
            if not class_counts[split_name].get("BENIGN", 0) or not class_counts[split_name].get("ATTACK", 0):
                raise RuntimeError(
                    f"Class coverage failed in {split_name}: {dict(class_counts[split_name])}"
                )

        overlapping_assignments = conn.execute(
            """
            SELECT COUNT(*) FROM feature_groups
            WHERE split IS NOT NULL AND split NOT IN ('train', 'test')
            """
        ).fetchone()[0]
        if overlapping_assignments:
            raise RuntimeError("Unexpected group split assignment value.")

        report = [
            "CIC-IDS2017 binary preprocessing report",
            "========================================",
            f"Raw directory: {RAW_DIR}",
            f"Source files: {len(files)}",
            f"Total input rows scanned: {total_rows:,}",
            f"Raw normalized multiclass label counts: {dict(label_counts)}",
            "",
            "Feature policy",
            "--------------",
            f"Model feature count: {len(features)}",
            f"Excluded identifier columns when present: {sorted(EXCLUDED_FEATURES)}",
            "Destination Port retained as a model feature.",
            "Target columns are excluded from the model features.",
            "Non-numeric and non-finite feature values are coerced to missing values.",
            "",
            "Grouping and ambiguity",
            "----------------------",
            "Group key: concatenated pair of independent 64-bit pandas hashes over numeric",
            "identifier-excluded features. A theoretical fingerprint collision is possible.",
            f"Consistent feature groups: {consistent_groups:,}",
            f"Rows in consistent groups retained: {consistent_rows:,}",
            f"Ambiguous groups containing both BENIGN and ATTACK: {ambiguous_groups:,}",
            f"Rows in ambiguous groups excluded from the primary binary split: {ambiguous_rows:,}",
            "Ambiguous rows remain in the raw dataset and are not silently relabeled.",
            "",
            "Split policy",
            "------------",
            f"Approximate split: {100 - TEST_PERCENT}% train / {TEST_PERCENT}% test by feature group.",
            "Deterministic group-level assignment based on the feature fingerprint.",
            f"Consistent groups assigned to train: {train_groups:,}",
            f"Consistent groups assigned to test: {test_groups:,}",
            f"Output train rows: {row_counts['train']:,}",
            f"Output test rows: {row_counts['test']:,}",
            f"Train binary class counts: {dict(class_counts['train'])}",
            f"Test binary class counts: {dict(class_counts['test'])}",
            "Train/test feature-group overlap: prevented by assigning each group once in SQLite.",
            "Class coverage check: PASS (BENIGN and ATTACK present in both splits).",
            "",
            "Outputs",
            "-------",
            f"Train CSV: {train_path}",
            f"Test CSV: {test_path}",
            f"SQLite grouping audit: {DB_PATH}",
            f"Report: {REPORT_PATH}",
            "",
            "Limitations",
            "-----------",
            "This is a random group-level holdout, not a temporal or cross-dataset test.",
            "No imputer, scaler, resampler, feature selector, or model is fitted in this script.",
            "The split ratio is approximate because groups, rather than individual rows, are split.",
        ]
        REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
        print("\n".join(report), flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
