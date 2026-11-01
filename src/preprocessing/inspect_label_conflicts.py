
import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw/cic_ids2017")
DB_PATH = Path("reports/split_feasibility_audit.sqlite")
OUTPUT_PATH = Path("reports/conflicting_fingerprint_rows.csv")
CHUNK_SIZE = 50_000

LABEL_MAPPING = {
    "Web Attack \uFFFD Brute Force": "Web Attack - Brute Force",
    "Web Attack \uFFFD XSS": "Web Attack - XSS",
    "Web Attack \uFFFD Sql Injection": "Web Attack - SQL Injection",
}

IDENTIFIER_COLUMNS = {
    "flow id",
    "source ip",
    "source port",
    "destination ip",
    "destination port",
    "timestamp",
}


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Audit database missing: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        conflict_hashes = {
            row[0]
            for row in conn.execute(
                "SELECT hash FROM feature_hashes "
                "WHERE conflicting_labels = 1"
            )
        }

    print(f"Flagged fingerprints: {len(conflict_hashes):,}")

    if not conflict_hashes:
        print("No conflicting fingerprints to inspect.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.unlink(missing_ok=True)

    total_matches = 0

    for path in sorted(DATA_DIR.glob("*.csv")):
        rows_seen = 0

        for chunk in pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        ):
            chunk.columns = [str(c).strip() for c in chunk.columns]

            label_cols = [
                c for c in chunk.columns if c.lower() == "label"
            ]
            if len(label_cols) != 1:
                raise ValueError(f"Unexpected Label column in {path.name}")

            label_col = label_cols[0]
            normalized_labels = chunk[label_col].replace(LABEL_MAPPING)

            feature_cols = [
                c for c in chunk.columns
                if c.lower() != "label"
                and c.lower() not in IDENTIFIER_COLUMNS
            ]

            hashes = pd.util.hash_pandas_object(
                chunk[feature_cols],
                index=False,
            ).astype("uint64").astype(str)

            mask = hashes.isin(conflict_hashes)

            if mask.any():
                matched = chunk.loc[mask].copy()
                matched.insert(0, "_source_file", path.name)
                matched.insert(
                    1,
                    "_source_row",
                    [rows_seen + i + 2 for i, keep in enumerate(mask) if keep],
                )
                matched.insert(
                    2,
                    "_fingerprint",
                    hashes.loc[mask].tolist(),
                )
                matched["_normalized_label"] = (
                    normalized_labels.loc[mask].tolist()
                )

                matched.to_csv(
                    OUTPUT_PATH,
                    mode="a",
                    header=not OUTPUT_PATH.exists()
                    or OUTPUT_PATH.stat().st_size == 0,
                    index=False,
                    encoding="utf-8",
                )
                total_matches += len(matched)

            rows_seen += len(chunk)

        print(f"Scanned: {path.name}")

    print(f"\nRows matching flagged fingerprints: {total_matches:,}")
    print(f"Detailed records: {OUTPUT_PATH}")
    print("Raw datasets were not modified.")


if __name__ == "__main__":
    main()
