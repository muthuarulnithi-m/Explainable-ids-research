
from collections import Counter
from pathlib import Path
import sqlite3

import pandas as pd


DATA_DIR = Path("data/raw/cic_ids2017")
REPORT_DIR = Path("reports")
CHUNK_SIZE = 50_000

LABEL_COLUMN_NAME = "label"

LABEL_MAPPING = {
    "Web Attack \uFFFD Brute Force": "Web Attack - Brute Force",
    "Web Attack \uFFFD XSS": "Web Attack - XSS",
    "Web Attack \uFFFD Sql Injection": "Web Attack - SQL Injection",
}

# Exclude identifiers from the feature fingerprint so that
# otherwise-identical traffic features can be compared.
IDENTIFIER_COLUMNS = {
    "flow id",
    "source ip",
    "source port",
    "destination ip",
    "destination port",
    "timestamp",
}


def main():
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    db_path = REPORT_DIR / "split_feasibility_audit.sqlite"
    report_path = REPORT_DIR / "split_feasibility_audit.txt"

    file_class_counts = {}
    file_rows = {}
    schema_reference = None

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_hashes (
                hash TEXT PRIMARY KEY,
                first_label TEXT NOT NULL,
                occurrences INTEGER NOT NULL,
                conflicting_labels INTEGER NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS hash_files (
                hash TEXT NOT NULL,
                filename TEXT NOT NULL,
                occurrences INTEGER NOT NULL,
                PRIMARY KEY (hash, filename)
            )
        """)

        # A fresh audit starts from an empty database.
        conn.execute("DELETE FROM feature_hashes")
        conn.execute("DELETE FROM hash_files")

        for path in files:
            print(f"Auditing: {path.name}")

            class_counts = Counter()
            row_count = 0

            for chunk in pd.read_csv(
                path,
                chunksize=CHUNK_SIZE,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8",
            ):
                chunk.columns = [str(c).strip() for c in chunk.columns]

                label_matches = [
                    c for c in chunk.columns
                    if c.lower() == LABEL_COLUMN_NAME
                ]

                if len(label_matches) != 1:
                    raise ValueError(
                        f"Expected one Label column in {path.name}; "
                        f"found {label_matches}"
                    )

                label_col = label_matches[0]

                if schema_reference is None:
                    schema_reference = set(chunk.columns)
                elif set(chunk.columns) != schema_reference:
                    raise ValueError(
                        f"Column schema mismatch in {path.name}"
                    )

                labels = chunk[label_col].replace(LABEL_MAPPING)
                class_counts.update(labels.tolist())
                row_count += len(chunk)

                feature_cols = [
                    c for c in chunk.columns
                    if c.lower() != LABEL_COLUMN_NAME
                    and c.lower() not in IDENTIFIER_COLUMNS
                ]

                if not feature_cols:
                    raise ValueError("No feature columns available.")

                # Hashes are used to identify candidate duplicate
                # feature rows. Flagged matches can be verified later.
                hashes = pd.util.hash_pandas_object(
                    chunk[feature_cols],
                    index=False,
                ).astype("uint64")

                hash_records = [
                    (str(int(h)), str(label))
                    for h, label in zip(hashes, labels)
                ]

                conn.executemany("""
                    INSERT INTO feature_hashes
                        (hash, first_label, occurrences, conflicting_labels)
                    VALUES (?, ?, 1, 0)
                    ON CONFLICT(hash) DO UPDATE SET
                        occurrences = occurrences + 1,
                        conflicting_labels = CASE
                            WHEN feature_hashes.first_label
                                 != excluded.first_label
                            THEN 1
                            ELSE feature_hashes.conflicting_labels
                        END
                """, hash_records)

                file_records = [
                    (str(int(h)), path.name)
                    for h in hashes
                ]

                conn.executemany("""
                    INSERT INTO hash_files (hash, filename, occurrences)
                    VALUES (?, ?, 1)
                    ON CONFLICT(hash, filename) DO UPDATE SET
                        occurrences = occurrences + 1
                """, file_records)

                conn.commit()

            file_class_counts[path.name] = class_counts
            file_rows[path.name] = row_count

            print(f"  Rows: {row_count:,}")
            for label, count in sorted(class_counts.items()):
                print(f"  {label}: {count:,}")

        duplicate_groups = conn.execute("""
            SELECT COUNT(*)
            FROM feature_hashes
            WHERE occurrences > 1
        """).fetchone()[0]

        repeated_rows = conn.execute("""
            SELECT COALESCE(SUM(occurrences - 1), 0)
            FROM feature_hashes
            WHERE occurrences > 1
        """).fetchone()[0]

        cross_file_groups = conn.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT hash
                FROM hash_files
                GROUP BY hash
                HAVING COUNT(DISTINCT filename) > 1
            )
        """).fetchone()[0]

        conflicting_groups = conn.execute("""
            SELECT COUNT(*)
            FROM feature_hashes
            WHERE conflicting_labels = 1
        """).fetchone()[0]

        total_rows = sum(file_rows.values())

        with report_path.open("w", encoding="utf-8") as report:
            report.write("CIC-IDS2017 SPLIT-FEASIBILITY AUDIT\n")
            report.write("=" * 40 + "\n\n")
            report.write(f"Total rows: {total_rows:,}\n")
            report.write(f"Files audited: {len(files)}\n\n")

            report.write("PER-FILE CLASS COVERAGE\n")
            report.write("-" * 40 + "\n")

            for filename, counts in file_class_counts.items():
                report.write(f"\n{filename} ({file_rows[filename]:,} rows)\n")
                for label, count in sorted(counts.items()):
                    report.write(f"  {label}: {count:,}\n")

            report.write("\nFEATURE-FINGERPRINT SUMMARY\n")
            report.write("-" * 40 + "\n")
            report.write(
                f"Distinct fingerprints occurring multiple times: "
                f"{duplicate_groups:,}\n"
            )
            report.write(
                f"Repeated row occurrences beyond the first: "
                f"{repeated_rows:,}\n"
            )
            report.write(
                f"Fingerprints appearing in multiple files: "
                f"{cross_file_groups:,}\n"
            )
            report.write(
                f"Fingerprints with conflicting labels: "
                f"{conflicting_groups:,}\n"
            )

            report.write(
                "\nNote: 64-bit feature hashes identify candidate duplicates; "
                "they are not proof of exact row equality. Identifier columns "
                "are excluded. Verify flagged groups before deciding splits. "
                "No raw CSV files were modified.\n"
            )

    print("\nAUDIT COMPLETED")
    print(f"Report: {report_path}")
    print(f"Fingerprint database: {db_path}")
    print("Raw CSV files were not modified.")


if __name__ == "__main__":
    main()
