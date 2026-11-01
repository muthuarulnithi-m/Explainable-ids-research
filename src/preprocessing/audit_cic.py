
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cic_ids2017"
CHUNK_SIZE = 100_000


def audit_file(file_path: Path) -> dict:
    print("\n" + "=" * 78)
    print(f"AUDITING: {file_path.name}")
    print("=" * 78)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # Read only the header first.
    header = pd.read_csv(file_path, nrows=0, encoding="utf-8-sig")
    original_columns = list(header.columns)
    columns = [str(col).strip() for col in original_columns]

    print(f"Columns: {len(columns)}")
    print(f"Original column names: {original_columns}")

    if len(set(columns)) != len(columns):
        print("WARNING: Duplicate column names after trimming whitespace.")

    label_columns = [
        col for col in columns if col.strip().lower() == "label"
    ]

    if len(label_columns) != 1:
        print("WARNING: Could not identify exactly one Label column.")
        label_column = None
    else:
        label_column = label_columns[0]

    row_count = 0
    missing_counts = Counter()
    label_counts = Counter()
    dtype_names = {}
    infinite_counts = Counter()
    chunk_duplicate_count = 0
    unnamed_columns = [
        col for col in columns
        if col.strip().lower().startswith("unnamed:")
    ]

    # Read in chunks instead of loading the entire file into memory.
    reader = pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    for chunk in reader:
        chunk.columns = [str(col).strip() for col in chunk.columns]
        row_count += len(chunk)

        # Count missing values.
        for col, count in chunk.isna().sum().items():
            missing_counts[col] += int(count)

        # Count labels after trimming whitespace.
        if label_column is not None and label_column in chunk.columns:
            labels = (
                chunk[label_column]
                .astype("string")
                .str.strip()
                .fillna("<MISSING>")
            )
            label_counts.update(labels.value_counts().to_dict())

        # Record observed dtypes and infinite numeric values.
        for col in chunk.columns:
            dtype_names.setdefault(col, set()).add(str(chunk[col].dtype))

        numeric = chunk.select_dtypes(include=[np.number])
        if not numeric.empty:
            inf = np.isinf(numeric.to_numpy()).sum(axis=0)
            for col, count in zip(numeric.columns, inf):
                infinite_counts[col] += int(count)

        # This detects duplicates within each chunk only.
        # It does not detect duplicates across different chunks.
        chunk_duplicate_count += int(chunk.duplicated().sum())

    print(f"\nRows: {row_count:,}")
    print(f"Columns: {len(columns)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    print("\nObserved data types:")
    for col, types in dtype_names.items():
        print(f"  {col}: {', '.join(sorted(types))}")

    print("\nMissing values:")
    nonzero_missing = {
        col: count for col, count in missing_counts.items() if count > 0
    }
    if nonzero_missing:
        for col, count in sorted(
            nonzero_missing.items(), key=lambda item: item[1], reverse=True
        ):
            print(f"  {col}: {count:,}")
    else:
        print("  None detected")

    print("\nLabel distribution:")
    if label_counts:
        for label, count in label_counts.most_common():
            print(f"  {label}: {count:,} ({count / row_count:.2%})")
    else:
        print("  No label counts available")

    print("\nInfinite numeric values:")
    nonzero_inf = {
        col: count for col, count in infinite_counts.items() if count > 0
    }
    if nonzero_inf:
        for col, count in nonzero_inf.items():
            print(f"  {col}: {count:,}")
    else:
        print("  None detected in columns parsed as numeric")

    print("\nDuplicate rows within individual chunks:")
    print(f"  {chunk_duplicate_count:,}")
    print("  Note: duplicates spanning chunk boundaries are not counted.")

    print("\nPotential unnamed columns:")
    print(f"  {unnamed_columns if unnamed_columns else 'None'}")

    return {
        "file": file_path.name,
        "rows": row_count,
        "columns": columns,
        "labels": dict(label_counts),
        "missing": dict(missing_counts),
        "chunk_duplicates": chunk_duplicate_count,
    }


def main() -> None:
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    print(f"Dataset directory: {DATA_DIR}")
    print(f"CSV files found: {len(files)}")

    results = [audit_file(path) for path in files]

    print("\n" + "=" * 78)
    print("CROSS-FILE SCHEMA COMPARISON")
    print("=" * 78)

    reference = set(results[0]["columns"])
    print(f"Reference file: {results[0]['file']}")

    for result in results:
        current = set(result["columns"])
        missing = sorted(reference - current)
        extra = sorted(current - reference)

        print(f"\n{result['file']}")
        print(f"  Rows: {result['rows']:,}")
        print(f"  Columns match reference: {current == reference}")
        print(f"  Missing columns: {missing or 'None'}")
        print(f"  Extra columns: {extra or 'None'}")

    print("\nAudit completed. No original files were modified.")


if __name__ == "__main__":
    main()
