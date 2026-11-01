
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "unsw_nb15"

FILES = {
    "Training": DATA_DIR / "UNSW_NB15_training-set.csv",
    "Testing": DATA_DIR / "UNSW_NB15_testing-set.csv",
}


def audit_dataset(name: str, file_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"{name.upper()} DATASET AUDIT")
    print("=" * 70)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)

    print(f"File: {file_path.name}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values (top 20 columns):")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.head(20).to_string())

    print("\nExact duplicate rows:")
    print(f"{df.duplicated().sum():,}")

    print("\nInfinite values in numeric columns:")
    numeric_df = df.select_dtypes(include=[np.number])
    infinite_counts = pd.Series(
        np.isinf(numeric_df.to_numpy()).sum(axis=0),
        index=numeric_df.columns,
    )
    infinite_counts = infinite_counts[infinite_counts > 0]

    if infinite_counts.empty:
        print("No infinite values detected.")
    else:
        print(infinite_counts.to_string())

    print("\nLabel distributions:")
    for column in ["label", "attack_cat"]:
        if column in df.columns:
            print(f"\n{column}:")
            print(
                df[column]
                .fillna("<MISSING>")
                .astype(str)
                .value_counts(dropna=False)
                .to_string()
            )
        else:
            print(f"{column}: column not found")

    return df


def main() -> None:
    datasets = {}

    for name, file_path in FILES.items():
        datasets[name] = audit_dataset(name, file_path)

    training_columns = set(datasets["Training"].columns)
    testing_columns = set(datasets["Testing"].columns)

    print("\n" + "=" * 70)
    print("TRAINING / TESTING SCHEMA COMPARISON")
    print("=" * 70)

    print("Columns only in training:")
    print(sorted(training_columns - testing_columns) or "None")

    print("\nColumns only in testing:")
    print(sorted(testing_columns - training_columns) or "None")

    if training_columns == testing_columns:
        print("\nColumn sets match.")
    else:
        print("\nWARNING: Column sets differ. Investigate before modeling.")


if __name__ == "__main__":
    main()
