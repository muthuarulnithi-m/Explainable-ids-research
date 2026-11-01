
from pathlib import Path

import pandas as pd


DATA_FILE = (
    Path("data/raw/cic_ids2017")
    / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
)

LABEL_COLUMN = " Label"

LABEL_MAPPING = {
    "Web Attack \uFFFD Brute Force": "Web Attack - Brute Force",
    "Web Attack \uFFFD XSS": "Web Attack - XSS",
    "Web Attack \uFFFD Sql Injection": "Web Attack - SQL Injection",
}


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(
        DATA_FILE,
        usecols=[LABEL_COLUMN],
        encoding="utf-8",
    )

    original_counts = df[LABEL_COLUMN].value_counts(dropna=False)
    normalized = df[LABEL_COLUMN].replace(LABEL_MAPPING)
    normalized_counts = normalized.value_counts(dropna=False)

    print("Original affected labels:")
    for original_label in LABEL_MAPPING:
        print(
            f"{original_label!r}: "
            f"{int(original_counts.get(original_label, 0))}"
        )

    print("\nNormalized labels:")
    for normalized_label in LABEL_MAPPING.values():
        print(
            f"{normalized_label!r}: "
            f"{int(normalized_counts.get(normalized_label, 0))}"
        )

    expected_counts = {
        "Web Attack - Brute Force": 1507,
        "Web Attack - XSS": 652,
        "Web Attack - SQL Injection": 21,
    }

    for label, expected_count in expected_counts.items():
        actual_count = int(normalized_counts.get(label, 0))
        if actual_count != expected_count:
            raise AssertionError(
                f"{label}: expected {expected_count}, got {actual_count}"
            )

    if len(df) != len(normalized):
        raise AssertionError("Row count changed during normalization.")

    print("\nPASS: Expected label counts are preserved.")
    print("PASS: No rows were added or removed.")
    print("PASS: The raw CSV was not modified.")
    print("Normalization was performed in memory only.")


if __name__ == "__main__":
    main()
