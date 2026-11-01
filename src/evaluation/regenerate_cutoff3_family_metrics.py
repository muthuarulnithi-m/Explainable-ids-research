
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "generalization"
    / "temporal"
    / "cutoff3"
)

PREDICTION_DIR = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"

MODELS = [
    "logistic_regression",
    "random_forest_balanced_subsample",
]

REQUIRED_COLUMNS = {
    "normalized_label",
    "y_true",
    "prediction",
    "fingerprint_overlap",
}


def parse_boolean(series):
    """Convert CSV boolean representations into actual booleans."""
    if pd.api.types.is_bool_dtype(series):
        return series

    values = series.astype(str).str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    parsed = values.map(mapping)

    if parsed.isna().any():
        invalid = values[parsed.isna()].unique().tolist()
        raise ValueError(
            f"Unexpected fingerprint_overlap values: {invalid}"
        )

    return parsed.astype(bool)


def generate_family_metrics(model_name):
    prediction_path = (
        PREDICTION_DIR / f"{model_name}_predictions.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)

    print(f"\nReading: {prediction_path.name}")

    df = pd.read_csv(prediction_path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["y_true"] = pd.to_numeric(
        df["y_true"], errors="raise"
    ).astype(int)

    df["prediction"] = pd.to_numeric(
        df["prediction"], errors="raise"
    ).astype(int)

    df["fingerprint_overlap"] = parse_boolean(
        df["fingerprint_overlap"]
    )

    if not set(df["y_true"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected y_true values.")

    if not set(df["prediction"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected prediction values.")

    conditions = {
        "natural_evaluation": df,
        "overlap_excluded_evaluation": df.loc[
            ~df["fingerprint_overlap"]
        ].copy(),
    }

    for condition_name, condition_df in conditions.items():
        rows = []

        for family in sorted(
            condition_df["normalized_label"].dropna().unique()
        ):
            family_df = condition_df.loc[
                condition_df["normalized_label"] == family
            ]

            actual = family_df["y_true"]
            predicted = family_df["prediction"]

            is_attack_family = bool((actual == 1).any())

            support = len(family_df)

            detected = int(
                ((actual == 1) & (predicted == 1)).sum()
            )

            missed = int(
                ((actual == 1) & (predicted == 0)).sum()
            )

            predicted_as_attack = int(
                (predicted == 1).sum()
            )

            attack_support = int((actual == 1).sum())

            recall = (
                detected / attack_support
                if attack_support > 0
                else None
            )

            rows.append({
                "attack_family": family,
                "support": support,
                "attack_support": attack_support,
                "detected_attacks": detected,
                "missed_attacks": missed,
                "predicted_as_attack": predicted_as_attack,
                "attack_recall": recall,
            })

        output_path = (
            METRIC_DIR
            / f"{model_name}_{condition_name}_family_metrics.csv"
        )

        pd.DataFrame(rows).to_csv(
            output_path,
            index=False,
        )

        print(f"\n{condition_name}")
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"Saved: {output_path}")


def main():
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        generate_family_metrics(model_name)

    print("\nFamily metric regeneration completed.")
    print("No model training or inference was performed.")


if __name__ == "__main__":
    main()
