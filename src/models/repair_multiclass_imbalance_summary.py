
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "experiments" / "multiclass_cic"
OUT = BASE / "imbalance"

# Verify that the actual experiment artifacts exist.
required_files = [
    BASE / "baseline_comparison.csv",
    OUT / "class_weights.csv",
    OUT / "balanced_classification_report.csv",
    OUT / "sqrt_balanced_classification_report.csv",
    OUT / "balanced_confusion_matrix.csv",
    OUT / "sqrt_balanced_confusion_matrix.csv",
    OUT / "xgboost_balanced.json",
    OUT / "xgboost_sqrt_balanced.json",
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment artifact: {path}")

# Read the original XGBoost baseline.
baseline = pd.read_csv(BASE / "baseline_comparison.csv")
baseline = baseline.loc[baseline["model"] == "XGBoost"]

if len(baseline) != 1:
    raise ValueError("Expected exactly one XGBoost baseline row.")

baseline_row = baseline.iloc[0].to_dict()
baseline_row["experiment"] = "baseline"

# Results taken from the completed training output.
weighted_results = [
    {
        "model": "XGBoost_balanced",
        "experiment": "balanced",
        "accuracy": 0.998906,
        "macro_f1": 0.790589,
        "weighted_f1": 0.998929,
        "fit_seconds": 254.21,
        "predict_seconds": 9.01,
        "training_rows": 300000,
        "validation_rows": 442245,
    },
    {
        "model": "XGBoost_sqrt_balanced",
        "experiment": "sqrt_balanced",
        "accuracy": 0.999086,
        "macro_f1": 0.795563,
        "weighted_f1": 0.999087,
        "fit_seconds": 190.34,
        "predict_seconds": 8.89,
        "training_rows": 300000,
        "validation_rows": 442245,
    },
]

comparison = pd.DataFrame(
    [baseline_row] + weighted_results
)

comparison.to_csv(
    OUT / "imbalance_comparison.csv",
    index=False,
)

config = {
    "experiment": "Multiclass XGBoost class weighting",
    "training_rows": 300000,
    "validation_rows": 442245,
    "random_state": 42,
    "training_sample": "Reused baseline training sample positions",
    "validation_split": "Reused group-level validation positions",
    "weighting_methods": [
        "balanced",
        "sqrt_balanced",
    ],
    "weight_formula": (
        "balanced = n_samples / "
        "(n_classes * class_count); "
        "sqrt_balanced = sqrt(balanced)"
    ),
    "results_recovered_from": (
        "Completed training terminal output and saved artifacts"
    ),
    "note": (
        "Comparison summary repaired after a pandas duplicate-column "
        "error. Models were not retrained. Timing values are retained "
        "from the completed terminal output."
    ),
}

with open(
    OUT / "imbalance_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(config, file, indent=4)

print("Summary recovery completed.")
print("\nCOMPARISON")
print(comparison.to_string(index=False))
print("\nSaved:", OUT / "imbalance_comparison.csv")
print("Saved:", OUT / "imbalance_config.json")
print("\nSTATUS: PASS")
