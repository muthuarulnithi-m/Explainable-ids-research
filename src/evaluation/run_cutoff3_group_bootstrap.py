
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "experiments/generalization/temporal/cutoff3/group_audit"
)

OUTPUT_DIR = (
    ROOT / "experiments/generalization/temporal/cutoff3/group_bootstrap"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LR_FILE = INPUT_DIR / "logistic_regression_predictions_with_groups.csv"
RF_FILE = INPUT_DIR / "random_forest_predictions_with_groups.csv"

N_BOOTSTRAPS = 5000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

METRICS = [
    "accuracy",
    "macro_f1",
    "attack_precision",
    "attack_recall",
    "attack_f1",
    "false_positive_rate",
]


def load_predictions(path, model_name):
    df = pd.read_csv(
        path,
        usecols=[
            "source_file",
            "source_row",
            "group_id",
            "y_true",
            "prediction",
        ],
        dtype={
            "source_file": "string",
            "group_id": "string",
        },
    )

    df["source_row"] = pd.to_numeric(
        df["source_row"], errors="raise"
    ).astype("int64")

    for col in ["y_true", "prediction"]:
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        ).astype("int8")

    if not set(df["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_name}: invalid y_true values.")

    if not set(df["prediction"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_name}: invalid prediction values.")

    if df["group_id"].isna().any():
        raise ValueError(f"{model_name}: missing group_id.")

    if df.duplicated(["source_file", "source_row"]).any():
        raise ValueError(f"{model_name}: duplicate prediction keys.")

    return df


def confusion_contributions(df, suffix):
    """
    Return one row per group with its TN, FP, FN and TP contributions.
    """

    y = df["y_true"].to_numpy()
    p = df["prediction"].to_numpy()

    work = pd.DataFrame({
        "group_id": df["group_id"].to_numpy(),
        "tn": ((y == 0) & (p == 0)).astype(np.int64),
        "fp": ((y == 0) & (p == 1)).astype(np.int64),
        "fn": ((y == 1) & (p == 0)).astype(np.int64),
        "tp": ((y == 1) & (p == 1)).astype(np.int64),
    })

    result = (
        work.groupby("group_id", sort=False)[
            ["tn", "fp", "fn", "tp"]
        ]
        .sum()
        .reset_index()
    )

    result = result.rename(
        columns={
            "tn": f"tn_{suffix}",
            "fp": f"fp_{suffix}",
            "fn": f"fn_{suffix}",
            "tp": f"tp_{suffix}",
        }
    )

    return result


def calculate_metrics(counts):
    tn, fp, fn, tp = map(float, counts)

    total = tn + fp + fn + tp
    if total == 0:
        return {metric: np.nan for metric in METRICS}

    accuracy = (tp + tn) / total

    attack_precision = (
        tp / (tp + fp) if (tp + fp) > 0 else 0.0
    )
    attack_recall = (
        tp / (tp + fn) if (tp + fn) > 0 else 0.0
    )

    attack_f1 = (
        2 * attack_precision * attack_recall
        / (attack_precision + attack_recall)
        if (attack_precision + attack_recall) > 0
        else 0.0
    )

    benign_precision = (
        tn / (tn + fn) if (tn + fn) > 0 else 0.0
    )
    benign_recall = (
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

    benign_f1 = (
        2 * benign_precision * benign_recall
        / (benign_precision + benign_recall)
        if (benign_precision + benign_recall) > 0
        else 0.0
    )

    macro_f1 = (benign_f1 + attack_f1) / 2

    false_positive_rate = (
        fp / (fp + tn) if (fp + tn) > 0 else np.nan
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "attack_f1": attack_f1,
        "false_positive_rate": false_positive_rate,
    }


def percentile_interval(values):
    alpha = 1.0 - CONFIDENCE_LEVEL
    return (
        float(np.nanpercentile(values, 100 * alpha / 2)),
        float(np.nanpercentile(values, 100 * (1 - alpha / 2))),
    )


def main():
    print("Loading saved predictions...")

    lr = load_predictions(LR_FILE, "Logistic Regression")
    rf = load_predictions(RF_FILE, "Random Forest")

    keys = ["source_file", "source_row"]

    if len(lr) != len(rf):
        raise ValueError("LR and RF prediction row counts differ.")

    paired = lr.merge(
        rf,
        on=keys,
        how="inner",
        suffixes=("_lr", "_rf"),
        validate="one_to_one",
    )

    if len(paired) != len(lr):
        raise ValueError("LR and RF prediction keys do not match.")

    checks = [
        ("group_id_lr", "group_id_rf"),
        ("y_true_lr", "y_true_rf"),
    ]

    for left, right in checks:
        if not (paired[left] == paired[right]).all():
            raise ValueError(f"Mismatch detected: {left} vs {right}")

    print(f"Paired evaluation rows: {len(paired):,}")

    lr_for_groups = paired[
        ["group_id_lr", "y_true_lr", "prediction_lr"]
    ].rename(
        columns={
            "group_id_lr": "group_id",
            "y_true_lr": "y_true",
            "prediction_lr": "prediction",
        }
    )

    rf_for_groups = paired[
        ["group_id_rf", "y_true_rf", "prediction_rf"]
    ].rename(
        columns={
            "group_id_rf": "group_id",
            "y_true_rf": "y_true",
            "prediction_rf": "prediction",
        }
    )

    lr_groups = confusion_contributions(lr_for_groups, "lr")
    rf_groups = confusion_contributions(rf_for_groups, "rf")

    groups = lr_groups.merge(
        rf_groups,
        on="group_id",
        how="inner",
        validate="one_to_one",
    )

    if len(groups) != lr_groups["group_id"].nunique():
        raise ValueError("Group coverage mismatch between models.")

    print(f"Evaluation groups: {len(groups):,}")

    lr_cols = ["tn_lr", "fp_lr", "fn_lr", "tp_lr"]
    rf_cols = ["tn_rf", "fp_rf", "fn_rf", "tp_rf"]

    lr_matrix = groups[lr_cols].to_numpy(dtype=np.int64)
    rf_matrix = groups[rf_cols].to_numpy(dtype=np.int64)

    lr_total = lr_matrix.sum(axis=0)
    rf_total = rf_matrix.sum(axis=0)

    # Full natural evaluation point estimates.
    point_rows = []

    for model_name, total in [
        ("logistic_regression", lr_total),
        ("random_forest", rf_total),
    ]:
        metrics = calculate_metrics(total)
        for metric, value in metrics.items():
            point_rows.append({
                "model": model_name,
                "condition": "all_evaluation_groups",
                "metric": metric,
                "value": value,
            })

    # Largest-group sensitivity analysis.

    group_sizes = (
    paired.groupby("group_id_lr", sort=False)
    .size()
    .rename("flow_count")
    )

    largest_group_id = group_sizes.idxmax()
    largest_group_size = int(group_sizes.max())

    largest_positions = np.flatnonzero(
    groups["group_id"].to_numpy() == largest_group_id
   )

    if len(largest_positions) != 1:
     raise ValueError(
        "Could not identify exactly one largest group."
    )

    largest_index = int(largest_positions[0])

    print(f"Largest group size: {largest_group_size:,}")
    print(f"Largest group ID: {largest_group_id}")

    lr_without_largest = lr_total - lr_matrix[largest_index]
    rf_without_largest = rf_total - rf_matrix[largest_index]

    for model_name, total in [
        ("logistic_regression", lr_without_largest),
        ("random_forest", rf_without_largest),
    ]:
        metrics = calculate_metrics(total)
        for metric, value in metrics.items():
            point_rows.append({
                "model": model_name,
                "condition": "largest_group_excluded",
                "metric": metric,
                "value": value,
            })

    pd.DataFrame(point_rows).to_csv(
        OUTPUT_DIR / "point_metrics_and_largest_group_sensitivity.csv",
        index=False,
    )

    # Cluster bootstrap: sample groups, retaining all flows per group.
    rng = np.random.default_rng(RANDOM_SEED)

    n_groups = len(groups)
    bootstrap_values = {
        "logistic_regression": {m: [] for m in METRICS},
        "random_forest": {m: [] for m in METRICS},
        "paired_difference_lr_minus_rf": {m: [] for m in METRICS},
    }

    print(f"\nRunning {N_BOOTSTRAPS:,} group bootstrap replicates...")

    for iteration in range(N_BOOTSTRAPS):
        sampled_indices = rng.integers(
            0, n_groups, size=n_groups
        )

        lr_counts = lr_matrix[sampled_indices].sum(axis=0)
        rf_counts = rf_matrix[sampled_indices].sum(axis=0)

        lr_metrics = calculate_metrics(lr_counts)
        rf_metrics = calculate_metrics(rf_counts)

        for metric in METRICS:
            lr_value = lr_metrics[metric]
            rf_value = rf_metrics[metric]

            bootstrap_values["logistic_regression"][metric].append(
                lr_value
            )
            bootstrap_values["random_forest"][metric].append(
                rf_value
            )

            bootstrap_values[
                "paired_difference_lr_minus_rf"
            ][metric].append(lr_value - rf_value)

        if (iteration + 1) % 100 == 0:
            print(
                f"Completed {iteration + 1:,}/"
                f"{N_BOOTSTRAPS:,}"
            )

    # Save confidence intervals.
    ci_rows = []

    for model_name, metric_values in bootstrap_values.items():
        for metric, values in metric_values.items():
            values = np.asarray(values, dtype=float)
            lower, upper = percentile_interval(values)

            ci_rows.append({
                "model_or_difference": model_name,
                "metric": metric,
                "bootstrap_replicates": N_BOOTSTRAPS,
                "confidence_level": CONFIDENCE_LEVEL,
                "lower_95_ci": lower,
                "upper_95_ci": upper,
                "bootstrap_mean": float(np.nanmean(values)),
            })

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(
        OUTPUT_DIR / "group_bootstrap_confidence_intervals.csv",
        index=False,
    )

    # Save exact bootstrap distributions for reproducibility.
    distribution_rows = []

    for model_name, metric_values in bootstrap_values.items():
        for metric, values in metric_values.items():
            for replicate, value in enumerate(values, start=1):
                distribution_rows.append({
                    "model_or_difference": model_name,
                    "metric": metric,
                    "replicate": replicate,
                    "value": value,
                })

    pd.DataFrame(distribution_rows).to_csv(
        OUTPUT_DIR / "group_bootstrap_distributions.csv",
        index=False,
    )

    print("\nGROUP BOOTSTRAP COMPLETE")
    print("\nPoint estimates and largest-group sensitivity:")
    print(pd.DataFrame(point_rows).to_string(index=False))

    print("\nGroup bootstrap confidence intervals:")
    print(ci_df.to_string(index=False))

    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
