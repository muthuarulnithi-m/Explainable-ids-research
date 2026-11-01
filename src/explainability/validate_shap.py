
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]

RUN_ID = "rf_balanced_subsample_shap_20261030"

DATA_PATH = ROOT / "data" / "processed" / "cic_binary_train.csv"
MODEL_PATH = ROOT / "src" / "models" / f"{RUN_ID}.joblib"

OUT_DIR = ROOT / "experiments" / "shap_analysis" / RUN_ID

SEED = 42
TOLERANCE = 1e-5

LABEL_COL = "normalized_label"
GROUP_COL = "group_id"

METADATA_COLS = [
    "source_file",
    "source_row",
    "group_id",
    LABEL_COL,
    "binary_label",
]


def main():

    print("Loading model...")
    pipeline = joblib.load(MODEL_PATH)

    imputer = pipeline.named_steps["impute"]
    classifier = pipeline.named_steps["clf"]

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    family = (
        df[LABEL_COL]
        .astype(str)
        .str.strip()
        .str.upper()
        .to_numpy()
    )

    y = (family != "BENIGN").astype(np.int8)
    groups = df[GROUP_COL].to_numpy()

    feature_cols = [
        col for col in df.columns
        if col not in METADATA_COLS
    ]

    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32)
        .to_numpy()
    )

    # Reconstruct validation split.
    print("Reconstructing validation split...")

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    _, val_idx = next(
        gss.split(X, y, groups)
    )

    # Load the exact SHAP sample positions.
    positions = np.load(
        OUT_DIR / "shap_validation_positions.npy"
    )

    if positions.max() >= len(val_idx):
        raise ValueError("Invalid SHAP validation positions.")

    X_raw = X[val_idx[positions]]
    y_sample = y[val_idx[positions]]

    X_sample = imputer.transform(X_raw)

    X_sample = pd.DataFrame(
        X_sample,
        columns=feature_cols,
    )

    print(f"Validation sample: {X_sample.shape}")

    # Recompute SHAP values and base values.
    print("Recomputing SHAP explanations...")

    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    explanation = explainer(X_sample)

    values = explanation.values
    base_values = explanation.base_values

    if values.ndim == 3:
        shap_values = values[:, :, 1]

        if base_values.ndim == 2:
            base_values = base_values[:, 1]

    elif values.ndim == 2:
        shap_values = values

    else:
        raise ValueError(
            f"Unexpected SHAP dimensions: {values.shape}"
        )

    base_values = np.asarray(base_values)

    if base_values.ndim == 0:
        base_values = np.full(
            len(X_sample),
            float(base_values),
        )

    if base_values.shape != (len(X_sample),):
        raise ValueError(
            f"Unexpected base value shape: {base_values.shape}"
        )

    # Reconstruct ATTACK output.
    reconstructed = (
        base_values + shap_values.sum(axis=1)
    )

    predicted = classifier.predict_proba(
        X_sample
    )[:, 1]

    absolute_errors = np.abs(
        reconstructed - predicted
    )

    max_error = float(absolute_errors.max())
    mean_error = float(absolute_errors.mean())

    additivity_passed = bool(
        np.allclose(
            reconstructed,
            predicted,
            rtol=1e-4,
            atol=TOLERANCE,
        )
    )

    # Compare recomputed SHAP values with saved CSV.
    saved_df = pd.read_csv(
        OUT_DIR / "shap_values.csv"
    )

    saved_values = saved_df[feature_cols].to_numpy()

    if saved_values.shape != shap_values.shape:
        raise ValueError(
            "Saved SHAP CSV has an unexpected shape."
        )

    saved_shap_max_difference = float(
        np.max(np.abs(saved_values - shap_values))
    )

    saved_labels_match = bool(
        np.array_equal(
            saved_df["actual_label"].to_numpy(),
            y_sample,
        )
    )

    # Save per-record validation results.
    validation_df = pd.DataFrame({
        "actual_label": y_sample,
        "predicted_attack_probability": predicted,
        "shap_reconstructed_output": reconstructed,
        "absolute_additivity_error": absolute_errors,
    })

    validation_df.to_csv(
        OUT_DIR / "shap_additivity_records.csv",
        index=False,
    )

    results = {
        "run_id": RUN_ID,
        "explained_output": "ATTACK class probability",
        "sample_size": int(len(X_sample)),
        "feature_count": int(len(feature_cols)),
        "max_absolute_additivity_error": max_error,
        "mean_absolute_additivity_error": mean_error,
        "tolerance": TOLERANCE,
        "additivity_check_passed": additivity_passed,
        "saved_shap_max_difference": saved_shap_max_difference,
        "saved_actual_labels_match": saved_labels_match,
        "shap_version": shap.__version__,
    }

    (OUT_DIR / "shap_validation_report.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n=== SHAP VALIDATION RESULTS ===")
    print(json.dumps(results, indent=2))

    if not additivity_passed:
        print("\nWARNING: SHAP additivity check did not pass.")
        print("Do not claim numerical validation is complete.")

    elif not saved_labels_match:
        print("\nWARNING: Saved actual labels do not match.")

    elif saved_shap_max_difference > 1e-6:
        print("\nWARNING: Saved and recomputed SHAP values differ.")

    else:
        print("\nSHAP numerical validation PASSED.")

    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
