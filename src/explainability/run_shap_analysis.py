
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

RUN_ID = "rf_balanced_subsample_shap_20261030"

DATA_PATH = ROOT / "data" / "processed" / "cic_binary_train.csv"
MODEL_PATH = ROOT / "src" / "models" / f"{RUN_ID}.joblib"

OUT_DIR = ROOT / "experiments" / "shap_analysis" / RUN_ID

SEED = 42
SHAP_SAMPLE_SIZE = 1000

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trained model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    pipeline = joblib.load(MODEL_PATH)

    imputer = pipeline.named_steps["impute"]
    classifier = pipeline.named_steps["clf"]

    print("Loading validation data...")

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

    # Reconstruct the same group-level validation split.
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    inner_idx, val_idx = next(
        gss.split(X, y, groups)
    )

    del inner_idx

    X_val = X[val_idx]
    y_val = y[val_idx]

    print(f"Validation rows: {len(X_val):,}")

    # Select a reproducible, stratified SHAP sample.
    all_val_positions = np.arange(len(y_val))

    shap_positions, _ = train_test_split(
        all_val_positions,
        train_size=SHAP_SAMPLE_SIZE,
        stratify=y_val,
        random_state=SEED,
    )

    X_shap_raw = X_val[shap_positions]
    y_shap = y_val[shap_positions]

    # Use the fitted training imputer.
    X_shap = imputer.transform(X_shap_raw)

    X_shap = pd.DataFrame(
        X_shap,
        columns=feature_cols,
    )

    print(f"SHAP sample shape: {X_shap.shape}")
    print(f"SHAP sample attack share: {y_shap.mean():.4f}")

    print("\nCreating TreeExplainer...")

    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    print("Calculating SHAP values...")

    explanation = explainer(X_shap)

    values = explanation.values

    # For binary Random Forest output, select ATTACK class (class 1).
    if values.ndim == 3:
        attack_values = values[:, :, 1]
        base_values = explanation.base_values

        if base_values.ndim == 2:
            attack_base_values = base_values[:, 1]
        else:
            attack_base_values = base_values

    elif values.ndim == 2:
        attack_values = values
        attack_base_values = explanation.base_values

    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    if attack_values.shape != (
        SHAP_SAMPLE_SIZE,
        len(feature_cols),
    ):
        raise ValueError(
            f"Unexpected ATTACK SHAP shape: {attack_values.shape}"
        )

    print("SHAP calculation completed.")
    print(f"SHAP matrix shape: {attack_values.shape}")

    # Save feature importance.
    importance = np.abs(attack_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_absolute_shap": importance,
    }).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance_df.to_csv(
        OUT_DIR / "shap_feature_importance.csv",
        index=False,
    )

    # Save individual SHAP values.
    shap_df = pd.DataFrame(
        attack_values,
        columns=feature_cols,
    )

    shap_df.insert(0, "actual_label", y_shap)

    shap_df.to_csv(
        OUT_DIR / "shap_values.csv",
        index=False,
    )

    # Save explanation sample positions.
    np.save(
        OUT_DIR / "shap_validation_positions.npy",
        shap_positions,
    )

    # Summary beeswarm plot.
    print("Generating SHAP summary plot...")

    shap.summary_plot(
        attack_values,
        X_shap,
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "shap_summary_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Global importance bar plot.
    print("Generating SHAP importance plot...")

    shap.summary_plot(
        attack_values,
        X_shap,
        plot_type="bar",
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Save explanation metadata.
    metadata = {
        "run_id": RUN_ID,
        "model": "RandomForestClassifier",
        "explainer": "TreeExplainer",
        "explained_class": "ATTACK (class 1)",
        "feature_perturbation": "tree_path_dependent",
        "model_output": "raw",
        "sample_size": SHAP_SAMPLE_SIZE,
        "sample_method": "stratified validation sample",
        "random_state": SEED,
        "validation_rows": int(len(X_val)),
        "sample_attack_share": float(y_shap.mean()),
        "feature_count": len(feature_cols),
        "shap_values_shape": list(attack_values.shape),
        "shap_version": shap.__version__,
        "note": (
            "SHAP values explain the model output for ATTACK class. "
            "They describe model attribution, not causal effects."
        ),
    }

    (OUT_DIR / "shap_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nTop 20 features:")
    print(importance_df.head(20).to_string(index=False))

    print("\nSHAP analysis completed.")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
