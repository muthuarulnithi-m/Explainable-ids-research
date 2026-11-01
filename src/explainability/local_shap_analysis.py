
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

from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]

RUN_ID = "rf_balanced_subsample_shap_20261030"

DATA_PATH = ROOT / "data" / "processed" / "cic_binary_train.csv"
MODEL_PATH = ROOT / "src" / "models" / f"{RUN_ID}.joblib"
OUT_DIR = ROOT / "experiments" / "shap_analysis" / RUN_ID

SEED = 42
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

    # Reconstruct the established validation split.
    print("Reconstructing validation split...")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    _, val_idx = next(
        splitter.split(X, y, groups)
    )

    X_val_raw = X[val_idx]
    y_val = y[val_idx]

    print(f"Validation records: {len(y_val):,}")

    # Apply the already-fitted training imputer.
    X_val = imputer.transform(X_val_raw)

    # Predict the full validation set.
    print("Predicting validation records...")

    probabilities = classifier.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)

    # Find representative cases.
    candidates = {
        "TP": np.where((y_val == 1) & (predictions == 1))[0],
        "TN": np.where((y_val == 0) & (predictions == 0))[0],
        "FP": np.where((y_val == 0) & (predictions == 1))[0],
        "FN": np.where((y_val == 1) & (predictions == 0))[0],
    }

    selected = {}

    for case, indices in candidates.items():

        if len(indices) == 0:
            print(f"WARNING: No {case} example found.")
            continue

        if case == "TP":
            chosen = indices[np.argmax(probabilities[indices])]

        elif case == "TN":
            chosen = indices[np.argmin(probabilities[indices])]

        elif case == "FP":
            chosen = indices[np.argmax(probabilities[indices])]

        else:  # FN
            chosen = indices[np.argmin(probabilities[indices])]

        selected[case] = int(chosen)

    if not selected:
        raise RuntimeError("No prediction cases were selected.")

    # Create explanation object for selected records only.
    selected_positions = list(selected.values())

    X_selected = pd.DataFrame(
        X_val[selected_positions],
        columns=feature_cols,
    )

    print("\nCreating TreeExplainer...")

    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    print("Calculating local SHAP values...")

    explanation = explainer(X_selected)

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
            f"Unexpected SHAP output shape: {values.shape}"
        )

    base_values = np.asarray(base_values)

    if base_values.ndim == 0:
        base_values = np.full(
            len(selected_positions),
            float(base_values),
        )

    if base_values.shape != (len(selected_positions),):
        raise ValueError(
            f"Unexpected base value shape: {base_values.shape}"
        )

    # Generate one waterfall plot per case.
    records = []

    for row_number, (case, val_position) in enumerate(selected.items()):

        actual = int(y_val[val_position])
        predicted = int(predictions[val_position])
        probability = float(probabilities[val_position])

        global_row = int(val_idx[val_position])

        print(f"\nGenerating {case} explanation...")

        local_explanation = shap.Explanation(
            values=shap_values[row_number],
            base_values=base_values[row_number],
            data=X_selected.iloc[row_number].to_numpy(),
            feature_names=feature_cols,
        )

        plt.figure(figsize=(11, 8))

        shap.plots.waterfall(
            local_explanation,
            max_display=15,
            show=False,
        )

        plt.title(
            f"{case} | Actual={actual} | "
            f"Predicted={predicted} | "
            f"ATTACK probability={probability:.4f}"
        )

        plt.tight_layout()

        plot_path = OUT_DIR / f"local_shap_{case.lower()}.png"

        plt.savefig(
            plot_path,
            dpi=250,
            bbox_inches="tight",
        )

        plt.close()

        records.append({
            "case": case,
            "dataset_row_index": global_row,
            "validation_position": int(val_position),
            "actual_label": actual,
            "predicted_label": predicted,
            "attack_probability": probability,
            "waterfall_plot": plot_path.name,
        })

    # Save selected case information.
    cases_df = pd.DataFrame(records)

    cases_df.to_csv(
        OUT_DIR / "local_shap_cases.csv",
        index=False,
    )

    metadata = {
        "run_id": RUN_ID,
        "explainer": "TreeExplainer",
        "explained_class": "ATTACK (class 1)",
        "selection": (
            "Most confident representative prediction "
            "within each available TP, TN, FP and FN category."
        ),
        "decision_threshold": 0.5,
        "validation_rows": int(len(y_val)),
        "cases_generated": list(selected.keys()),
        "feature_count": len(feature_cols),
        "shap_version": shap.__version__,
    }

    (OUT_DIR / "local_shap_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n=== LOCAL EXPLANATION SUMMARY ===")
    print(cases_df.to_string(index=False))

    print("\nLocal SHAP analysis completed.")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
