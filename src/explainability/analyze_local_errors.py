
from __future__ import annotations

import json
import re
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

CASES_PATH = OUT_DIR / "local_shap_cases.csv"

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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved local SHAP cases...")
    cases = pd.read_csv(CASES_PATH)

    required_cases = {"TP", "TN", "FP", "FN"}
    available_cases = set(cases["case"].astype(str))

    if not required_cases.issubset(available_cases):
        raise ValueError(
            f"Missing cases. Expected {required_cases}, found {available_cases}"
        )

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

    if len(feature_cols) != 78:
        raise ValueError(
            f"Expected 78 features, found {len(feature_cols)}"
        )

    print("Reconstructing validation split...")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    _, val_idx = next(splitter.split(df, y, groups))

    # Reuse the exact selected records from the previous local SHAP run.
    selected_positions = cases["validation_position"].astype(int).tolist()
    selected_global_rows = cases["dataset_row_index"].astype(int).tolist()

    if any(pos < 0 or pos >= len(val_idx) for pos in selected_positions):
        raise ValueError("A saved validation position is out of range.")

    reconstructed_rows = [int(val_idx[pos]) for pos in selected_positions]

    if reconstructed_rows != selected_global_rows:
        raise RuntimeError(
            "Saved dataset row indices do not match the reconstructed split."
        )

    print("Preparing selected records...")
    X_raw = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32)
        .to_numpy()
    )

    X_selected_raw = X_raw[reconstructed_rows]
    X_selected = imputer.transform(X_selected_raw)

    print("Checking selected predictions...")
    probabilities = classifier.predict_proba(X_selected)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)

    # Recompute explanations for exactly these four saved records.
    print("Calculating SHAP contributions...")
    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    explanation = explainer(
        pd.DataFrame(X_selected, columns=feature_cols)
    )

    values = explanation.values
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        shap_values = values[:, :, 1]
        if base_values.ndim == 2:
            base_values = base_values[:, 1]
    elif values.ndim == 2:
        shap_values = values
    else:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

    if base_values.ndim == 0:
        base_values = np.full(len(cases), float(base_values))

    if base_values.shape != (len(cases),):
        raise ValueError(f"Unexpected base values shape: {base_values.shape}")

    case_details = []
    feature_value_rows = []
    shap_rows = []

    for i, case_row in cases.reset_index(drop=True).iterrows():
        case = str(case_row["case"])
        global_row = reconstructed_rows[i]

        actual = int(y[global_row])
        predicted = int(predictions[i])
        probability = float(probabilities[i])

        original = df.iloc[global_row]

        # Verify that the saved case still matches the reconstructed prediction.
        if actual != int(case_row["actual_label"]):
            raise RuntimeError(f"{case}: actual label mismatch.")

        if predicted != int(case_row["predicted_label"]):
            raise RuntimeError(f"{case}: predicted label mismatch.")

        case_details.append({
            "case": case,
            "dataset_row_index": global_row,
            "validation_position": int(case_row["validation_position"]),
            "source_file": original.get("source_file"),
            "source_row": original.get("source_row"),
            "group_id": original.get("group_id"),
            "attack_family": family[global_row],
            "actual_label": actual,
            "predicted_label": predicted,
            "attack_probability": probability,
            "shap_base_value": float(base_values[i]),
            "shap_value_sum": float(np.sum(shap_values[i])),
            "shap_output_sum": float(
                base_values[i] + np.sum(shap_values[i])
            ),
        })

        for j, feature in enumerate(feature_cols):
            raw_value = X_selected_raw[i, j]
            imputed_value = X_selected[i, j]
            contribution = float(shap_values[i, j])

            feature_value_rows.append({
                "case": case,
                "feature": feature,
                "original_value": (
                    float(raw_value) if np.isfinite(raw_value) else np.nan
                ),
                "imputed_value": float(imputed_value),
            })

            shap_rows.append({
                "case": case,
                "feature": feature,
                "feature_value": float(imputed_value),
                "shap_value": contribution,
                "absolute_shap": abs(contribution),
                "impact_direction": (
                    "towards ATTACK" if contribution > 0
                    else "towards BENIGN" if contribution < 0
                    else "no contribution"
                ),
            })

    details_df = pd.DataFrame(case_details)
    feature_values_df = pd.DataFrame(feature_value_rows)
    shap_df = pd.DataFrame(shap_rows)

    shap_df = shap_df.sort_values(
        ["case", "absolute_shap"],
        ascending=[True, False],
    )

    details_df.to_csv(
        OUT_DIR / "local_error_case_details.csv",
        index=False,
    )

    feature_values_df.to_csv(
        OUT_DIR / "local_error_feature_values.csv",
        index=False,
    )

    shap_df.to_csv(
        OUT_DIR / "local_error_shap_contributions.csv",
        index=False,
    )

    # Audit pandas-style duplicate column names such as Feature and Feature.1.
    print("Auditing duplicate-looking feature columns...")
    duplicate_rows = []

    for feature in feature_cols:
        match = re.match(r"^(.*)\.(\d+)$", feature)

        if not match:
            continue

        base_feature = match.group(1)

        if base_feature not in df.columns:
            continue

        left = pd.to_numeric(df[base_feature], errors="coerce")
        right = pd.to_numeric(df[feature], errors="coerce")

        equal_mask = (
            (left == right)
            | (left.isna() & right.isna())
        )

        both_present = left.notna() & right.notna()

        duplicate_rows.append({
            "base_feature": base_feature,
            "suffixed_feature": feature,
            "rows_compared": int(len(df)),
            "equal_rows_including_both_missing": int(equal_mask.sum()),
            "both_present_rows": int(both_present.sum()),
            "equal_when_both_present": int(
                (left[both_present] == right[both_present]).sum()
            ),
            "different_when_both_present": int(
                (left[both_present] != right[both_present]).sum()
            ),
        })

    duplicate_df = pd.DataFrame(duplicate_rows)

    duplicate_df.to_csv(
        OUT_DIR / "duplicate_feature_audit.csv",
        index=False,
    )

    config = {
        "run_id": RUN_ID,
        "source_cases": CASES_PATH.name,
        "model": MODEL_PATH.name,
        "validation_split": (
            "Reconstructed GroupShuffleSplit, test_size=0.20, random_state=42"
        ),
        "selected_records": len(cases),
        "feature_count": len(feature_cols),
        "shap_explained_class": "ATTACK (class 1)",
        "decision_threshold": 0.5,
        "duplicate_audit": (
            "Checks pandas-style suffixed feature names against their base columns."
        ),
        "shap_version": shap.__version__,
    }

    (OUT_DIR / "local_error_analysis_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\n=== CASE DETAILS ===")
    print(details_df.to_string(index=False))

    print("\n=== TOP 10 SHAP CONTRIBUTIONS PER CASE ===")
    for case in ["TP", "TN", "FP", "FN"]:
        print(f"\n{case}")
        print(
            shap_df[shap_df["case"] == case]
            .head(10)[["feature", "feature_value", "shap_value"]]
            .to_string(index=False)
        )

    print("\n=== DUPLICATE FEATURE AUDIT ===")
    if duplicate_df.empty:
        print("No pandas-style suffixed feature pairs found.")
    else:
        print(duplicate_df.to_string(index=False))

    print("\nLocal error analysis completed.")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
