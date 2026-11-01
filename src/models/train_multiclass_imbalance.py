
from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "cic_binary_train.csv"
OUT = ROOT / "experiments" / "multiclass_cic" / "imbalance"

BASE = ROOT / "experiments" / "multiclass_cic"

TARGET = "normalized_label"

METADATA = [
    "source_file",
    "source_row",
    "group_id",
    "normalized_label",
    "binary_label",
]

SEED = 42
TREES = 300
MAX_DEPTH = 8
LEARNING_RATE = 0.1
N_JOBS = 4


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MULTICLASS XGBOOST IMBALANCE EXPERIMENT")
    print("=" * 65)

    required = [
        DATA_PATH,
        BASE / "baseline_training_sample_positions.npy",
        BASE / "validation_positions.npy",
        BASE / "multiclass_imputer.joblib",
        BASE / "multiclass_label_encoder.joblib",
        BASE / "baseline_comparison.csv",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    # --------------------------------------------------
    # 1. Load exact baseline sample and validation rows
    # --------------------------------------------------

    print("\n[1/6] Loading saved experiment positions...")

    sample_positions = np.load(
        BASE / "baseline_training_sample_positions.npy"
    )
    validation_positions = np.load(
        BASE / "validation_positions.npy"
    )

    print("Training sample:", len(sample_positions))
    print("Validation rows:", len(validation_positions))

    # --------------------------------------------------
    # 2. Load only required columns
    # --------------------------------------------------

    print("\n[2/6] Loading dataset features...")

    header = pd.read_csv(DATA_PATH, nrows=0)
    columns = header.columns.tolist()

    features = [c for c in columns if c not in METADATA]

    if len(features) != 78:
        raise ValueError(f"Expected 78 features, got {len(features)}")

    dtype_map = {c: "float32" for c in features}

    data = pd.read_csv(
        DATA_PATH,
        usecols=features + [TARGET],
        dtype=dtype_map,
        low_memory=False,
    )

    if data[TARGET].isna().any():
        raise ValueError("Missing target labels found.")

    data[features] = data[features].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------
    # 3. Transform using saved preprocessing objects
    # --------------------------------------------------

    print("\n[3/6] Applying saved preprocessing...")

    encoder = joblib.load(
        BASE / "multiclass_label_encoder.joblib"
    )
    imputer = joblib.load(
        BASE / "multiclass_imputer.joblib"
    )

    train_frame = data.iloc[sample_positions]
    validation_frame = data.iloc[validation_positions]

    X_train = train_frame[features].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    X_validation = validation_frame[features].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y_train = encoder.transform(train_frame[TARGET].to_numpy())
    y_validation = encoder.transform(
        validation_frame[TARGET].to_numpy()
    )

    X_train = imputer.transform(X_train).astype(
        np.float32,
        copy=False,
    )
    X_validation = imputer.transform(X_validation).astype(
        np.float32,
        copy=False,
    )

    class_names = encoder.classes_.tolist()
    class_ids = np.arange(len(class_names))

    train_counts = np.bincount(
        y_train,
        minlength=len(class_names),
    )

    if np.any(train_counts == 0):
        missing = [
            class_names[i]
            for i, count in enumerate(train_counts)
            if count == 0
        ]
        raise ValueError(
            f"Training sample has no examples for: {missing}"
        )

    print("Training matrix:", X_train.shape)
    print("Validation matrix:", X_validation.shape)
    print("Classes:", len(class_names))

    # --------------------------------------------------
    # 4. Calculate class weights
    # --------------------------------------------------

    print("\n[4/6] Calculating class weights...")

    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=y_train,
    )

    sqrt_weights = np.sqrt(balanced_weights)

    balanced_map = dict(
        zip(class_ids.tolist(), balanced_weights.tolist())
    )
    sqrt_map = dict(
        zip(class_ids.tolist(), sqrt_weights.tolist())
    )

    weight_report = pd.DataFrame({
        "class_name": class_names,
        "training_count": train_counts,
        "balanced_weight": balanced_weights,
        "sqrt_balanced_weight": sqrt_weights,
    })

    weight_report.to_csv(
        OUT / "class_weights.csv",
        index=False,
    )

    print(weight_report.to_string(index=False))

    # --------------------------------------------------
    # 5. Train weighted XGBoost models
    # --------------------------------------------------

    configurations = {
        "balanced": balanced_map,
        "sqrt_balanced": sqrt_map,
    }

    results = []

    for config_name, weight_map in configurations.items():

        print("\n" + "=" * 65)
        print("CONFIGURATION:", config_name.upper())
        print("=" * 65)

        sample_weights = np.array(
            [weight_map[int(label)] for label in y_train],
            dtype=np.float32,
        )

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(class_names),
            n_estimators=TREES,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=N_JOBS,
        )

        start = time.time()

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
        )

        fit_seconds = time.time() - start

        start = time.time()
        predictions = model.predict(X_validation)
        predict_seconds = time.time() - start

        accuracy = accuracy_score(y_validation, predictions)

        macro_f1 = f1_score(
            y_validation,
            predictions,
            labels=class_ids,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_validation,
            predictions,
            labels=class_ids,
            average="weighted",
            zero_division=0,
        )

        report_text = classification_report(
            y_validation,
            predictions,
            labels=class_ids,
            target_names=class_names,
            zero_division=0,
        )

        report_dict = classification_report(
            y_validation,
            predictions,
            labels=class_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        pd.DataFrame(report_dict).transpose().to_csv(
            OUT / f"{config_name}_classification_report.csv"
        )

        with open(
            OUT / f"{config_name}_classification_report.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report_text)

        matrix = confusion_matrix(
            y_validation,
            predictions,
            labels=class_ids,
        )

        pd.DataFrame(
            matrix,
            index=class_names,
            columns=class_names,
        ).to_csv(
            OUT / f"{config_name}_confusion_matrix.csv"
        )

        model.save_model(
            OUT / f"xgboost_{config_name}.json"
        )

        result = {
            "model": f"XGBoost_{config_name}",
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "training_rows": int(len(y_train)),
            "validation_rows": int(len(y_validation)),
        }

        results.append(result)

        print(f"\nAccuracy:       {accuracy:.6f}")
        print(f"Macro F1:       {macro_f1:.6f}")
        print(f"Weighted F1:    {weighted_f1:.6f}")
        print(f"Fit time:       {fit_seconds:.2f} seconds")
        print(f"Predict time:   {predict_seconds:.2f} seconds")

        print("\nCLASSIFICATION REPORT")
        print(report_text)

        del model
        del sample_weights
        del predictions
        gc.collect()

    # --------------------------------------------------
    # 6. Save comparison and configuration
    # --------------------------------------------------

    print("\n[6/6] Saving experiment records...")

    new_results = pd.DataFrame(results)

    baseline = pd.read_csv(
        BASE / "baseline_comparison.csv"
    )

    baseline_xgb = baseline[
        baseline["model"] == "XGBoost"
    ].copy()

    if len(baseline_xgb) != 1:
        raise ValueError("Could not uniquely identify baseline XGBoost.")

    baseline_xgb["experiment"] = "baseline"
    new_results["experiment"] = new_results["model"]

    comparison = pd.concat(
        [
            baseline_xgb.rename(
                columns={"model": "experiment"}
            ),
            new_results,
        ],
        ignore_index=True,
        sort=False,
    )

    comparison.to_csv(
        OUT / "imbalance_comparison.csv",
        index=False,
    )

    config = {
        "experiment": "Multiclass XGBoost class weighting",
        "training_sample_positions": (
            "Reused baseline_training_sample_positions.npy"
        ),
        "validation_positions": "Reused validation_positions.npy",
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "features": features,
        "class_names": class_names,
        "random_state": SEED,
        "weighting_methods": [
            "balanced",
            "sqrt_balanced",
        ],
        "weight_formula": (
            "balanced = n_samples / "
            "(n_classes * class_count); "
            "sqrt_balanced = sqrt(balanced)"
        ),
        "xgboost_parameters": {
            "n_estimators": TREES,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": N_JOBS,
        },
        "xgboost_version": xgboost.__version__,
    }

    save_json(
        OUT / "imbalance_config.json",
        config,
    )

    print("\nFINAL COMPARISON")
    print(comparison.to_string(index=False))

    print("\nOutput directory:", OUT)
    print("\nSTATUS: IMBALANCE EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
