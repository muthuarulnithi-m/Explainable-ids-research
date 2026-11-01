
from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "cic_binary_train.csv"
EXPERIMENT_DIR = ROOT / "experiments" / "multiclass_cic"

TRAIN_POSITIONS_PATH = EXPERIMENT_DIR / "train_positions.npy"
VALIDATION_POSITIONS_PATH = EXPERIMENT_DIR / "validation_positions.npy"

TARGET_COLUMN = "normalized_label"

METADATA_COLUMNS = [
    "source_file",
    "source_row",
    "group_id",
    "normalized_label",
    "binary_label",
]

SAMPLE_SIZE = 300_000
RANDOM_STATE = 42

# Random Forest settings
RF_TREES = 100
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 2

# XGBoost settings
XGB_TREES = 300
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_N_JOBS = 4


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    class_names,
):
    print("\n" + "=" * 65)
    print(f"TRAINING: {model_name}")
    print("=" * 65)

    start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - start

    print(f"Training completed in {fit_seconds:.2f} seconds.")

    start = time.time()
    predictions = model.predict(X_validation)
    predict_seconds = time.time() - start

    accuracy = accuracy_score(y_validation, predictions)
    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_validation,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        EXPERIMENT_DIR / f"{model_name.lower()}_classification_report.csv"
    )

    report_text = classification_report(
        y_validation,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )

    with open(
        EXPERIMENT_DIR / f"{model_name.lower()}_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=np.arange(len(class_names)),
    )

    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(
        EXPERIMENT_DIR / f"{model_name.lower()}_confusion_matrix.csv"
    )

    result = {
        "model": model_name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "validation_rows": int(len(y_validation)),
        "training_rows": int(len(y_train)),
    }

    print("\nSUMMARY")
    print(f"Accuracy:       {accuracy:.6f}")
    print(f"Macro F1:       {macro_f1:.6f}")
    print(f"Weighted F1:    {weighted_f1:.6f}")
    print(f"Fit time:       {fit_seconds:.2f} seconds")
    print(f"Predict time:   {predict_seconds:.2f} seconds")

    print("\nCLASSIFICATION REPORT")
    print(report_text)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not TRAIN_POSITIONS_PATH.exists():
        raise FileNotFoundError(TRAIN_POSITIONS_PATH)

    if not VALIDATION_POSITIONS_PATH.exists():
        raise FileNotFoundError(VALIDATION_POSITIONS_PATH)

    print("=" * 65)
    print("CIC MULTICLASS BASELINE EXPERIMENT")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load saved split positions
    # --------------------------------------------------------

    print("\n[1/7] Loading saved split positions...")

    train_positions = np.load(TRAIN_POSITIONS_PATH)
    validation_positions = np.load(VALIDATION_POSITIONS_PATH)

    print(f"Training partition:   {len(train_positions):,}")
    print(f"Validation partition: {len(validation_positions):,}")

    # --------------------------------------------------------
    # 2. Identify and load model features
    # --------------------------------------------------------

    print("\n[2/7] Identifying model features...")

    header = pd.read_csv(DATA_PATH, nrows=0)
    all_columns = header.columns.tolist()

    feature_columns = [
        column
        for column in all_columns
        if column not in METADATA_COLUMNS
    ]

    if TARGET_COLUMN not in all_columns:
        raise ValueError("Target column is missing.")

    if len(feature_columns) != 78:
        raise ValueError(
            f"Expected 78 features, found {len(feature_columns)}."
        )

    dtype_mapping = {
        column: "float32"
        for column in feature_columns
    }

    print(f"Model features: {len(feature_columns)}")
    print("\nLoading dataset. This may take a few minutes...")

    data = pd.read_csv(
        DATA_PATH,
        usecols=feature_columns + [TARGET_COLUMN],
        dtype=dtype_mapping,
        low_memory=False,
    )

    if len(data) <= max(
        train_positions.max(),
        validation_positions.max(),
    ):
        raise ValueError("Saved split positions exceed dataset length.")

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Missing target labels detected.")

    # Convert infinite feature values into missing values.
    data[feature_columns] = data[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print(f"Dataset rows: {len(data):,}")
    print(f"Feature count: {len(feature_columns)}")

    # --------------------------------------------------------
    # 3. Create reproducible stratified training sample
    # --------------------------------------------------------

    print("\n[3/7] Creating stratified training sample...")

    all_training_labels = data.iloc[train_positions][TARGET_COLUMN]

    sample_positions, _ = train_test_split(
        train_positions,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=all_training_labels,
    )

    # Keep validation partition unchanged.
    X_train = data.iloc[sample_positions][feature_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    X_validation = data.iloc[validation_positions][feature_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y_train_text = data.iloc[sample_positions][TARGET_COLUMN].to_numpy()
    y_validation_text = data.iloc[validation_positions][TARGET_COLUMN].to_numpy()

    # Save exact sampled row positions for reproducibility.
    np.save(
        EXPERIMENT_DIR / "baseline_training_sample_positions.npy",
        sample_positions,
    )

    del data
    del all_training_labels
    gc.collect()

    print(f"Training sample: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")

    # --------------------------------------------------------
    # 4. Encode labels and preprocess features
    # --------------------------------------------------------

    print("\n[4/7] Encoding labels and imputing missing values...")

    encoder = LabelEncoder()

    encoder.fit(np.concatenate([
        y_train_text,
        y_validation_text,
    ]))

    y_train = encoder.transform(y_train_text)
    y_validation = encoder.transform(y_validation_text)

    class_names = encoder.classes_.tolist()

    if len(class_names) != 15:
        raise ValueError(
            f"Expected 15 classes, found {len(class_names)}."
        )

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train).astype(
        np.float32,
        copy=False,
    )

    X_validation = imputer.transform(X_validation).astype(
        np.float32,
        copy=False,
    )

    print(f"Classes: {len(class_names)}")
    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_validation.shape}")

    # Save fitted preprocessing objects.
    joblib.dump(
        imputer,
        EXPERIMENT_DIR / "multiclass_imputer.joblib",
    )

    joblib.dump(
        encoder,
        EXPERIMENT_DIR / "multiclass_label_encoder.joblib",
    )

    # --------------------------------------------------------
    # 5. Random Forest baseline
    # --------------------------------------------------------

    rf_model = RandomForestClassifier(
        n_estimators=RF_TREES,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_result = evaluate_model(
        "RandomForest",
        rf_model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        class_names,
    )

    joblib.dump(
        rf_model,
        EXPERIMENT_DIR / "multiclass_random_forest.joblib",
    )

    del rf_model
    gc.collect()

    # --------------------------------------------------------
    # 6. XGBoost baseline
    # --------------------------------------------------------

    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        n_estimators=XGB_TREES,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=XGB_N_JOBS,
    )

    xgb_result = evaluate_model(
        "XGBoost",
        xgb_model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        class_names,
    )

    xgb_model.save_model(
        EXPERIMENT_DIR / "multiclass_xgboost.json"
    )

    del xgb_model
    gc.collect()

    # --------------------------------------------------------
    # 7. Save comparison and experiment configuration
    # --------------------------------------------------------

    print("\n[7/7] Saving experiment results...")

    comparison = pd.DataFrame([
        rf_result,
        xgb_result,
    ])

    comparison.to_csv(
        EXPERIMENT_DIR / "baseline_comparison.csv",
        index=False,
    )

    config = {
        "experiment": "CIC multiclass baseline models",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "target": TARGET_COLUMN,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "metadata_excluded": METADATA_COLUMNS,
        "class_names": class_names,
        "class_count": len(class_names),
        "split_method": "GroupShuffleSplit",
        "split_random_state": RANDOM_STATE,
        "training_sample_size": SAMPLE_SIZE,
        "sample_method": "Stratified sampling from saved training partition",
        "sample_random_state": RANDOM_STATE,
        "validation_rows": int(len(y_validation)),
        "imputation": "Median, fitted on training sample only",
        "models": {
            "RandomForest": {
                "n_estimators": RF_TREES,
                "max_depth": RF_MAX_DEPTH,
                "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
                "class_weight": None,
                "random_state": RANDOM_STATE,
            },
            "XGBoost": {
                "objective": "multi:softprob",
                "num_class": len(class_names),
                "n_estimators": XGB_TREES,
                "max_depth": XGB_MAX_DEPTH,
                "learning_rate": XGB_LEARNING_RATE,
                "subsample": XGB_SUBSAMPLE,
                "colsample_bytree": XGB_COLSAMPLE,
                "tree_method": "hist",
                "eval_metric": "mlogloss",
                "n_jobs": XGB_N_JOBS,
                "random_state": RANDOM_STATE,
            },
        },
        "versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    save_json(
        EXPERIMENT_DIR / "baseline_run_config.json",
        config,
    )

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    print("\nOutput directory:")
    print(EXPERIMENT_DIR)

    print("\nSTATUS: BASELINE EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
