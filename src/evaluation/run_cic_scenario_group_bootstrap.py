
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT / "data" / "processed" / "cic_binary_test.csv"

MODEL_DIR = ROOT / "experiments" / "multiclass_cic"

IMPUTER_PATH = MODEL_DIR / "multiclass_imputer.joblib"
ENCODER_PATH = MODEL_DIR / "multiclass_label_encoder.joblib"

SCENARIO_CONFIG_PATH = (
    ROOT
    / "experiments"
    / "generalization"
    / "cic_scenario_v2"
    / "scenario_config.json"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "generalization"
    / "cic_scenario_group_bootstrap"
)

MODELS = {
    "xgboost_baseline": (
        MODEL_DIR / "multiclass_xgboost.json"
    ),
    "xgboost_balanced": (
        MODEL_DIR / "imbalance" / "xgboost_balanced.json"
    ),
    "xgboost_sqrt_balanced": (
        MODEL_DIR / "imbalance" / "xgboost_sqrt_balanced.json"
    ),
}

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 42
CI_LEVEL = 95
CHUNK_SIZE = 100_000

EXPECTED_TEST_ROWS = 567_870
EXPECTED_GROUPS = 503_927
EXPECTED_FEATURES = 78
EXPECTED_CLASSES = 15

ELIGIBLE_CLASSES = {
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "PortScan",
    "SSH-Patator",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
}

METADATA_COLUMNS = {
    "source_file",
    "source_row",
    "group_id",
    "normalized_label",
    "binary_label",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def verify_hash(path: Path, expected: str, name: str) -> None:
    actual = sha256_of(path)

    require(
        actual == expected,
        f"{name} SHA256 mismatch. Expected {expected}, got {actual}",
    )

    print(f"[PASS] {name} SHA256 verified")


# ============================================================
# METRICS
# ============================================================

def confusion_matrix_codes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    number_of_classes: int,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construct a complete K x K confusion matrix.

    Rows: actual class.
    Columns: predicted class.
    """

    pair_codes = (
        y_true.astype(np.int64) * number_of_classes
        + y_pred.astype(np.int64)
    )

    matrix = np.bincount(
        pair_codes,
        weights=sample_weights,
        minlength=number_of_classes * number_of_classes,
    )

    return matrix.reshape(
        number_of_classes,
        number_of_classes,
    )


def headline_macro_f1(
    matrix: np.ndarray,
    eligible_codes: np.ndarray,
) -> float:
    """
    Macro F1 over eligible classes with positive true support.

    All 15 classes remain in the confusion matrix, so predictions
    into an ineligible class still count as false negatives for
    the actual eligible class.
    """

    true_support = matrix.sum(axis=1)

    selected = eligible_codes[
        true_support[eligible_codes] > 0
    ]

    if len(selected) == 0:
        return float("nan")

    true_positive = np.diag(matrix)

    predicted_support = matrix.sum(axis=0)

    denominator = (
        2 * true_positive
        + (predicted_support - true_positive)
        + (true_support - true_positive)
    )

    class_f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominator != 0,
    )

    return float(np.mean(class_f1[selected]))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    number_of_classes: int,
    eligible_codes: np.ndarray,
) -> tuple[float, np.ndarray]:

    matrix = confusion_matrix_codes(
        y_true,
        y_pred,
        number_of_classes,
    )

    score = headline_macro_f1(
        matrix,
        eligible_codes,
    )

    return score, matrix


# ============================================================
# BOOTSTRAP
# ============================================================

def group_bootstrap(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    group_codes: np.ndarray,
    number_of_groups: int,
    number_of_classes: int,
    eligible_codes: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Paired cluster bootstrap.

    Every replicate samples group IDs with replacement.
    The same group sample is used for all three models.

    Group multiplicities are mapped back to their flows.
    """

    rng = np.random.default_rng(seed)

    model_names = list(predictions)

    scores = {
        name: np.empty(reps, dtype=float)
        for name in model_names
    }

    paired_differences = {
        "baseline_minus_balanced": np.empty(reps),
        "baseline_minus_sqrt_balanced": np.empty(reps),
        "balanced_minus_sqrt_balanced": np.empty(reps),
    }

    number_of_rows = len(y_true)

    print(
        f"\nStarting {reps:,} paired group-bootstrap replicates"
    )
    print(
        f"Groups: {number_of_groups:,}; "
        f"flows: {number_of_rows:,}"
    )

    start = time.time()

    for replicate in range(reps):

        # Draw exactly G groups with replacement.
        sampled_groups = rng.integers(
            0,
            number_of_groups,
            size=number_of_groups,
        )

        # Number of times each group was selected.
        group_multiplicity = np.bincount(
            sampled_groups,
            minlength=number_of_groups,
        )

        # Every flow receives its group's multiplicity.
        row_weights = group_multiplicity[group_codes]

        replicate_scores = {}

        for name in model_names:

            matrix = confusion_matrix_codes(
                y_true,
                predictions[name],
                number_of_classes,
                sample_weights=row_weights,
            )

            score = headline_macro_f1(
                matrix,
                eligible_codes,
            )

            scores[name][replicate] = score
            replicate_scores[name] = score

        paired_differences[
            "baseline_minus_balanced"
        ][replicate] = (
            replicate_scores["xgboost_baseline"]
            - replicate_scores["xgboost_balanced"]
        )

        paired_differences[
            "baseline_minus_sqrt_balanced"
        ][replicate] = (
            replicate_scores["xgboost_baseline"]
            - replicate_scores["xgboost_sqrt_balanced"]
        )

        paired_differences[
            "balanced_minus_sqrt_balanced"
        ][replicate] = (
            replicate_scores["xgboost_balanced"]
            - replicate_scores["xgboost_sqrt_balanced"]
        )

        if (replicate + 1) % 100 == 0:
            elapsed = time.time() - start

            print(
                f"Completed {replicate + 1:,}/{reps:,} "
                f"replicates | elapsed {elapsed / 60:.1f} min"
            )

    return scores, paired_differences


def confidence_interval(
    values: np.ndarray,
    confidence_level: float = 95,
) -> tuple[float, float]:

    alpha = (100 - confidence_level) / 2

    lower, upper = np.percentile(
        values,
        [alpha, 100 - alpha],
    )

    return float(lower), float(upper)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("========== CIC GROUP BOOTSTRAP ==========")

    # --------------------------------------------------------
    # 1. Read and verify original scenario configuration
    # --------------------------------------------------------

    config = json.loads(
        SCENARIO_CONFIG_PATH.read_text(encoding="utf-8")
    )

    test_metadata = config["input_files"]["test"]

    verify_hash(
        TEST_PATH,
        test_metadata["sha256"],
        "test dataset",
    )

    verify_hash(
        IMPUTER_PATH,
        config["imputer_sha256"],
        "imputer",
    )

    verify_hash(
        ENCODER_PATH,
        config["label_encoder_sha256"],
        "label encoder",
    )

    for name, path in MODELS.items():
        verify_hash(
            path,
            config["models"][name]["sha256"],
            name,
        )

    # --------------------------------------------------------
    # 2. Load saved preprocessing and label artifacts
    # --------------------------------------------------------

    imputer = joblib.load(IMPUTER_PATH)
    encoder = joblib.load(ENCODER_PATH)

        # --------------------------------------------------------
    # 2. Load saved preprocessing and label artifacts
    # --------------------------------------------------------

    imputer = joblib.load(IMPUTER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # Detect feature order safely.
    if hasattr(imputer, "feature_names_in_"):
        feature_names = list(imputer.feature_names_in_)
        order_source = "imputer.feature_names_in_"
    else:
        feature_names = [
            c
            for c in pd.read_csv(TEST_PATH, nrows=0).columns
            if c not in METADATA_COLUMNS
        ]
        order_source = (
            "non-metadata columns in file order; "
            "verified by the reproduction check"
        )

    # Validate feature count.
    require(
        len(feature_names) == EXPECTED_FEATURES,
        f"found {len(feature_names)} features, "
        f"expected {EXPECTED_FEATURES}",
    )

    require(
        len(imputer.statistics_) == EXPECTED_FEATURES,
        "imputer statistics count does not match feature count",
    )

    print(
        f"[PASS] Feature order source: {order_source}; "
        f"{len(feature_names)} features"
    )

    classes = [str(value) for value in encoder.classes_]

    require(
        len(classes) == EXPECTED_CLASSES,
        "Unexpected class count",
    )

    require(
        set(ELIGIBLE_CLASSES).issubset(set(classes)),
        "Eligible class list does not match encoder",
    )

    eligible_codes = np.array(
        [
            index
            for index, label in enumerate(classes)
            if label in ELIGIBLE_CLASSES
        ],
        dtype=int,
    )

    require(
        len(eligible_codes) == 12,
        "Expected exactly 12 eligible classes",
    )

    print("[PASS] Saved imputer and encoder loaded")
    print("[PASS] 78 features and 15 classes verified")
    classes = [str(value) for value in encoder.classes_]

    require(
        len(feature_names) == EXPECTED_FEATURES,
        "Unexpected feature count",
    )

    require(
        len(classes) == EXPECTED_CLASSES,
        "Unexpected class count",
    )

    require(
        set(ELIGIBLE_CLASSES).issubset(set(classes)),
        "Eligible class list does not match encoder",
    )

    eligible_codes = np.array(
        [
            index
            for index, label in enumerate(classes)
            if label in ELIGIBLE_CLASSES
        ],
        dtype=int,
    )

    require(
        len(eligible_codes) == 12,
        "Expected exactly 12 eligible classes",
    )

    print("[PASS] Saved imputer and encoder loaded")
    print("[PASS] 78 features and 15 classes verified")

    # --------------------------------------------------------
    # 3. Load test data
    # --------------------------------------------------------

    usecols = (
        feature_names
        + [
            "group_id",
            "normalized_label",
        ]
    )

    test = pd.read_csv(
        TEST_PATH,
        usecols=usecols,
    )

    require(
        len(test) == EXPECTED_TEST_ROWS,
        "Unexpected test row count",
    )

    require(
        test["group_id"].notna().all(),
        "Missing group IDs found",
    )

    y_true = encoder.transform(
        test["normalized_label"].to_numpy(dtype=object)
    ).astype(np.int64)

    # Factorize group IDs into compact integer codes.
    group_codes, unique_groups = pd.factorize(
        test["group_id"],
        sort=False,
    )

    number_of_groups = len(unique_groups)

    require(
        number_of_groups == EXPECTED_GROUPS,
        "Unexpected number of test groups",
    )

    require(
        group_codes.min() == 0,
        "Invalid group codes",
    )

    print(
        f"[PASS] Test rows: {len(test):,}; "
        f"unique groups: {number_of_groups:,}"
    )

    # --------------------------------------------------------
    # 4. Impute using the saved imputer only
    # --------------------------------------------------------

    x_raw = test[feature_names].astype(float)

    require(
        int(np.isinf(x_raw.to_numpy()).sum()) == 0,
        "Infinite feature values found",
    )

    x_test = pd.DataFrame(
        imputer.transform(x_raw),
        columns=feature_names,
    )

    del x_raw

    # --------------------------------------------------------
    # 5. Load saved models and predict
    # --------------------------------------------------------

    predictions = {}

    for name, path in MODELS.items():

        print(f"\nLoading {name}")

        model = xgb.XGBClassifier()
        model.load_model(str(path))

        stored_features = model.get_booster().feature_names

        if stored_features is not None:
            require(
                list(stored_features) == feature_names,
                f"{name}: model feature order mismatch",
            )

        predicted_codes = np.empty(
            len(test),
            dtype=np.int64,
        )

        for start in range(0, len(test), CHUNK_SIZE):

            end = min(
                start + CHUNK_SIZE,
                len(test),
            )

            predicted_codes[start:end] = np.asarray(
                model.predict(x_test.iloc[start:end])
            ).astype(np.int64)

        require(
            predicted_codes.min() >= 0
            and predicted_codes.max() < EXPECTED_CLASSES,
            f"{name}: prediction code out of range",
        )

        predictions[name] = predicted_codes

        print(
            f"[PASS] {name}: "
            f"{len(predicted_codes):,} predictions"
        )

        del model

    # --------------------------------------------------------
    # 6. Point estimates on the full test set
    # --------------------------------------------------------

    point_rows = []

    for name, predicted_codes in predictions.items():

        score, matrix = calculate_metrics(
            y_true,
            predicted_codes,
            EXPECTED_CLASSES,
            eligible_codes,
        )

        point_rows.append(
            {
                "model": name,
                "rows": len(y_true),
                "groups": number_of_groups,
                "headline_macro_f1": score,
            }
        )

        pd.DataFrame(
            matrix,
            index=classes,
            columns=classes,
        ).to_csv(
            OUTPUT_DIR / f"{name}_confusion_matrix.csv"
        )

    point_df = pd.DataFrame(point_rows)
        # --------------------------------------------------------
    # Validate point estimates against the original Phase 6.1
    # pooled test results before running the bootstrap.
    # --------------------------------------------------------

    reference_path = (
        ROOT
        / "experiments"
        / "generalization"
        / "cic_scenario_v2"
        / "scenario_metrics_v2.csv"
    )

    reference_df = pd.read_csv(reference_path)

    reference_rows = reference_df[
        reference_df["scenario"] == "ALL_TEST_DAYS"
    ].set_index("model")

    expected_models = {
        "xgboost_baseline",
        "xgboost_balanced",
        "xgboost_sqrt_balanced",
    }

    if not expected_models.issubset(reference_rows.index):
        raise RuntimeError(
            "Original Phase 6.1 reference results are incomplete."
        )

    print("\n========== POINT ESTIMATE VALIDATION ==========")

    for name in sorted(expected_models):
        actual = float(
            point_df.loc[
                point_df["model"] == name,
                "headline_macro_f1",
            ].iloc[0]
        )

        expected = float(
            reference_rows.loc[name, "headline_macro_f1"]
        )

        print(
            f"{name}: new={actual:.9f}, "
            f"original={expected:.9f}"
        )

        if not np.isclose(
            actual,
            expected,
            atol=1e-10,
            rtol=0,
        ):
            raise RuntimeError(
                f"Point estimate mismatch for {name}. "
                "Bootstrap cancelled."
            )

    print("All three point estimates match the original results.")
    point_df.to_csv(
        OUTPUT_DIR / "point_estimates.csv",
        index=False,
    )

    print("\n========== POINT ESTIMATES ==========")
    print(point_df.to_string(index=False))

    # --------------------------------------------------------
    # 7. Largest-group-excluded sensitivity analysis
    # --------------------------------------------------------

    group_sizes = np.bincount(
        group_codes,
        minlength=number_of_groups,
    )

    largest_group_code = int(np.argmax(group_sizes))
    largest_group_size = int(group_sizes[largest_group_code])

    keep_mask = group_codes != largest_group_code

    require(
        int(keep_mask.sum()) == len(test) - largest_group_size,
        "Largest-group exclusion row count mismatch",
    )

    sensitivity_rows = []

    for name, predicted_codes in predictions.items():

        score, _ = calculate_metrics(
            y_true[keep_mask],
            predicted_codes[keep_mask],
            EXPECTED_CLASSES,
            eligible_codes,
        )

        sensitivity_rows.append(
            {
                "model": name,
                "excluded_group_size": largest_group_size,
                "remaining_rows": int(keep_mask.sum()),
                "headline_macro_f1": score,
            }
        )

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    sensitivity_df.to_csv(
        OUTPUT_DIR / "largest_group_sensitivity.csv",
        index=False,
    )

    print("\n========== LARGEST GROUP SENSITIVITY ==========")
    print(
        f"Excluded group size: {largest_group_size:,}"
    )
    print(sensitivity_df.to_string(index=False))

    # --------------------------------------------------------
    # 8. Paired group bootstrap
    # --------------------------------------------------------

    bootstrap_scores, paired_differences = group_bootstrap(
        y_true=y_true,
        predictions=predictions,
        group_codes=group_codes,
        number_of_groups=number_of_groups,
        number_of_classes=EXPECTED_CLASSES,
        eligible_codes=eligible_codes,
        reps=BOOTSTRAP_REPS,
        seed=BOOTSTRAP_SEED,
    )

    # --------------------------------------------------------
    # 9. Confidence intervals for model scores
    # --------------------------------------------------------

    interval_rows = []

    for name, values in bootstrap_scores.items():

        lower, upper = confidence_interval(
            values,
            CI_LEVEL,
        )

        interval_rows.append(
            {
                "model": name,
                "bootstrap_replicates": BOOTSTRAP_REPS,
                "seed": BOOTSTRAP_SEED,
                "bootstrap_unit": "group_id",
                "point_estimate": float(
                    point_df.loc[
                        point_df["model"] == name,
                        "headline_macro_f1",
                    ].iloc[0]
                ),
                "ci_level": CI_LEVEL,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )

    intervals_df = pd.DataFrame(interval_rows)

    intervals_df.to_csv(
        OUTPUT_DIR / "group_bootstrap_confidence_intervals.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 10. Paired model differences
    # --------------------------------------------------------

    difference_rows = []

    point_lookup = dict(
        zip(
            point_df["model"],
            point_df["headline_macro_f1"],
        )
    )

    point_differences = {
        "baseline_minus_balanced": (
            point_lookup["xgboost_baseline"]
            - point_lookup["xgboost_balanced"]
        ),
        "baseline_minus_sqrt_balanced": (
            point_lookup["xgboost_baseline"]
            - point_lookup["xgboost_sqrt_balanced"]
        ),
        "balanced_minus_sqrt_balanced": (
            point_lookup["xgboost_balanced"]
            - point_lookup["xgboost_sqrt_balanced"]
        ),
    }

    for name, values in paired_differences.items():

        lower, upper = confidence_interval(
            values,
            CI_LEVEL,
        )

        difference_rows.append(
            {
                "comparison": name,
                "point_difference": point_differences[name],
                "bootstrap_replicates": BOOTSTRAP_REPS,
                "bootstrap_unit": "paired_group_id",
                "ci_level": CI_LEVEL,
                "ci_lower": lower,
                "ci_upper": upper,
                "excludes_zero": bool(
                    lower > 0 or upper < 0
                ),
            }
        )

    differences_df = pd.DataFrame(difference_rows)

    differences_df.to_csv(
        OUTPUT_DIR / "paired_model_differences.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 11. Save bootstrap distributions
    # --------------------------------------------------------

    distribution_data = {}

    for name, values in bootstrap_scores.items():
        distribution_data[name] = values

    for name, values in paired_differences.items():
        distribution_data[name] = values

    pd.DataFrame(
        distribution_data
    ).to_csv(
        OUTPUT_DIR / "group_bootstrap_distributions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 12. Save reproducibility configuration
    # --------------------------------------------------------

    output_config = {
        "phase": "6.1 CIC scenario group-clustered bootstrap",
        "protocol": (
            "Existing saved multiclass XGBoost models evaluated on "
            "the unchanged cic_binary_test.csv. No training, refitting, "
            "or threshold tuning."
        ),
        "bootstrap_unit": "group_id",
        "paired_resampling": True,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_interval": (
            f"{CI_LEVEL}% percentile interval"
        ),
        "test_rows": len(test),
        "unique_test_groups": number_of_groups,
        "largest_group_size": largest_group_size,
        "eligible_classes": sorted(ELIGIBLE_CLASSES),
        "all_classes_in_confusion_matrix": classes,
        "headline_metric": (
            "Macro F1 over eligible classes with positive true "
            "support in each replicate; zero_division=0."
        ),
        "largest_group_sensitivity": True,
        "limitations": [
            "Pooled group bootstrap is not a capture-day bootstrap.",
            "Some group IDs span multiple capture files.",
            "Three rare classes remain excluded from headline macro F1.",
            "Their predictions remain represented in the full confusion matrices.",
            "Bootstrap uncertainty is conditional on this fixed test dataset "
            "and the existing trained models.",
        ],
        "input_test_sha256": sha256_of(TEST_PATH),
        "imputer_sha256": sha256_of(IMPUTER_PATH),
        "encoder_sha256": sha256_of(ENCODER_PATH),
        "model_sha256": {
            name: sha256_of(path)
            for name, path in MODELS.items()
        },
    }

    (OUTPUT_DIR / "group_bootstrap_config.json").write_text(
        json.dumps(output_config, indent=2),
        encoding="utf-8",
    )

    print("\n========== PAIRED DIFFERENCES ==========")
    print(differences_df.to_string(index=False))

    print("\n========== MODEL CONFIDENCE INTERVALS ==========")
    print(intervals_df.to_string(index=False))

    print("\n========== COMPLETED ==========")
    print(f"Outputs: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
