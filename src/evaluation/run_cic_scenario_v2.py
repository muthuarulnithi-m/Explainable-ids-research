"""
Phase 6.1 v2 - CIC scenario-based evaluation with corrected metrics.

Evaluates the three saved multiclass XGBoost models on cic_binary_test.csv per
capture file. Nothing is trained, refitted or tuned. Headline macro F1 uses only
the 12 eligible classes (>= 30 validation rows, pre-registered). Writes only to
experiments\\generalization\\cic_scenario_v2. Writes nothing if any check fails.

Run from the project root:
    python src\\evaluation\\run_cic_scenario_v2.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenario_metrics import scenario_metrics  # noqa: E402

warnings.filterwarnings("ignore", message="A single label was found")

PROCESSED = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED / "cic_binary_train.csv"
TEST_PATH = PROCESSED / "cic_binary_test.csv"

MODEL_DIR = ROOT / "experiments" / "multiclass_cic"
IMPUTER_PATH = MODEL_DIR / "multiclass_imputer.joblib"
ENCODER_PATH = MODEL_DIR / "multiclass_label_encoder.joblib"
VALIDATION_POSITIONS = MODEL_DIR / "validation_positions.npy"
CLASS_DISTRIBUTION = MODEL_DIR / "class_distribution.csv"
MODELS = {
    "xgboost_baseline": MODEL_DIR / "multiclass_xgboost.json",
    "xgboost_balanced": MODEL_DIR / "imbalance" / "xgboost_balanced.json",
    "xgboost_sqrt_balanced": MODEL_DIR / "imbalance" / "xgboost_sqrt_balanced.json",
}

OUTPUT_DIR = ROOT / "experiments" / "generalization" / "cic_scenario_v2"

METADATA_COLUMNS = ["source_file", "source_row", "group_id",
                    "normalized_label", "binary_label"]
EXPECTED_FEATURES = 78
EXPECTED_CLASSES = 15
EXPECTED_TRAIN_ROWS = 2_255_853
EXPECTED_TEST_ROWS = 567_870
POOLED = "ALL_TEST_DAYS"
BENIGN = "BENIGN"

# Pre-registered eligibility: validation rows >= 30. Fixed before this run.
ELIGIBILITY_MIN_VALIDATION_ROWS = 30
PRE_REGISTERED_ELIGIBLE = {
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest",
    "DoS slowloris", "FTP-Patator", "PortScan", "SSH-Patator",
    "Web Attack - Brute Force", "Web Attack - XSS",
}

RECORDED_VALIDATION_MACRO_F1 = {
    "xgboost_baseline": 0.787566,
    "xgboost_balanced": 0.790589,
    "xgboost_sqrt_balanced": 0.795563,
}
REPRODUCTION_TOLERANCE = 1e-5
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 42


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"\nFAILED: {message}\nNo output was written.")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cm_from_codes(t: np.ndarray, p: np.ndarray, k: int, weights=None) -> np.ndarray:
    pair = t * k + p
    return np.bincount(pair, weights=weights, minlength=k * k).reshape(k, k).astype(float)


def headline_from_cm(cm: np.ndarray, eligible_mask: np.ndarray) -> float:
    """Macro F1 over eligible classes with true support > 0 (zero_division = 0)."""
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    present = eligible_mask & (support > 0)
    return float(f1[present].mean()) if present.any() else float("nan")


def bootstrap_headline(t, p, eligible_mask, reps, seed) -> np.ndarray:
    k = len(eligible_mask)
    n = len(t)
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        weights = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        out[r] = headline_from_cm(cm_from_codes(t, p, k, weights), eligible_mask)
    return out


def load_validation_rows(feature_names: list[str]) -> pd.DataFrame:
    positions = np.load(VALIDATION_POSITIONS)
    keep = np.zeros(EXPECTED_TRAIN_ROWS, dtype=bool)
    keep[positions] = True
    parts, offset = [], 0
    for chunk in pd.read_csv(TRAIN_PATH, usecols=feature_names + ["normalized_label"],
                             chunksize=200_000):
        n = len(chunk)
        parts.append(chunk.loc[keep[offset:offset + n]])
        offset += n
    require(offset == EXPECTED_TRAIN_ROWS,
            f"training file has {offset} rows, expected {EXPECTED_TRAIN_ROWS}")
    validation = pd.concat(parts, ignore_index=True)
    require(len(validation) == len(positions),
            f"loaded {len(validation)} validation rows, expected {len(positions)}")
    return validation


def main() -> None:
    print("=" * 70)
    print("PHASE 6.1 v2 - CIC SCENARIO EVALUATION (corrected metrics)")
    print("=" * 70)

    # 1. Files and output folder
    for path in [TRAIN_PATH, TEST_PATH, IMPUTER_PATH, ENCODER_PATH,
                 VALIDATION_POSITIONS, CLASS_DISTRIBUTION, *MODELS.values()]:
        require(path.exists(), f"missing file: {path}")
    require(not (OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir())),
            f"output folder already has content: {OUTPUT_DIR}")
    print("[PASS] required files exist; cic_scenario_v2 is new")

    # 2. Eligibility: validation support only, checked against the pre-registered set
    dist = pd.read_csv(CLASS_DISTRIBUTION)
    val_counts = dict(zip(dist["class_name"], dist["validation_count"]))
    for threshold in range(8, 95):
        at_threshold = {c for c, n in val_counts.items() if n >= threshold}
        require(at_threshold == PRE_REGISTERED_ELIGIBLE,
                f"eligible set changes at threshold {threshold}")
    eligible = {c for c, n in val_counts.items() if n >= ELIGIBILITY_MIN_VALIDATION_ROWS}
    require(eligible == PRE_REGISTERED_ELIGIBLE,
            f"eligible set {sorted(eligible)} differs from the pre-registered set")
    print("[PASS] eligibility matches pre-registered 12 classes; "
          "unchanged for thresholds 8-94")

    # 3. Saved preprocessing and labels
    imputer = joblib.load(IMPUTER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    require(len(getattr(imputer, "statistics_", [])) == EXPECTED_FEATURES,
            f"imputer statistics do not cover {EXPECTED_FEATURES} features")
    if hasattr(imputer, "feature_names_in_"):
        feature_names = list(imputer.feature_names_in_)
        order_source = "imputer.feature_names_in_"
    else:
        feature_names = [c for c in pd.read_csv(TEST_PATH, nrows=0).columns
                         if c not in METADATA_COLUMNS]
        order_source = ("non-metadata columns in file order; verified by the "
                        "reproduction check")
    require(len(feature_names) == EXPECTED_FEATURES,
            f"found {len(feature_names)} features, expected {EXPECTED_FEATURES}")
    require(len(encoder.classes_) == EXPECTED_CLASSES, "encoder class count mismatch")
    classes = [str(c) for c in encoder.classes_]
    eligible_mask = np.array([c in eligible for c in classes])
    require(int(eligible_mask.sum()) == 12, "eligible mask does not cover 12 classes")
    print(f"[PASS] imputer and encoder: {len(feature_names)} features, "
          f"{len(classes)} classes; order source: {order_source}")

    # 4. Test file checks
    test = pd.read_csv(TEST_PATH)
    require(len(test) == EXPECTED_TEST_ROWS,
            f"test file has {len(test)} rows, expected {EXPECTED_TEST_ROWS}")
    missing = [c for c in feature_names if c not in test.columns]
    require(not missing, f"test file lacks features: {missing[:5]}")
    x_raw = test[feature_names].astype(float)
    infinite = int(np.isinf(x_raw.to_numpy()).sum())
    require(infinite == 0, f"{infinite} infinite values in test features")
    unknown = set(test["normalized_label"].unique()) - set(classes)
    require(not unknown, f"labels unknown to encoder: {unknown}")
    print(f"[PASS] test file: {len(test):,} rows, {int(x_raw.isna().sum().sum()):,} "
          "NaN cells (imputed), 0 infinite, all labels known")

    train_groups = set(pd.read_csv(TRAIN_PATH, usecols=["group_id"])["group_id"].unique())
    shared = train_groups & set(test["group_id"].unique())
    require(not shared, f"{len(shared)} group_id values shared between train and test")
    print("[PASS] zero group_id overlap between training and test files")

    x_test = pd.DataFrame(imputer.transform(x_raw), columns=feature_names)
    y_test = test["normalized_label"].to_numpy(dtype=object)
    y_test_codes = encoder.transform(y_test)
    scenarios = (test["source_file"].str.replace(".pcap_ISCX.csv", "", regex=False)
                 .to_numpy(dtype=object))
    k = len(classes)

    # 5. Validation rows
    validation = load_validation_rows(feature_names)
    x_val = pd.DataFrame(imputer.transform(validation[feature_names].astype(float)),
                         columns=feature_names)
    y_val = validation["normalized_label"].to_numpy(dtype=object)
    y_val_codes = encoder.transform(y_val)
    eligible_codes = np.where(eligible_mask)[0]
    print(f"[PASS] validation rows reloaded: {len(validation):,}")

    summary_rows, class_frames, interval_rows, model_meta = [], [], [], {}
    confusion_dir = OUTPUT_DIR / "confusion_matrices"
    pooled_mask = np.ones(len(y_test), dtype=bool)

    for name, path in MODELS.items():
        model = xgboost.XGBClassifier()
        model.load_model(str(path))

        stored = model.get_booster().feature_names
        if stored is None:
            feature_status = "not stored in model file; order verified by reproduction"
        else:
            require(list(stored) == feature_names,
                    f"{name}: stored feature names differ from imputer order")
            feature_status = "stored feature names match"

        # 6. Reproduction: all-15 validation macro F1 must match Phase 5
        val_codes = np.asarray(model.predict(x_val)).astype(int)
        require(val_codes.min() >= 0 and val_codes.max() < k,
                f"{name}: validation prediction codes out of range")
        val_all15 = f1_score(y_val_codes, val_codes, labels=list(range(k)),
                             average="macro", zero_division=0)
        recorded = RECORDED_VALIDATION_MACRO_F1[name]
        require(abs(val_all15 - recorded) <= REPRODUCTION_TOLERANCE,
                f"{name}: reproduced {val_all15:.6f}, recorded {recorded:.6f}")

        # 7. Validation headline: own calculation must match scikit-learn
        val_point = headline_from_cm(cm_from_codes(y_val_codes, val_codes, k), eligible_mask)
        val_sklearn = f1_score(y_val_codes, val_codes, labels=list(eligible_codes),
                               average="macro", zero_division=0)
        require(abs(val_point - val_sklearn) <= 1e-9,
                f"{name}: validation headline {val_point} != scikit-learn {val_sklearn}")

        val_boot = bootstrap_headline(y_val_codes, val_codes, eligible_mask,
                                      BOOTSTRAP_REPS, BOOTSTRAP_SEED)
        interval_rows.append({
            "model": name, "dataset": "validation", "point_estimate": val_point,
            "ci_low_2.5": float(np.nanpercentile(val_boot, 2.5)),
            "ci_high_97.5": float(np.nanpercentile(val_boot, 97.5)),
            "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
        })

        # 8. Test predictions (default argmax, no threshold tuning)
        test_codes = np.asarray(model.predict(x_test)).astype(int)
        require(test_codes.min() >= 0 and test_codes.max() < k,
                f"{name}: test prediction codes out of range")
        test_pred = encoder.inverse_transform(test_codes).astype(object)

        test_point = headline_from_cm(cm_from_codes(y_test_codes, test_codes, k), eligible_mask)
        test_boot = bootstrap_headline(y_test_codes, test_codes, eligible_mask,
                                       BOOTSTRAP_REPS, BOOTSTRAP_SEED)
        interval_rows.append({
            "model": name, "dataset": "test_pooled", "point_estimate": test_point,
            "ci_low_2.5": float(np.nanpercentile(test_boot, 2.5)),
            "ci_high_97.5": float(np.nanpercentile(test_boot, 97.5)),
            "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
        })

        # 9. Per scenario, with a pooled reference row
        for scenario in [*sorted(set(scenarios)), POOLED]:
            mask = pooled_mask if scenario == POOLED else (scenarios == scenario)
            yt, yp = y_test[mask], test_pred[mask]
            has_attacks = bool(set(yt.tolist()) - {BENIGN})
            m = scenario_metrics(yt, yp, eligible, benign_label=BENIGN)

            if scenario == POOLED:
                require(abs(m["headline_macro_f1"] - test_point) <= 1e-9,
                        f"{name}: pooled headline disagrees with bootstrap point estimate")

            summary_rows.append({
                "model": name,
                "scenario": scenario,
                "rows": m["rows"],
                "accuracy": m["accuracy"],
                "primary_metric": "headline_macro_f1" if has_attacks
                                  else "benign_false_positive_rate",
                "headline_macro_f1": m["headline_macro_f1"] if has_attacks else float("nan"),
                "union_macro_f1_diagnostic": m["union_macro_f1"],
                "eligible_classes_present": m["eligible_classes_present"],
                "not_evaluable_classes_present": ";".join(m["not_evaluable_classes_present"]),
                "absent_class_predictions": m["absent_class_predictions"],
                "benign_recall": m["benign_recall"],
                "benign_false_positive_rate": m["benign_false_positive_rate"],
            })

            pc = m["per_class"].copy()
            pc.insert(0, "scenario", scenario)
            pc.insert(0, "model", name)
            class_frames.append(pc)

            confusion_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(m["confusion_matrix"], index=m["confusion_labels"],
                         columns=m["confusion_labels"]).to_csv(
                confusion_dir / f"{name}__{scenario}.csv")

        model_meta[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_of(path),
            "feature_status": feature_status,
            "validation_all15_reproduced": round(float(val_all15), 6),
            "validation_all15_recorded": recorded,
        }

    # 10. Outputs (only reached if every check passed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    distribution = (pd.DataFrame({"scenario": scenarios, "class": y_test})
                    .groupby(["scenario", "class"]).size().reset_index(name="test_rows"))
    distribution["eligible"] = distribution["class"].isin(eligible)
    distribution.to_csv(OUTPUT_DIR / "scenario_distribution.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "scenario_metrics_v2.csv", index=False)
    pd.concat(class_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "per_class_v2.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(OUTPUT_DIR / "bootstrap_intervals.csv", index=False)

    config = {
        "phase": "6.1 CIC scenario-based evaluation, v2 (corrected metrics; main result A)",
        "protocol": ("Saved multiclass XGBoost models evaluated per capture file on "
                     "cic_binary_test.csv. No training, refitting or threshold tuning."),
        "eligibility_rule": (f"class eligible if validation_count >= "
                             f"{ELIGIBILITY_MIN_VALIDATION_ROWS}, from class_distribution.csv"),
        "pre_registered_eligible_classes": sorted(PRE_REGISTERED_ELIGIBLE),
        "eligibility_disclosure": ("Run-1 per-class test results were examined before this rule "
                                   "was written. The rule uses validation counts only and is "
                                   "unchanged for thresholds 8-94."),
        "headline_definition": ("macro F1 over eligible classes present in the scenario's truth; "
                                "zero_division = 0"),
        "diagnostics": ["union_macro_f1_diagnostic", "absent_class_predictions",
                        "benign_recall", "benign_false_positive_rate"],
        "benign_only_scenarios": "primary metric is benign_false_positive_rate; headline is NaN",
        "not_evaluable_classes": "Heartbleed, Infiltration, Web Attack - SQL Injection: reported "
                                 "in per_class_v2.csv with eligible = False, not averaged",
        "bootstrap": {"unit": "test flow (pooled) and validation row", "reps": BOOTSTRAP_REPS,
                      "seed": BOOTSTRAP_SEED, "interval": "2.5th and 97.5th percentiles"},
        "features": feature_names,
        "feature_order_source": order_source,
        "classes": classes,
        "input_files": {
            "test": {"path": str(TEST_PATH.relative_to(ROOT)),
                     "sha256": sha256_of(TEST_PATH), "rows": len(test)},
            "train": {"path": str(TRAIN_PATH.relative_to(ROOT)),
                      "sha256": sha256_of(TRAIN_PATH), "rows": EXPECTED_TRAIN_ROWS},
            "class_distribution": {"path": str(CLASS_DISTRIBUTION.relative_to(ROOT)),
                                   "sha256": sha256_of(CLASS_DISTRIBUTION)},
        },
        "imputer_sha256": sha256_of(IMPUTER_PATH),
        "label_encoder_sha256": sha256_of(ENCODER_PATH),
        "models": model_meta,
        "random_seed": f"bootstrap seed {BOOTSTRAP_SEED}; training seed 42 in multiclass_config.json",
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "Each capture file is a scenario; several attack classes occur in one file only.",
            "Models were trained on all capture days; this is per-day evaluation, not scenario-independent training.",
            "Monday is benign-only; it gives false-positive information only.",
            "cic_binary_test.csv was used to compare binary LR and RF baselines; it is not a clean holdout for binary work.",
            "Three classes are not evaluable; the training sample holds 1, 3 and 3 flows for them.",
        ],
    }
    with open(OUTPUT_DIR / "scenario_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)

    print("\nHEADLINE (pooled test, 12 eligible classes)")
    pooled = pd.DataFrame(summary_rows)
    print(pooled[pooled.scenario == POOLED][["model", "headline_macro_f1",
          "union_macro_f1_diagnostic"]].round(4).to_string(index=False))
    print("\nBOOTSTRAP INTERVALS")
    print(pd.DataFrame(interval_rows).round(4).to_string(index=False))
    print(f"\nOutput: {OUTPUT_DIR}")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
