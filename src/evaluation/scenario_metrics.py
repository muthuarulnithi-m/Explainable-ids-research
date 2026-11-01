"""
Metric definitions for Phase 6.1 v2. Pure functions, no file I/O.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def scenario_metrics(y_true, y_pred, eligible_classes, benign_label="BENIGN"):
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    truth_labels = sorted(set(y_true.tolist()))
    union_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    eligible_present = [c for c in truth_labels if c in eligible_classes]
    not_evaluable = [c for c in truth_labels if c not in eligible_classes]

    # Headline: eligible classes present in the truth, nothing else.
    headline = (
        f1_score(y_true, y_pred, labels=eligible_present,
                 average="macro", zero_division=0)
        if eligible_present else float("nan")
    )

    # Diagnostic: macro over the union of truth and predicted labels.
    union_macro = f1_score(y_true, y_pred, labels=union_labels,
                           average="macro", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=truth_labels, zero_division=0)
    per_class = pd.DataFrame({
        "class": truth_labels,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eligible": [c in eligible_classes for c in truth_labels],
    })

    absent_predictions = int(np.sum(~np.isin(y_pred, truth_labels)))

    benign_mask = y_true == benign_label
    if benign_mask.any():
        benign_recall = float(np.mean(y_pred[benign_mask] == benign_label))
        benign_fpr = 1.0 - benign_recall
    else:
        benign_recall = benign_fpr = float("nan")

    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "headline_macro_f1": float(headline),
        "union_macro_f1": float(union_macro),
        "eligible_classes_present": len(eligible_present),
        "not_evaluable_classes_present": not_evaluable,
        "absent_class_predictions": absent_predictions,
        "benign_recall": benign_recall,
        "benign_false_positive_rate": benign_fpr,
        "confusion_labels": union_labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=union_labels),
        "per_class": per_class,
    }
