
import numpy as np

from src.evaluation.run_cic_scenario_group_bootstrap import (
    confusion_matrix_codes,
    headline_macro_f1,
    confidence_interval,
)


def test_confusion_matrix_counts_all_classes():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0])

    matrix = confusion_matrix_codes(
        y_true,
        y_pred,
        number_of_classes=3,
    )

    expected = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ])

    np.testing.assert_array_equal(matrix, expected)


def test_headline_uses_only_eligible_classes_with_true_support():
    matrix = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ])

    eligible = np.array([0, 1])

    # Class 0 F1 = 1/2
    # Class 1 F1 = 2/3
    score = headline_macro_f1(matrix, eligible)

    expected = (0.5 + 2 / 3) / 2

    assert np.isclose(score, expected)


def test_ineligible_prediction_still_counts_against_eligible_class():
    matrix = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
    ])

    eligible = np.array([0, 1])

    score = headline_macro_f1(matrix, eligible)

    # Class 0 F1 = 2/3; class 1 F1 = 1.
    expected = (2 / 3 + 1.0) / 2

    assert np.isclose(score, expected)


def test_group_weights_duplicate_all_flows_in_selected_group():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 0])

    group_codes = np.array([0, 0, 1])
    group_multiplicity = np.array([2, 1])

    row_weights = group_multiplicity[group_codes]

    matrix = confusion_matrix_codes(
        y_true,
        y_pred,
        number_of_classes=2,
        sample_weights=row_weights,
    )

    expected = np.array([
        [2, 0],
        [1, 2],
    ])

    np.testing.assert_array_equal(matrix, expected)


def test_confidence_interval():
    values = np.arange(1000, dtype=float)

    lower, upper = confidence_interval(values, 95)

    assert lower < upper
    assert lower >= 0
    assert upper <= 999
