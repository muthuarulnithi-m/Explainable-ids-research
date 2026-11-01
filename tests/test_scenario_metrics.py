import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))
from scenario_metrics import scenario_metrics  # noqa: E402


def test_hand_checked_case():
    # Truth [BENIGN, BENIGN, DDoS]; prediction [BENIGN, DDoS, PortScan]
    m = scenario_metrics(
        ["BENIGN", "BENIGN", "DDoS"],
        ["BENIGN", "DDoS", "PortScan"],
        eligible_classes={"BENIGN", "DDoS"},
    )
    assert m["headline_macro_f1"] == pytest.approx(1 / 3)   # (2/3 + 0) / 2
    assert m["union_macro_f1"] == pytest.approx(2 / 9)      # (2/3 + 0 + 0) / 3
    assert m["absent_class_predictions"] == 1
    assert m["benign_recall"] == pytest.approx(0.5)
    assert m["benign_false_positive_rate"] == pytest.approx(0.5)


def test_perfect_prediction_gives_one():
    m = scenario_metrics(["BENIGN", "Bot", "Bot"], ["BENIGN", "Bot", "Bot"],
                         eligible_classes={"BENIGN", "Bot"})
    assert m["headline_macro_f1"] == pytest.approx(1.0)
    assert m["absent_class_predictions"] == 0


def test_only_not_evaluable_classes_gives_nan():
    m = scenario_metrics(["Heartbleed"], ["Heartbleed"],
                         eligible_classes={"BENIGN", "Bot"})
    assert m["headline_macro_f1"] != m["headline_macro_f1"]  # NaN
    assert m["not_evaluable_classes_present"] == ["Heartbleed"]


def test_benign_only_day_reports_false_positive_rate():
    # Monday-style: four benign flows, two wrongly flagged.
    m = scenario_metrics(
        ["BENIGN", "BENIGN", "BENIGN", "BENIGN"],
        ["BENIGN", "Bot", "BENIGN", "DDoS"],
        eligible_classes={"BENIGN", "Bot", "DDoS"},
    )
    assert m["benign_recall"] == pytest.approx(0.5)
    assert m["benign_false_positive_rate"] == pytest.approx(0.5)
    assert m["absent_class_predictions"] == 2
