"""Tests that keep withheld cases visible in policy evaluation."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.candidate_selective_evaluation import evaluate_policy_view


@pytest.fixture
def example():
    return {
        "actual": ["A", "A", "B", "B", "C"],
        "probabilities": np.array([
            [0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05],
            [0.20, 0.60, 0.20],
            [0.05, 0.05, 0.90],
        ]),
        "classes": ["A", "B", "C"],
        "reported_unit_counts": [2, 1, 2, 2, 2],
        "policy": {
            "classes": ["A", "B", "C"],
            "mode": "thresholds",
            "min_reporting_units": 2,
            "min_score": 0.78,
            "min_margin": 0.20,
            "minimum_accepted_per_required_view": 100,
            "minimum_accuracy_wilson_lower": 0.90,
        },
    }


def test_overall_report_separates_coverage_from_accuracy(example):
    report, accepted = evaluate_policy_view(**example)

    np.testing.assert_array_equal(
        accepted, [True, False, True, False, True]
    )

    summary = report["summary"]
    assert summary["observations"] == 5
    assert summary["accepted"] == 3
    assert summary["withheld"] == 2
    assert summary["coverage"] == pytest.approx(0.6)
    assert summary["accepted_accuracy"] == pytest.approx(2 / 3)
    assert summary["unfiltered_accuracy"] == pytest.approx(0.8)
    assert summary["correct_accepted_fraction_of_all"] == pytest.approx(0.4)
    assert summary["meets_count_and_accuracy_targets"] is False


def test_per_condition_recall_keeps_withheld_cases_in_all_case_measure(example):
    report, _ = evaluate_policy_view(**example)
    rows = {row["condition"]: row for row in report["per_condition"]}

    condition = rows["A"]
    assert condition["support_all"] == 2
    assert condition["accepted_true_cases"] == 1
    assert condition["withheld_true_cases"] == 1

    assert condition["accepted_recall"] == 1.0
    assert condition["correct_accepted_recall_all_cases"] == 0.5

    # One accepted B case was incorrectly predicted as A.
    assert condition["accepted_precision"] == 0.5
    assert condition["accepted_predictions_of_condition"] == 2

    assert rows["B"]["correct_accepted"] == 0
    assert rows["B"]["accepted_recall"] == 0.0
    assert rows["B"]["correct_accepted_recall_all_cases"] == 0.0
    assert rows["B"]["accepted_precision"] is None


def test_reporting_unit_subgroups_reconcile_with_overall_counts(example):
    report, _ = evaluate_policy_view(**example)
    rows = report["by_reporting_units"]

    assert sum(row["observations"] for row in rows) == 5
    assert sum(row["accepted"] for row in rows) == 3
    assert sum(row["withheld"] for row in rows) == 2

    one_unit = next(
        row for row in rows if row["reported_unit_count"] == 1
    )
    assert one_unit["accepted"] == 0
    assert one_unit["accepted_accuracy"] is None


def test_withhold_all_policy_has_no_selective_accuracy(example):
    example["policy"].update({
        "mode": "withhold_all",
        "min_score": None,
        "min_margin": None,
    })

    report, accepted = evaluate_policy_view(**example)

    assert not accepted.any()
    assert report["summary"]["coverage"] == 0.0
    assert report["summary"]["accepted_accuracy"] is None
    assert report["accepted_classification"] is None

    for row in report["per_condition"]:
        assert row["withheld_true_cases"] == row["support_all"]
        assert row["correct_accepted_recall_all_cases"] == 0.0


def test_condition_without_true_support_has_undefined_recall(example):
    example["actual"][-1] = "A"
    report, _ = evaluate_policy_view(**example)

    condition = next(
        row for row in report["per_condition"]
        if row["condition"] == "C"
    )
    assert condition["support_all"] == 0
    assert condition["accepted_precision"] == 0.0
    assert condition["accepted_recall"] is None
    assert condition["correct_accepted_recall_all_cases"] is None


def test_policy_is_not_modified(example):
    original = deepcopy(example["policy"])

    evaluate_policy_view(**example)

    assert example["policy"] == original


def test_class_order_mismatch_is_rejected(example):
    example["policy"]["classes"] = ["B", "A", "C"]

    with pytest.raises(ValueError, match="Class order"):
        evaluate_policy_view(**example)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "unknown"),
        ("min_reporting_units", 1),
        ("minimum_accepted_per_required_view", 50),
        ("minimum_accuracy_wilson_lower", 0.80),
    ],
)
def test_changed_protocol_rules_are_rejected(example, field, value):
    example["policy"][field] = value

    with pytest.raises(ValueError):
        evaluate_policy_view(**example)