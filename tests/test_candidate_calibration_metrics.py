"""Tests for calibration measurements and representation selection."""

import numpy as np
import pytest

from scripts.candidate_calibration_metrics import (
    SETTINGS,
    evaluate_probability_view,
    select_probability_representation,
)
from scripts.candidate_probability_calibration import temperature_scale


def test_metrics_match_hand_calculated_multiclass_example():
    report = evaluate_probability_view(
        ["A", "B"],
        [[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]],
        ["A", "B", "C"],
    )
    summary = report["summary"]

    assert summary["observations"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["log_loss"] == pytest.approx(
        -(np.log(0.7) + np.log(0.3)) / 2
    )

    # First row: 0.09 + 0.04 + 0.01 = 0.14.
    # Second row: 0.04 + 0.49 + 0.25 = 0.78.
    assert summary["multiclass_brier_score"] == pytest.approx(0.46)
    assert summary["top_label_ece"] == pytest.approx(0.4)
    assert summary["mean_top_score"] == pytest.approx(0.6)


def test_reliability_bins_retain_counts_and_empty_bins():
    report = evaluate_probability_view(
        ["A", "B"],
        [[0.8, 0.2], [0.82, 0.18]],
        ["A", "B"],
    )
    bins = report["reliability_bins"]

    assert len(bins) == 10
    assert sum(row["count"] for row in bins) == 2
    assert bins[8]["count"] == 2
    assert bins[8]["correct_count"] == 1
    assert bins[8]["accuracy"] == 0.5
    assert bins[8]["mean_top_score"] == pytest.approx(0.81)
    assert report["summary"]["top_label_ece"] == pytest.approx(0.31)

    assert bins[0]["count"] == 0
    assert bins[0]["accuracy"] is None
    assert bins[0]["mean_top_score"] is None
    assert bins[0]["absolute_gap"] is None


def test_probability_one_is_included_in_last_bin():
    report = evaluate_probability_view(
        ["A", "B"],
        [[1.0, 0.0], [0.0, 1.0]],
        ["A", "B"],
    )

    assert report["reliability_bins"][9]["count"] == 2
    assert report["summary"]["accuracy"] == 1.0
    assert report["summary"]["log_loss"] == 0.0
    assert report["summary"]["multiclass_brier_score"] == 0.0
    assert report["summary"]["top_label_ece"] == 0.0


def test_temperature_scaling_preserves_classification_metrics():
    probabilities = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.3, 0.5],
        [0.1, 0.6, 0.3],
    ])
    actual = ["A", "B", "B"]
    classes = ["A", "B", "C"]

    original = evaluate_probability_view(actual, probabilities, classes)
    scaled = evaluate_probability_view(
        actual,
        temperature_scale(probabilities, 0.7),
        classes,
    )

    for metric in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ):
        assert original["summary"][metric] == scaled["summary"][metric]

    assert original["per_condition"] == scaled["per_condition"]
    assert original["confusion_matrix"] == scaled["confusion_matrix"]


def test_probability_columns_follow_supplied_class_order():
    report = evaluate_probability_view(
        ["B", "A"],
        [[0.9, 0.1], [0.2, 0.8]],
        ["B", "A"],
    )

    assert report["summary"]["accuracy"] == 1.0
    assert report["confusion_labels"] == ["B", "A"]


@pytest.mark.parametrize(
    "classes",
    [["A"], ["A", "A"], ["A", ""]],
)
def test_invalid_class_definitions_are_rejected(classes):
    with pytest.raises(ValueError, match="class definitions"):
        evaluate_probability_view(["A"], [[0.8, 0.2]], classes)


def test_unknown_actual_label_is_rejected():
    with pytest.raises(ValueError, match="Unexpected actual"):
        evaluate_probability_view(
            ["Unknown"],
            [[0.8, 0.2]],
            ["A", "B"],
        )


def test_temperature_is_selected_when_improvement_exceeds_requirement():
    original = {setting: 0.8 for setting in SETTINGS}
    scaled = {setting: 0.7 for setting in SETTINGS}

    result = select_probability_representation(original, scaled)

    assert result["selected_representation"] == "temperature_scaled"
    assert result["log_loss_improvement"] == pytest.approx(0.1)
    assert result["deployment_approved"] is False


@pytest.mark.parametrize("scaled_loss", [0.8, 0.9, 0.8 - 0.5e-6])
def test_original_is_retained_without_sufficient_improvement(scaled_loss):
    result = select_probability_representation(
        {setting: 0.8 for setting in SETTINGS},
        {setting: scaled_loss for setting in SETTINGS},
    )

    assert result["selected_representation"] == "original"


def test_missing_view_is_rejected():
    with pytest.raises(ValueError, match="all five views"):
        select_probability_representation(
            {"full": 0.8},
            {setting: 0.7 for setting in SETTINGS},
        )


def test_nonfinite_log_loss_is_rejected():
    original = {setting: 0.8 for setting in SETTINGS}
    scaled = {setting: 0.7 for setting in SETTINGS}
    scaled["full"] = np.nan

    with pytest.raises(ValueError, match="Invalid log losses"):
        select_probability_representation(original, scaled)