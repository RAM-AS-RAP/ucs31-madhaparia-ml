"""Tests for abstention decisions and accepted-prediction statistics."""

import numpy as np
import pytest

from scripts.candidate_uncertainty_policy import (
    assess_probability_rows,
    summarize_acceptance,
    wilson_interval,
)


def test_wilson_interval_matches_known_example():
    lower, upper = wilson_interval(95, 100)

    assert lower == pytest.approx(0.8882495308)
    assert upper == pytest.approx(0.9784563208)

    # Observed accuracy of 95% does not itself establish a lower bound of 90%.
    assert lower < 0.90


@pytest.mark.parametrize("successes", [0, 100])
def test_wilson_endpoints_stay_within_probability_range(successes):
    lower, upper = wilson_interval(successes, 100)

    assert 0 <= lower <= upper <= 1


def test_wilson_with_no_observations_is_undefined():
    assert wilson_interval(0, 0) == (None, None)


@pytest.mark.parametrize(
    "successes,total",
    [(2, 1), (-1, 10), (0, -1), (True, 10)],
)
def test_invalid_wilson_counts_are_rejected(successes, total):
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


def test_zero_and_one_unit_inputs_are_withheld_even_with_high_scores():
    result = assess_probability_rows(
        [[0.99, 0.01], [0.99, 0.01], [0.99, 0.01]],
        [0, 1, 2],
        min_score=0.9,
        min_margin=0.1,
    )

    np.testing.assert_array_equal(
        result["accepted"], [False, False, True]
    )
    assert result["score_passed"].all()
    assert result["margin_passed"].all()


def test_score_and_margin_requirements_are_both_enforced():
    result = assess_probability_rows(
        [
            [0.75, 0.25],
            [0.625, 0.375],
            [0.5, 0.5],
        ],
        [2, 2, 2],
        min_score=0.625,
        min_margin=0.5,
    )

    np.testing.assert_array_equal(
        result["score_passed"], [True, True, False]
    )
    np.testing.assert_array_equal(
        result["margin_passed"], [True, False, False]
    )
    np.testing.assert_array_equal(
        result["accepted"], [True, False, False]
    )


def test_threshold_equality_is_accepted():
    result = assess_probability_rows(
        [[0.75, 0.25]],
        [2],
        min_score=0.75,
        min_margin=0.5,
    )

    assert result["accepted"][0]


@pytest.mark.parametrize("units", [[1.5], [np.nan], [-1]])
def test_invalid_reporting_unit_counts_are_rejected(units):
    with pytest.raises(ValueError, match="Reporting-unit"):
        assess_probability_rows(
            [[0.9, 0.1]],
            units,
            min_score=0.8,
            min_margin=0.1,
        )


@pytest.mark.parametrize(
    "score,margin",
    [(1.1, 0.1), (0.8, -0.1), (True, 0.1), (0.8, np.nan)],
)
def test_invalid_thresholds_are_rejected(score, margin):
    with pytest.raises(ValueError, match="Thresholds"):
        assess_probability_rows(
            [[0.9, 0.1]],
            [2],
            min_score=score,
            min_margin=margin,
        )


def test_summary_uses_accepted_cases_for_selective_accuracy():
    report = summarize_acceptance(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
    )

    assert report["observations"] == 4
    assert report["accepted"] == 2
    assert report["withheld"] == 2
    assert report["coverage"] == 0.5
    assert report["correct_accepted"] == 1
    assert report["incorrect_accepted"] == 1
    assert report["accepted_accuracy"] == 0.5
    assert report["accepted_error_rate"] == 0.5
    assert report["correct_accepted_fraction_of_all"] == 0.25


def test_all_withheld_does_not_report_perfect_accuracy():
    report = summarize_acceptance(
        np.array([True, False]),
        np.array([False, False]),
    )

    assert report["coverage"] == 0.0
    assert report["abstention_rate"] == 1.0
    assert report["accepted_accuracy"] is None
    assert report["accepted_error_rate"] is None
    assert report["accepted_accuracy_wilson_lower"] is None
    assert report["accepted_accuracy_wilson_upper"] is None


def test_empty_subgroup_has_undefined_rates():
    report = summarize_acceptance(
        np.array([], dtype=bool),
        np.array([], dtype=bool),
    )

    assert report["observations"] == 0
    assert report["coverage"] is None
    assert report["accepted_accuracy"] is None


def test_numeric_flags_are_rejected_instead_of_silently_converted():
    with pytest.raises(ValueError, match="boolean arrays"):
        summarize_acceptance([1, 0], [1, 1])