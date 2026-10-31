"""Tests for withholding public conclusions from uncertain predictions."""

from copy import deepcopy

import pytest

from inference.candidate_assessment import assess_candidate_scores


@pytest.fixture
def policy():
    return {
        "policy_id": "test_research_policy",
        "classes": ["A", "B"],
        "mode": "thresholds",
        "min_reporting_units": 2,
        "min_score": 0.78,
        "min_margin": 0.20,
        "probability_representation": "original",
        "temperature": None,
    }


def test_accepted_prediction_returns_leading_condition(policy):
    result = assess_candidate_scores(
        ["A", "B"], [0.9, 0.1], 2, policy
    )

    assert result["status"] == "prediction_available"
    assert result["predicted_condition"] == "A"
    assert result["model_score"] == pytest.approx(0.9)
    assert result["uncertain"] is False
    assert result["uncertainty_reasons"] == []
    assert "diagnostics" not in result


def test_low_score_returns_no_public_condition_or_score(policy):
    result = assess_candidate_scores(
        ["A", "B"], [0.6, 0.4], 2, policy
    )

    assert result["status"] == "insufficient_information"
    assert result["predicted_condition"] is None
    assert result["model_score"] is None
    assert result["uncertain"] is True
    assert "below_score_threshold" in result["uncertainty_reasons"]
    assert "diagnostics" not in result


@pytest.mark.parametrize("unit_count", [0, 1])
def test_limited_information_withholds_even_a_high_score(policy, unit_count):
    result = assess_candidate_scores(
        ["A", "B"], [0.99, 0.01], unit_count, policy
    )

    assert result["predicted_condition"] is None
    assert result["model_score"] is None
    assert "too_few_supported_reporting_units" in result["uncertainty_reasons"]


def test_calibration_is_applied_before_threshold_checks(policy):
    policy.update({
        "probability_representation": "temperature_scaled",
        "temperature": 0.5,
    })

    result = assess_candidate_scores(
        ["A", "B"], [0.7, 0.3], 2, policy
    )

    # T=0.5 gives 0.7^2 / (0.7^2 + 0.3^2).
    expected = 0.49 / 0.58
    assert result["status"] == "prediction_available"
    assert result["model_score"] == pytest.approx(expected)


def test_diagnostics_do_not_restore_a_withheld_public_prediction(policy):
    result = assess_candidate_scores(
        ["A", "B"],
        [0.6, 0.4],
        2,
        policy,
        include_diagnostics=True,
    )

    assert result["predicted_condition"] is None
    assert result["model_score"] is None
    assert result["diagnostics"]["research_only"] is True
    assert result["diagnostics"]["ranked_conditions"][0]["condition"] == "A"


def test_withhold_all_policy_never_returns_a_leading_condition(policy):
    policy.update({
        "mode": "withhold_all",
        "min_score": None,
        "min_margin": None,
    })

    result = assess_candidate_scores(
        ["A", "B"], [0.99, 0.01], 5, policy
    )

    assert result["status"] == "insufficient_information"
    assert result["predicted_condition"] is None
    assert "no_qualifying_research_policy" in result["uncertainty_reasons"]


def test_policy_is_not_modified(policy):
    original = deepcopy(policy)

    assess_candidate_scores(["A", "B"], [0.9, 0.1], 2, policy)

    assert policy == original


def test_class_order_mismatch_is_rejected(policy):
    with pytest.raises(ValueError, match="frozen policy order"):
        assess_candidate_scores(
            ["B", "A"], [0.9, 0.1], 2, policy
        )


def test_invalid_probability_distribution_is_rejected(policy):
    with pytest.raises(ValueError):
        assess_candidate_scores(
            ["A", "B"], [0.9, 0.9], 2, policy
        )


def test_unexpected_probability_representation_is_rejected(policy):
    policy["probability_representation"] = "unknown"

    with pytest.raises(ValueError, match="Unknown probability"):
        assess_candidate_scores(
            ["A", "B"], [0.9, 0.1], 2, policy
        )


def test_temperature_outside_protocol_bounds_is_rejected(policy):
    policy.update({
        "probability_representation": "temperature_scaled",
        "temperature": 10.0,
    })

    with pytest.raises(ValueError, match="Invalid frozen temperature"):
        assess_candidate_scores(
            ["A", "B"], [0.9, 0.1], 2, policy
        )