"""Verify public-response parity with the evaluated assessment."""

import copy

import numpy as np
import pytest

from respiratory_candidate_runtime.assessment import (
    assess_candidate_scores,
)
from inference.candidate_assessment import (
    assess_candidate_scores as reference_assess,
)


FROZEN_TEMPERATURE = 0.6408387217626598


def make_policy(representation="temperature_scaled"):
    return {
        "policy_id": "candidate_v1_research_uncertainty_v1",
        "classes": ["A", "B"],
        "mode": "thresholds",
        "min_reporting_units": 2,
        "min_score": 0.78,
        "min_margin": 0.20,
        "probability_representation": representation,
        "temperature": (
            FROZEN_TEMPERATURE
            if representation == "temperature_scaled"
            else None
        ),
    }


def assert_response_equal(actual, expected):
    """Compare structure exactly and floating scores with tolerance."""

    assert type(actual) is type(expected)

    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()

        for key in expected:
            assert_response_equal(actual[key], expected[key])

    elif isinstance(expected, list):
        assert len(actual) == len(expected)

        for first, second in zip(actual, expected):
            assert_response_equal(first, second)

    elif isinstance(expected, float):
        assert np.isfinite(actual)
        assert actual == pytest.approx(
            expected,
            rel=0,
            abs=1e-12,
        )

    else:
        assert actual == expected


@pytest.mark.parametrize(
    "probabilities, units",
    [
        ([0.95, 0.05], 2),
        ([0.55, 0.45], 2),
        ([0.50, 0.50], 2),
        ([0.95, 0.05], 1),
        ([0.95, 0.05], 0),
        ([0.05, 0.95], 3),
    ],
)
@pytest.mark.parametrize("include_diagnostics", [False, True])
@pytest.mark.parametrize(
    "representation",
    ["original", "temperature_scaled"],
)
def test_response_matches_reference(
    probabilities,
    units,
    include_diagnostics,
    representation,
):
    policy = make_policy(representation)
    before_policy = copy.deepcopy(policy)
    before_probabilities = probabilities.copy()

    actual = assess_candidate_scores(
        policy["classes"],
        probabilities,
        units,
        policy,
        include_diagnostics=include_diagnostics,
    )
    expected = reference_assess(
        policy["classes"],
        probabilities,
        units,
        policy,
        include_diagnostics=include_diagnostics,
    )

    assert_response_equal(actual, expected)
    assert policy == before_policy
    assert probabilities == before_probabilities

    if actual["status"] == "insufficient_information":
        assert actual["predicted_condition"] is None
        assert actual["model_score"] is None
        assert actual["uncertain"] is True

    if not include_diagnostics:
        assert "diagnostics" not in actual


def test_diagnostics_do_not_change_public_response():
    policy = make_policy()
    probabilities = [0.55, 0.45]

    public = assess_candidate_scores(
        policy["classes"],
        probabilities,
        2,
        policy,
    )
    diagnostic_result = assess_candidate_scores(
        policy["classes"],
        probabilities,
        2,
        policy,
        include_diagnostics=True,
    )

    diagnostics = diagnostic_result.pop("diagnostics")

    assert_response_equal(diagnostic_result, public)
    assert diagnostics["research_only"] is True
    assert len(diagnostics["ranked_conditions"]) == 2
    assert public["predicted_condition"] is None
    assert public["model_score"] is None


def test_score_boundary_controls_public_fields_without_rounding():
    policy = make_policy("original")
    threshold = policy["min_score"]

    for score, accepted in (
        (np.nextafter(threshold, 0.0), False),
        (threshold, True),
        (np.nextafter(threshold, 1.0), True),
    ):
        result = assess_candidate_scores(
            policy["classes"],
            [score, 1.0 - score],
            2,
            policy,
        )

        assert result["uncertain"] is not accepted

        if accepted:
            assert result["status"] == "prediction_available"
            assert result["predicted_condition"] == "A"
            assert result["model_score"] == score
            assert result["uncertainty_reasons"] == []
        else:
            assert result["status"] == "insufficient_information"
            assert result["predicted_condition"] is None
            assert result["model_score"] is None
            assert result["uncertainty_reasons"] == [
                "below_score_threshold"
            ]


def test_withhold_all_never_exposes_public_leader():
    policy = make_policy()
    policy.update({
        "mode": "withhold_all",
        "min_score": None,
        "min_margin": None,
    })

    actual = assess_candidate_scores(
        policy["classes"],
        [0.999, 0.001],
        5,
        policy,
        include_diagnostics=True,
    )
    expected = reference_assess(
        policy["classes"],
        [0.999, 0.001],
        5,
        policy,
        include_diagnostics=True,
    )

    assert_response_equal(actual, expected)
    assert actual["predicted_condition"] is None
    assert actual["model_score"] is None
    assert actual["uncertainty_reasons"] == [
        "no_qualifying_research_policy"
    ]


def test_21_class_ranking_matches_reference():
    policy = make_policy()
    policy["classes"] = [f"condition_{index}" for index in range(21)]

    probabilities = np.full(21, 0.05 / 20)
    probabilities[7] = 0.95

    actual = assess_candidate_scores(
        policy["classes"],
        probabilities,
        3,
        policy,
        include_diagnostics=True,
    )
    expected = reference_assess(
        policy["classes"],
        probabilities,
        3,
        policy,
        include_diagnostics=True,
    )

    assert_response_equal(actual, expected)
    assert actual["predicted_condition"] == "condition_7"

    ranked = actual["diagnostics"]["ranked_conditions"]

    assert len(ranked) == 21
    assert [item["rank"] for item in ranked] == list(range(1, 22))
    assert len({item["condition"] for item in ranked}) == 21


@pytest.mark.parametrize(
    "field, value",
    [
        ("classes", ["B", "A"]),
        ("mode", "unknown"),
        ("min_reporting_units", 1),
        ("probability_representation", "unknown"),
        ("temperature", True),
        ("temperature", 0.1),
        ("temperature", 5.0),
        ("temperature", np.nan),
    ],
)
def test_incompatible_assessment_settings_remain_rejected(field, value):
    policy = make_policy()
    policy[field] = value

    for assessor in (assess_candidate_scores, reference_assess):
        with pytest.raises(ValueError):
            assessor(
                ["A", "B"],
                [0.9, 0.1],
                2,
                policy,
            )


def test_non_boolean_diagnostics_flag_is_rejected():
    policy = make_policy()

    for assessor in (assess_candidate_scores, reference_assess):
        with pytest.raises(ValueError):
            assessor(
                policy["classes"],
                [0.9, 0.1],
                2,
                policy,
                include_diagnostics="false",
            )