"""Check uncertainty decisions against boundaries and the reference."""

import numpy as np
import pytest

from respiratory_candidate_runtime.policy import assess_probability_rows
from scripts.candidate_uncertainty_policy import (
    assess_probability_rows as reference_assess,
)


def assert_reference_match(
    probabilities,
    units,
    *,
    min_score=0.78,
    min_margin=0.20,
):
    actual = assess_probability_rows(
        probabilities,
        units,
        min_score=min_score,
        min_margin=min_margin,
    )
    expected = reference_assess(
        probabilities,
        units,
        min_score=min_score,
        min_margin=min_margin,
    )

    assert actual.keys() == expected.keys()

    # Decisions, class indices and these directly calculated scores
    # should match exactly for identical probability inputs.
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name])

    return actual


def test_frozen_score_threshold_is_inclusive_without_rounding():
    threshold = 0.78

    scores = np.array([
        np.nextafter(threshold, 0.0),
        threshold,
        np.nextafter(threshold, 1.0),
    ])
    probabilities = np.column_stack([scores, 1.0 - scores])

    result = assert_reference_match(
        probabilities,
        [2, 2, 2],
    )

    np.testing.assert_array_equal(
        result["accepted"],
        [False, True, True],
    )

    # All three round to the same displayed percentage.
    assert len(set(np.round(scores * 100, 2))) == 1


def test_margin_threshold_is_inclusive():
    probabilities = [[0.5, 0.3, 0.2]]
    margin = 0.5 - 0.3

    # A lower score threshold isolates the margin gate in this
    # synthetic unit test. It does not modify the deployed policy.
    at_boundary = assert_reference_match(
        probabilities,
        [2],
        min_score=0.4,
        min_margin=margin,
    )
    above_boundary = assert_reference_match(
        probabilities,
        [2],
        min_score=0.4,
        min_margin=np.nextafter(margin, 1.0),
    )

    assert bool(at_boundary["accepted"][0])
    assert not bool(above_boundary["accepted"][0])
    assert not bool(above_boundary["margin_passed"][0])


def test_information_gate_requires_two_units_even_with_high_score():
    probabilities = np.tile([0.99, 0.01], (4, 1))

    result = assert_reference_match(
        probabilities,
        [0, 1, 2, 3],
    )

    np.testing.assert_array_equal(
        result["accepted"],
        [False, False, True, True],
    )


def test_tied_leaders_fail_positive_margin_requirement():
    result = assert_reference_match(
        [[0.5, 0.5]],
        [2],
        min_score=0.4,
        min_margin=0.20,
    )

    assert result["top_indices"][0] == 0
    assert result["score_margins"][0] == 0
    assert not bool(result["accepted"][0])


def test_varied_21_class_rows_match_reference_without_mutation():
    rng = np.random.default_rng(58)
    probabilities = rng.dirichlet(
        np.full(21, 0.1),
        size=200,
    )
    units = rng.integers(0, 6, size=200)

    before_probabilities = probabilities.copy()
    before_units = units.copy()

    assert_reference_match(probabilities, units)

    np.testing.assert_array_equal(
        probabilities,
        before_probabilities,
    )
    np.testing.assert_array_equal(units, before_units)


@pytest.mark.parametrize(
    "units",
    [
        [],
        [1, 2],
        [-1],
        [1.5],
        [np.nan],
        [np.inf],
        [True],
        ["2"],
    ],
    ids=[
        "empty",
        "wrong-length",
        "negative",
        "fractional",
        "nan",
        "infinity",
        "boolean",
        "string",
    ],
)
def test_invalid_unit_counts_match_reference_rejection(units):
    for assessor in (assess_probability_rows, reference_assess):
        with pytest.raises(ValueError):
            assessor(
                [[0.9, 0.1]],
                units,
                min_score=0.78,
                min_margin=0.20,
            )


@pytest.mark.parametrize("field", ["min_score", "min_margin"])
@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, np.nan, np.inf, True, [0.5]],
)
def test_invalid_thresholds_match_reference_rejection(field, value):
    thresholds = {
        "min_score": 0.78,
        "min_margin": 0.20,
    }
    thresholds[field] = value

    for assessor in (assess_probability_rows, reference_assess):
        with pytest.raises(ValueError):
            assessor(
                [[0.9, 0.1]],
                [2],
                **thresholds,
            )