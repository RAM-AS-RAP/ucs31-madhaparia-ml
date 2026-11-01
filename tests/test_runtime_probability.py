"""Check portable probability behaviour against the frozen reference."""

import numpy as np
import pytest

from respiratory_candidate_runtime.probability import (
    temperature_scale,
    validate_probabilities,
)
from scripts.candidate_probability_calibration import (
    temperature_scale as reference_temperature_scale,
    validate_probabilities as reference_validate_probabilities,
)


FROZEN_TEMPERATURE = 0.6408387217626598


@pytest.mark.parametrize(
    "temperature",
    [0.25, FROZEN_TEMPERATURE, 1.0, 2.0, 4.0],
)
def test_matches_reference_for_21_class_distributions(temperature):
    rng = np.random.default_rng(58)

    probabilities = rng.dirichlet(
        np.ones(21),
        size=100,
    )

    # Include zeros, equal leaders and a uniform distribution.
    edge_cases = np.zeros((3, 21), dtype=np.float64)
    edge_cases[0, 0] = 1.0
    edge_cases[1, :2] = 0.5
    edge_cases[2, :] = 1.0 / 21

    probabilities = np.vstack([probabilities, edge_cases])
    before = probabilities.copy()

    expected = reference_temperature_scale(
        probabilities,
        temperature,
    )
    actual = temperature_scale(
        probabilities,
        temperature,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        actual.argmax(axis=1),
        expected.argmax(axis=1),
    )
    np.testing.assert_array_equal(
        probabilities,
        before,
    )

    assert np.isfinite(actual).all()
    assert (actual >= 0).all()
    assert (actual <= 1).all()

    np.testing.assert_allclose(
        actual.sum(axis=1),
        1.0,
        rtol=0,
        atol=1e-12,
    )


def test_matches_independent_two_class_calculation():
    # At T=0.5, probabilities are squared and then normalized.
    actual = temperature_scale([[0.8, 0.2]], 0.5)

    expected = np.array([
        [0.64 / 0.68, 0.04 / 0.68],
    ])

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_identity_temperature_preserves_positive_probabilities():
    probabilities = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.3, 0.6],
    ])

    actual = temperature_scale(probabilities, 1.0)

    np.testing.assert_allclose(
        actual,
        probabilities,
        rtol=0,
        atol=1e-12,
    )


def test_zero_scores_and_ties_preserve_reference_leaders():
    probabilities = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
    ])

    actual = temperature_scale(
        probabilities,
        FROZEN_TEMPERATURE,
    )

    # The evaluated probability floor makes zero entries positive.
    assert (actual > 0).all()

    np.testing.assert_array_equal(
        actual.argmax(axis=1),
        [0, 0],
    )


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        [0.5, 0.5],
        np.empty((0, 21)),
        [[1.0]],
        [[[0.5, 0.5]]],
        [[-0.1, 1.1]],
        [[0.2, 0.2]],
        [[np.nan, 0.5]],
        [[np.inf, 0.0]],
    ],
    ids=[
        "empty",
        "one-dimensional",
        "zero-rows",
        "one-class",
        "three-dimensional",
        "outside-range",
        "incorrect-sum",
        "nan",
        "infinity",
    ],
)
def test_invalid_probabilities_remain_invalid(probabilities):
    for validator in (
        validate_probabilities,
        reference_validate_probabilities,
    ):
        with pytest.raises(ValueError):
            validator(probabilities)

    with pytest.raises(ValueError):
        temperature_scale(
            probabilities,
            FROZEN_TEMPERATURE,
        )


@pytest.mark.parametrize(
    "temperature",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        True,
        np.bool_(False),
        [1.0],
        None,
    ],
    ids=[
        "zero",
        "negative",
        "nan",
        "infinity",
        "python-bool",
        "numpy-bool",
        "list",
        "none",
    ],
)
def test_invalid_temperatures_match_reference_rejection(temperature):
    probabilities = [[0.7, 0.2, 0.1]]

    for scaler in (
        temperature_scale,
        reference_temperature_scale,
    ):
        with pytest.raises(ValueError):
            scaler(probabilities, temperature)