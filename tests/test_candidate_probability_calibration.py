"""Numerical and behavior tests for temperature scaling."""

import numpy as np
import pytest

from scripts.candidate_probability_calibration import (
    fit_temperature,
    negative_log_likelihood,
    temperature_scale,
)


def test_identity_temperature_preserves_nonzero_probabilities():
    probabilities = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.3, 0.6],
    ])
    original = probabilities.copy()

    result = temperature_scale(probabilities, 1.0)

    np.testing.assert_allclose(result, probabilities, atol=1e-12)
    np.testing.assert_array_equal(probabilities, original)


def test_temperature_changes_sharpness_but_preserves_leading_class():
    probabilities = np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.2, 0.7],
    ])

    sharper = temperature_scale(probabilities, 0.5)
    flatter = temperature_scale(probabilities, 2.0)

    assert (sharper.max(axis=1) > probabilities.max(axis=1)).all()
    assert (flatter.max(axis=1) < probabilities.max(axis=1)).all()

    for result in (sharper, flatter):
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            result.argmax(axis=1),
            probabilities.argmax(axis=1),
        )


def test_zero_probabilities_and_ties_remain_valid():
    probabilities = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
    ])

    result = temperature_scale(probabilities, 4.0)

    assert np.isfinite(result).all()
    assert (result > 0).all()
    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    np.testing.assert_array_equal(result.argmax(axis=1), [0, 0])


def test_weighted_log_loss_matches_hand_calculation():
    probabilities = [[0.8, 0.2], [0.3, 0.7]]

    actual = negative_log_likelihood(
        probabilities,
        [0, 1],
        sample_weight=[2.0, 1.0],
    )
    expected = -(2 * np.log(0.8) + np.log(0.7)) / 3

    assert actual == pytest.approx(expected)


def test_fitted_temperature_matches_known_interior_solution():
    # Every prediction is 90% for class 0, but class 0 is correct 80%
    # of the time. The optimum rescales the prediction to [0.8, 0.2].
    probabilities = np.tile([0.9, 0.1], (10, 1))
    targets = np.array([0] * 8 + [1] * 2)

    fitted = fit_temperature(probabilities, targets)
    expected_temperature = np.log(9.0) / np.log(4.0)

    assert fitted["temperature"] == pytest.approx(
        expected_temperature, rel=1e-5
    )
    assert fitted["boundary_solution"] is None
    assert (
        fitted["weighted_log_loss_after"]
        < fitted["weighted_log_loss_before"]
    )


def test_weights_match_equivalent_repeated_observations():
    weighted = fit_temperature(
        [[0.9, 0.1], [0.9, 0.1]],
        [0, 1],
        sample_weight=[8.0, 2.0],
    )
    repeated = fit_temperature(
        np.tile([0.9, 0.1], (10, 1)),
        [0] * 8 + [1] * 2,
    )

    assert weighted["temperature"] == pytest.approx(
        repeated["temperature"], rel=1e-5
    )
    assert weighted["total_sample_weight"] == 10.0


def test_lower_boundary_solution_is_recorded():
    fitted = fit_temperature(
        [[0.6, 0.4], [0.4, 0.6]],
        [0, 1],
    )

    assert fitted["temperature"] == 0.25
    assert fitted["boundary_solution"] == "lower"


def test_upper_boundary_solution_is_recorded():
    fitted = fit_temperature(
        np.tile([0.99, 0.01], (4, 1)),
        [0, 0, 1, 1],
    )

    assert fitted["temperature"] == 4.0
    assert fitted["boundary_solution"] == "upper"


@pytest.mark.parametrize(
    "probabilities",
    [
        [[0.8, 0.3]],
        [[-0.1, 1.1]],
        [[np.nan, 0.5]],
        [[0.5, np.inf]],
        [0.5, 0.5],
    ],
)
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError):
        temperature_scale(probabilities, 1.0)


@pytest.mark.parametrize("temperature", [0, -1, np.nan, True])
def test_invalid_temperatures_are_rejected(temperature):
    with pytest.raises(ValueError, match="finite positive"):
        temperature_scale([[0.6, 0.4]], temperature)


@pytest.mark.parametrize(
    "weights",
    [[1, -1], [0, 0], [1, np.nan], [1]],
)
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="Weights"):
        fit_temperature(
            [[0.8, 0.2], [0.3, 0.7]],
            [0, 1],
            sample_weight=weights,
        )


@pytest.mark.parametrize("targets", [[0], [0, 2], [0.0, 1.0]])
def test_invalid_target_indices_are_rejected(targets):
    with pytest.raises(ValueError):
        fit_temperature(
            [[0.8, 0.2], [0.3, 0.7]],
            targets,
        )