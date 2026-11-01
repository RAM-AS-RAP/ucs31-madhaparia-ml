"""Probability validation and frozen temperature scaling.

The numerical behaviour follows the evaluated research implementation.
This module contains no temperature fitting or model training.
"""

import numpy as np
from scipy.special import logsumexp


PROBABILITY_FLOOR = 1e-12
TEMPERATURE_BOUNDS = (0.25, 4.0)


def validate_probabilities(probabilities):
    """Validate a non-empty rows-by-classes probability matrix."""

    values = np.asarray(probabilities, dtype=np.float64)

    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] < 2
    ):
        raise ValueError("Expected a non-empty rows-by-classes matrix")

    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("Probabilities must be finite and within [0, 1]")

    if not np.allclose(
        values.sum(axis=1),
        1.0,
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError("Probability rows must sum to one")

    return values


def temperature_scale(probabilities, temperature):
    """Apply a supplied positive temperature without fitting it.

    The resource loader and assessment layer must separately enforce
    the frozen policy and its permitted temperature bounds.
    """

    values = validate_probabilities(probabilities)

    if (
        isinstance(temperature, (bool, np.bool_))
        or not np.isscalar(temperature)
    ):
        raise ValueError("Temperature must be a finite positive number")

    temperature = float(temperature)

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a finite positive number")

    log_values = np.log(
        np.clip(values, PROBABILITY_FLOOR, 1.0)
    )
    scaled = log_values / temperature
    log_normalizer = logsumexp(
        scaled,
        axis=1,
        keepdims=True,
    )
    calibrated = np.exp(scaled - log_normalizer)

    validate_probabilities(calibrated)

    if not np.array_equal(
        values.argmax(axis=1),
        calibrated.argmax(axis=1),
    ):
        raise ValueError("Temperature scaling changed a leading-class index")

    return calibrated