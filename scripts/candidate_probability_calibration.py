"""Temperature scaling for a frozen candidate probability distribution."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


PROBABILITY_FLOOR = 1e-12
TEMPERATURE_BOUNDS = (0.25, 4.0)


def validate_probabilities(probabilities):
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
        values.sum(axis=1), 1.0, rtol=0, atol=1e-8
    ):
        raise ValueError("Probability rows must sum to one")

    return values


def validate_targets_and_weights(values, target_indices, sample_weight):
    targets = np.asarray(target_indices)

    if targets.shape != (len(values),):
        raise ValueError("Targets must contain one class index per row")

    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Target indices must be integers")

    if (targets < 0).any() or (targets >= values.shape[1]).any():
        raise ValueError("Target index outside the class range")

    if sample_weight is None:
        weights = np.ones(len(values), dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)

    if (
        weights.shape != (len(values),)
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Weights must be finite, positive and match the rows")

    total_weight = float(weights.sum())
    if not np.isfinite(total_weight):
        raise ValueError("Total sample weight must be finite")

    return targets, weights / total_weight, total_weight


def negative_log_likelihood(
    probabilities,
    target_indices,
    sample_weight=None,
):
    """Return weighted mean multiclass log loss using the protocol floor."""

    values = validate_probabilities(probabilities)
    targets, normalized_weights, _ = validate_targets_and_weights(
        values, target_indices, sample_weight
    )

    correct_probabilities = values[np.arange(len(values)), targets]
    losses = -np.log(
        np.clip(correct_probabilities, PROBABILITY_FLOOR, 1.0)
    )

    return float(np.dot(normalized_weights, losses))


def temperature_scale(probabilities, temperature):
    """Apply one positive temperature without changing the class columns."""

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
    log_normalizer = logsumexp(scaled, axis=1, keepdims=True)
    calibrated = np.exp(scaled - log_normalizer)

    validate_probabilities(calibrated)

    if not np.array_equal(
        values.argmax(axis=1),
        calibrated.argmax(axis=1),
    ):
        raise ValueError("Temperature scaling changed a leading-class index")

    return calibrated


def fit_temperature(probabilities, target_indices, sample_weight=None):
    """Fit one temperature; callers must supply calibration-fit rows only."""

    values = validate_probabilities(probabilities)
    targets, normalized_weights, total_weight = validate_targets_and_weights(
        values, target_indices, sample_weight
    )

    log_values = np.log(
        np.clip(values, PROBABILITY_FLOOR, 1.0)
    )
    row_indices = np.arange(len(values))
    lower, upper = TEMPERATURE_BOUNDS

    def objective(log_temperature):
        temperature = float(np.exp(log_temperature))
        scaled = log_values / temperature

        losses = (
            logsumexp(scaled, axis=1)
            - scaled[row_indices, targets]
        )
        return float(np.dot(normalized_weights, losses))

    optimizer_options = {"xatol": 1e-8, "maxiter": 500}

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(lower)), float(np.log(upper))),
        method="bounded",
        options=optimizer_options,
    )

    if (
        not result.success
        or not np.isfinite(result.x)
        or not np.isfinite(result.fun)
    ):
        raise RuntimeError(f"Temperature optimization failed: {result.message}")

    optimized_temperature = float(
        np.clip(np.exp(result.x), lower, upper)
    )

    # Explicitly check both endpoints and T=1, since bounded optimization
    # searches the interior and a constrained optimum may be at a boundary.
    candidates = [
        optimized_temperature,
        lower,
        1.0,
        upper,
    ]
    scored = [
        (objective(float(np.log(value))), value)
        for value in candidates
    ]
    best_loss, temperature = min(scored)

    identity_loss = objective(0.0)
    if identity_loss <= best_loss + 1e-12:
        temperature = 1.0

    calibrated = temperature_scale(values, temperature)
    before_loss = negative_log_likelihood(
        values, targets, sample_weight
    )
    after_loss = negative_log_likelihood(
        calibrated, targets, sample_weight
    )

    if after_loss > before_loss + 1e-10:
        raise RuntimeError("Fitted temperature increased fitting log loss")

    boundary = None
    if temperature == lower:
        boundary = "lower"
    elif temperature == upper:
        boundary = "upper"

    return {
        "method": "scalar_temperature_on_log_probabilities",
        "temperature": float(temperature),
        "temperature_bounds": [lower, upper],
        "probability_floor": PROBABILITY_FLOOR,
        "observations": len(values),
        "classes": values.shape[1],
        "total_sample_weight": total_weight,
        "weighted_log_loss_before": before_loss,
        "weighted_log_loss_after": after_loss,
        "leading_class_indices_unchanged": True,
        "boundary_solution": boundary,
        "optimizer": {
            "method": "bounded",
            "variable": "log_temperature",
            "options": optimizer_options,
            "success": bool(result.success),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
            "interior_temperature": optimized_temperature,
            "endpoints_and_identity_checked": True,
        },
    }