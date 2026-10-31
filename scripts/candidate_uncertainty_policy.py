"""Decision checks and statistics for the research abstention policy."""

from statistics import NormalDist

import numpy as np

from scripts.candidate_probability_calibration import validate_probabilities


MIN_REPORTING_UNITS = 2
WILSON_Z = NormalDist().inv_cdf(0.975)


def wilson_interval(successes, total):
    """Return the endpoints of a two-sided 95% Wilson interval."""

    for value in (successes, total):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError("Counts must be integers")

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Invalid success and total counts")

    if total == 0:
        return None, None

    proportion = successes / total
    z_squared = WILSON_Z ** 2
    denominator = 1 + z_squared / total

    centre = (
        proportion + z_squared / (2 * total)
    ) / denominator

    half_width = (
        WILSON_Z
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total ** 2)
        )
        / denominator
    )

    return (
        float(max(0.0, centre - half_width)),
        float(min(1.0, centre + half_width)),
    )


def assess_probability_rows(
    probabilities,
    reported_unit_counts,
    *,
    min_score,
    min_margin,
):
    """Apply the fixed information gate and supplied score thresholds.

    Returned top indices are internal diagnostics.
    A caller must not expose them as leading conditions for withheld rows.
    """

    values = validate_probabilities(probabilities)
    units = np.asarray(reported_unit_counts)

    if (
        units.shape != (len(values),)
        or not np.issubdtype(units.dtype, np.number)
        or not np.isfinite(units).all()
        or (units < 0).any()
        or not np.equal(units, np.floor(units)).all()
    ):
        raise ValueError("Reporting-unit counts must be non-negative integers")

    thresholds = {}
    for name, value in (
        ("min_score", min_score),
        ("min_margin", min_margin),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isscalar(value)
        ):
            raise ValueError("Thresholds must be numbers within [0, 1]")

        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Thresholds must be numbers within [0, 1]")

        thresholds[name] = value

    top_indices = values.argmax(axis=1)
    top_scores = values[np.arange(len(values)), top_indices]
    second_scores = np.partition(values, -2, axis=1)[:, -2]
    margins = top_scores - second_scores

    enough_information = units >= MIN_REPORTING_UNITS
    score_passed = top_scores >= thresholds["min_score"]
    margin_passed = margins >= thresholds["min_margin"]

    return {
        "accepted": enough_information & score_passed & margin_passed,
        "enough_reporting_units": enough_information,
        "score_passed": score_passed,
        "margin_passed": margin_passed,
        "top_indices": top_indices,
        "top_scores": top_scores,
        "score_margins": margins,
    }


def summarize_acceptance(correct, accepted):
    """Describe accepted predictions without treating abstentions as correct."""

    correct = np.asarray(correct)
    accepted = np.asarray(accepted)

    if (
        correct.ndim != 1
        or accepted.shape != correct.shape
        or correct.dtype != np.dtype(bool)
        or accepted.dtype != np.dtype(bool)
    ):
        raise ValueError("Expected aligned one-dimensional boolean arrays")

    total = len(correct)
    accepted_count = int(accepted.sum())
    withheld_count = total - accepted_count
    correct_accepted = int((correct & accepted).sum())
    incorrect_accepted = accepted_count - correct_accepted

    lower, upper = wilson_interval(correct_accepted, accepted_count)

    return {
        "observations": total,
        "accepted": accepted_count,
        "withheld": withheld_count,
        "coverage": accepted_count / total if total else None,
        "abstention_rate": withheld_count / total if total else None,
        "correct_accepted": correct_accepted,
        "incorrect_accepted": incorrect_accepted,
        "accepted_accuracy": (
            correct_accepted / accepted_count if accepted_count else None
        ),
        "accepted_error_rate": (
            incorrect_accepted / accepted_count if accepted_count else None
        ),
        "accepted_accuracy_wilson_lower": lower,
        "accepted_accuracy_wilson_upper": upper,
        "accepted_error_wilson_lower": (
            1 - upper if upper is not None else None
        ),
        "accepted_error_wilson_upper": (
            1 - lower if lower is not None else None
        ),
        "correct_accepted_fraction_of_all": (
            correct_accepted / total if total else None
        ),
    }