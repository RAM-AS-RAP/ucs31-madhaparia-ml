"""Apply uncertainty thresholds without fitting or selecting them."""

import numpy as np

from .probability import validate_probabilities


MIN_REPORTING_UNITS = 2


def assess_probability_rows(
    probabilities,
    reported_unit_counts,
    *,
    min_score,
    min_margin,
):
    """Return internal decisions for validated probability rows.

    Thresholds must come from the verified policy in normal inference.
    Leading-class indices are internal diagnostics, not public results.
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
        raise ValueError(
            "Reporting-unit counts must be non-negative integers"
        )

    thresholds = {}

    for name, value in (
        ("min_score", min_score),
        ("min_margin", min_margin),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isscalar(value)
        ):
            raise ValueError(
                "Thresholds must be numbers within [0, 1]"
            )

        value = float(value)

        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(
                "Thresholds must be numbers within [0, 1]"
            )

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