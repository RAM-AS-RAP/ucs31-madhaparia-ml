"""Response assessment for the separate research candidate interface."""

import numpy as np

from scripts.candidate_probability_calibration import (
    TEMPERATURE_BOUNDS,
    temperature_scale,
    validate_probabilities,
)
from scripts.candidate_uncertainty_policy import (
    MIN_REPORTING_UNITS,
    assess_probability_rows,
)


def assess_candidate_scores(
    classes,
    original_probabilities,
    reported_unit_count,
    policy,
    *,
    include_diagnostics=False,
):
    """Assess one prediction using the frozen calibration and policy.

    Input validation and safety handling must occur before this function.
    Diagnostic rankings are optional research output, not a user conclusion.
    """

    if type(include_diagnostics) is not bool:
        raise ValueError("include_diagnostics must be a boolean")

    classes = list(classes)
    if (
        len(classes) < 2
        or any(not isinstance(label, str) or not label.strip()
               for label in classes)
        or len(classes) != len(set(classes))
        or classes != list(policy["classes"])
    ):
        raise ValueError("Classes must match the frozen policy order")

    raw = np.asarray(original_probabilities, dtype=float)
    if raw.shape != (len(classes),):
        raise ValueError("Expected one probability per model class")

    validate_probabilities(raw[np.newaxis, :])

    if policy["min_reporting_units"] != MIN_REPORTING_UNITS:
        raise ValueError("Information gate differs from the protocol")

    representation = policy["probability_representation"]

    if representation == "temperature_scaled":
        temperature = policy["temperature"]

        if (
            isinstance(temperature, (bool, np.bool_))
            or not np.isscalar(temperature)
        ):
            raise ValueError("Invalid frozen temperature")

        temperature = float(temperature)
        lower, upper = TEMPERATURE_BOUNDS

        if not np.isfinite(temperature) or not lower <= temperature <= upper:
            raise ValueError("Invalid frozen temperature")

        scores = temperature_scale(
            raw[np.newaxis, :], temperature
        )[0]

    elif representation == "original":
        if policy["temperature"] is not None:
            raise ValueError("Original probabilities must not apply temperature")
        scores = raw.copy()

    else:
        raise ValueError("Unknown probability representation")

    mode = policy["mode"]

    if mode == "thresholds":
        min_score = policy["min_score"]
        min_margin = policy["min_margin"]
    elif mode == "withhold_all":
        if policy["min_score"] is not None or policy["min_margin"] is not None:
            raise ValueError("Withhold-all policy must not contain thresholds")
        min_score, min_margin = 0.0, 0.0
    else:
        raise ValueError("Unknown policy mode")

    decision = assess_probability_rows(
        scores[np.newaxis, :],
        [reported_unit_count],
        min_score=min_score,
        min_margin=min_margin,
    )

    reasons = []

    if not decision["enough_reporting_units"][0]:
        reasons.append("too_few_supported_reporting_units")

    if mode == "withhold_all":
        reasons.append("no_qualifying_research_policy")
    else:
        if not decision["score_passed"][0]:
            reasons.append("below_score_threshold")
        if not decision["margin_passed"][0]:
            reasons.append("close_alternative")

    accepted = mode == "thresholds" and bool(decision["accepted"][0])
    top_index = int(decision["top_indices"][0])

    result = {
        "model_version": "candidate_v1_research",
        "policy_id": policy["policy_id"],
        "status": (
            "prediction_available" if accepted else "insufficient_information"
        ),
        "predicted_condition": classes[top_index] if accepted else None,
        "model_score": float(scores[top_index]) if accepted else None,
        "uncertain": not accepted,
        "uncertainty_reasons": reasons,
        "reported_unit_count": int(reported_unit_count),
        "probability_representation": representation,
        "message": (
            "A leading condition met the research model's acceptance rules."
            if accepted
            else "There is insufficient information to return a leading condition."
        ),
        "score_note": (
            "Model scores were evaluated on simulated respiratory-condition "
            "data. They are not clinically validated individual disease risks."
        ),
    }

    if include_diagnostics:
        order = sorted(
            range(len(classes)),
            key=lambda index: (-float(scores[index]), index),
        )

        result["diagnostics"] = {
            "research_only": True,
            "score_margin": float(decision["score_margins"][0]),
            "ranked_conditions": [
                {
                    "rank": rank,
                    "condition": classes[index],
                    "original_model_score": float(raw[index]),
                    "assessed_model_score": float(scores[index]),
                }
                for rank, index in enumerate(order, start=1)
            ],
        }

    return result