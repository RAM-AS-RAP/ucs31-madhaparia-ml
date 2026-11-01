"""Internal prediction flow for already-verified runtime resources."""

import numpy as np
import pandas as pd

from .assessment import assess_candidate_scores
from .errors import CandidateInferenceError, CandidateInputError
from .features import (
    normalize_context_codes,
    validate_candidate_features,
)
from .policy import MIN_REPORTING_UNITS


def _predict_with_resources(
    feature_values,
    *,
    model,
    schema,
    policy,
    safety_flags=(),
    unsupported_symptoms=(),
    include_diagnostics=False,
):
    """Assess one observation using resources verified by the loader.

    This internal function does not verify or load artifact files.
    It does not perform NLP or a complete safety assessment.
    """

    if type(include_diagnostics) is not bool:
        raise CandidateInputError(
            "include_diagnostics must be a boolean"
        )

    flags = normalize_context_codes(safety_flags, "safety_flags")
    unsupported = normalize_context_codes(
        unsupported_symptoms,
        "unsupported_symptoms",
    )

    row, unit_count, absent = validate_candidate_features(
        feature_values,
        schema,
    )

    flags = sorted(
        set(flags)
        | (set(unsupported) & set(schema["safety_only_codes"]))
    )

    context = {
        "safety_flags": flags,
        "safety_review_required": bool(flags),
        "unsupported_symptoms": unsupported,
        "unsupported_absent_concepts": absent,
    }

    reasons = []

    if flags:
        reasons.append("safety_flags_present")
    if unsupported:
        reasons.append("unsupported_symptoms_present")
    if absent:
        reasons.append("explicit_negative_evidence_not_evaluated")
    if unit_count < MIN_REPORTING_UNITS:
        reasons.append("too_few_supported_reporting_units")

    if reasons:
        if flags:
            status = "safety_review_required"
            message = (
                "Safety flags require review before a model conclusion."
            )
        elif unsupported or absent:
            status = "unsupported_input"
            message = (
                "The supplied evidence is outside this candidate's "
                "evaluated inputs."
            )
        else:
            status = "insufficient_information"
            message = (
                "There is insufficient information to return "
                "a leading condition."
            )

        return {
            "model_version": "candidate_v1_research",
            "policy_id": policy["policy_id"],
            "status": status,
            "predicted_condition": None,
            "model_score": None,
            "uncertain": True,
            "uncertainty_reasons": reasons,
            "reported_unit_count": unit_count,
            "probability_representation": (
                policy["probability_representation"]
            ),
            "message": message,
            "prediction_performed": False,
            **context,
        }

    # Caller input has already been validated. Failures here are
    # inference failures, not ordinary insufficient-information results.
    try:
        frame = pd.DataFrame(
            [row],
            columns=schema["feature_names"],
        )
        probabilities = model.predict_proba(frame)

        if np.asarray(probabilities).shape != (
            1,
            len(model.classes_),
        ):
            raise ValueError(
                "Unexpected model probability output shape"
            )

        result = assess_candidate_scores(
            list(model.classes_),
            probabilities[0],
            unit_count,
            policy,
            include_diagnostics=include_diagnostics,
        )
    except Exception as error:
        raise CandidateInferenceError(
            "Candidate inference failed."
        ) from error

    result.update(context)
    result["prediction_performed"] = True

    return result