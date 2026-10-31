"""Separate research-candidate prediction interface."""

import json
import math
from functools import lru_cache
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from inference.candidate_assessment import assess_candidate_scores
from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_feature_builder import get_reporting_units
from scripts.candidate_uncertainty_policy import MIN_REPORTING_UNITS


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "data/candidate_experiment_v1"
CALIBRATION = EXPERIMENT / "calibration"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_candidate_resources():
    """Verify and load the frozen research resources once per process."""

    paths = {
        "model": EXPERIMENT / "models/rf_group_weighted/model.joblib",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "policy": CALIBRATION / "uncertainty_policy/policy.json",
        "search_manifest": (
            CALIBRATION / "uncertainty_policy/search_manifest.json"
        ),
        "view_manifest": CALIBRATION / "views/view_manifest.json",
        "evaluation_manifest": (
            CALIBRATION / "uncertainty_evaluation/evaluation_manifest.json"
        ),
        "evaluation_decision": (
            CALIBRATION / "uncertainty_evaluation/evaluation_decision.json"
        ),
        "calibration_helper": (
            ROOT / "scripts/candidate_probability_calibration.py"
        ),
        "uncertainty_helper": ROOT / "scripts/candidate_uncertainty_policy.py",
        "feature_builder": ROOT / "scripts/candidate_feature_builder.py",
    }

    before = {name: file_hash(path) for name, path in paths.items()}

    schema = read_json(paths["candidate_schema"])
    policy = read_json(paths["policy"])
    search = read_json(paths["search_manifest"])
    views = read_json(paths["view_manifest"])
    evaluation = read_json(paths["evaluation_manifest"])
    decision = read_json(paths["evaluation_decision"])

    if before["policy"] != search["output_sha256"]["policy.json"]:
        raise ValueError("Frozen policy fingerprint mismatch")

    for name in (
        "model",
        "candidate_schema",
        "policy",
        "search_manifest",
        "view_manifest",
        "calibration_helper",
        "uncertainty_helper",
    ):
        if before[name] != evaluation["input_sha256"][name]:
            raise ValueError(f"Evaluated dependency changed: {name}")

    if (
        before["evaluation_decision"]
        != evaluation["output_sha256"]["evaluation_decision.json"]
    ):
        raise ValueError("Evaluation decision fingerprint mismatch")

    if decision["policy_sha256"] != before["policy"]:
        raise ValueError("Evaluation decision refers to another policy")

    if (
        before["feature_builder"]
        != views["input_sha256"]["feature_builder"]
    ):
        raise ValueError("Reporting-unit implementation changed")

    if (
        policy["mode"] == "thresholds"
        and not decision["research_targets_met_on_all_required_views"]
    ):
        raise ValueError("Policy did not meet the evaluation research targets")

    if (
        policy["model_sha256"] != before["model"]
        or policy["candidate_schema_sha256"] != before["candidate_schema"]
    ):
        raise ValueError("Policy model or schema reference mismatch")

    if policy["feature_names"] != schema["feature_names"]:
        raise ValueError("Policy feature order differs from schema")

    if (
        len(schema["feature_names"]) != 70
        or len(set(schema["feature_names"])) != 70
        or len(policy["classes"]) != 21
        or len(set(policy["classes"])) != 21
    ):
        raise ValueError("Unexpected candidate dimensions")

    if policy["min_reporting_units"] != MIN_REPORTING_UNITS:
        raise ValueError("Information gate differs from the evaluated policy")

    model = joblib.load(paths["model"])

    if list(model.classes_) != policy["classes"]:
        raise ValueError("Model class order differs from policy")

    if model.n_features_in_ != len(schema["feature_names"]):
        raise ValueError("Model feature count differs from schema")

    if (
        hasattr(model, "feature_names_in_")
        and list(model.feature_names_in_) != schema["feature_names"]
    ):
        raise ValueError("Model feature order differs from schema")

    after = {name: file_hash(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("Candidate resources changed while loading")

    return model, schema, policy


def normalize_context_codes(values, name):
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError(f"{name} must be a collection of strings")

    if any(
        not isinstance(value, str) or not value.strip()
        for value in values
    ):
        raise ValueError(f"{name} contains an invalid value")

    return sorted({value.strip() for value in values})


def validate_candidate_features(feature_values, schema):
    if not isinstance(feature_values, dict):
        raise ValueError("Features must be supplied as a dictionary")

    names = schema["feature_names"]
    if set(feature_values) != set(names):
        raise ValueError("Feature mismatch: supply exactly the candidate features")

    row = {}

    for name in names:
        value = feature_values[name]

        if name == "AGE":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError("AGE must be a numeric value between 0 and 120")

        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"Invalid feature value: {name}") from None

        if not math.isfinite(value):
            raise ValueError(f"Non-finite feature value: {name}")

        if name == "AGE":
            if not 0 <= value <= 120:
                raise ValueError("AGE must be between 0 and 120")
        elif value not in (0.0, 1.0):
            raise ValueError(f"{name} must be encoded as 0 or 1")

        row[name] = value

    concepts = schema["symptom_features"]
    present = {
        code for code in concepts if row[f"{code}__present"] == 1
    }
    absent = {
        code for code in concepts if row[f"{code}__absent"] == 1
    }

    if present & absent:
        raise ValueError("Conflicting present and absent symptom states")

    # Negative evidence is recorded separately and will block prediction.
    # The reporting-unit count uses positive evidence only.
    state = {
        "present": [code for code in concepts if code in present],
        "absent": [],
        "unknown": [code for code in concepts if code not in present],
    }
    units = get_reporting_units(state, schema)

    return row, len(units), sorted(absent)


def predict_candidate(
    feature_values,
    *,
    safety_flags=(),
    unsupported_symptoms=(),
    include_diagnostics=False,
):
    """Predict from confirmed source-defined candidate features.

    This interface does not perform NLP or a complete safety assessment.
    Callers must retain their existing safety checks and pass their flags.
    """

    if type(include_diagnostics) is not bool:
        raise ValueError("include_diagnostics must be a boolean")

    flags = normalize_context_codes(safety_flags, "safety_flags")
    unsupported = normalize_context_codes(
        unsupported_symptoms, "unsupported_symptoms"
    )

    model, schema, policy = load_candidate_resources()
    row, unit_count, absent = validate_candidate_features(
        feature_values, schema
    )

    # Recognized safety-only symptoms must not disappear merely because
    # they have no numerical model feature.
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
            message = "Safety flags require review before a model conclusion."
        elif unsupported or absent:
            status = "unsupported_input"
            message = "The supplied evidence is outside this candidate's evaluated inputs."
        else:
            status = "insufficient_information"
            message = "There is insufficient information to return a leading condition."

        return {
            "model_version": "candidate_v1_research",
            "policy_id": policy["policy_id"],
            "status": status,
            "predicted_condition": None,
            "model_score": None,
            "uncertain": True,
            "uncertainty_reasons": reasons,
            "reported_unit_count": unit_count,
            "probability_representation": policy["probability_representation"],
            "message": message,
            "prediction_performed": False,
            **context,
        }

    frame = pd.DataFrame([row], columns=schema["feature_names"])
    probabilities = model.predict_proba(frame)

    if np.asarray(probabilities).shape != (1, len(model.classes_)):
        raise ValueError("Unexpected model probability output shape")

    result = assess_candidate_scores(
        list(model.classes_),
        probabilities[0],
        unit_count,
        policy,
        include_diagnostics=include_diagnostics,
    )
    result.update(context)
    result["prediction_performed"] = True
    return result