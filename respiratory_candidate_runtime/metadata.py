"""Validate candidate metadata before model deserialization."""

import math

from .errors import BundleValidationError
from .policy import MIN_REPORTING_UNITS
from .probability import TEMPERATURE_BOUNDS


def _require(condition, message):
    if not condition:
        raise BundleValidationError(message)


def _unique_strings(value, count, description):
    _require(
        isinstance(value, list),
        f"{description} must be a list.",
    )
    _require(
        len(value) == count
        and all(
            isinstance(item, str)
            and bool(item.strip())
            for item in value
        ),
        f"Invalid {description}.",
    )
    _require(
        len(set(value)) == count,
        f"{description} must contain unique values.",
    )


def _number(value, lower, upper, description):
    _require(
        type(value) in (int, float),
        f"{description} must be numeric.",
    )
    _require(
        math.isfinite(value) and lower <= value <= upper,
        f"{description} is outside its supported range.",
    )


def validate_schema(schema):
    """Check the schema structure used by the portable predictor."""

    _require(isinstance(schema, dict), "Schema must be an object.")
    _require(
        schema.get("schema_id") == "respiratory_candidate_v1",
        "Unsupported schema identity.",
    )

    base = schema.get("base_symptom_features")
    concepts = schema.get("symptom_features")
    names = schema.get("feature_names")
    derived = schema.get("derived_features")

    _unique_strings(base, 32, "base symptom concepts")
    _unique_strings(concepts, 34, "symptom concepts")
    _unique_strings(names, 70, "feature names")

    _require(
        type(schema.get("feature_count")) is int
        and schema["feature_count"] == 70,
        "Schema must specify 70 numerical features.",
    )
    _require(
        type(schema.get("symptom_concept_count")) is int
        and schema["symptom_concept_count"] == 34,
        "Schema must specify 34 symptom concepts.",
    )
    _require(
        isinstance(derived, dict)
        and set(derived) == {"head_region_pain", "chest_region_pain"},
        "Unexpected derived symptom concepts.",
    )
    _require(
        not (set(base) & set(derived))
        and set(concepts) == set(base) | set(derived),
        "Base and derived concepts do not match the schema.",
    )

    expected_names = (
        ["AGE", "SEX_M"]
        + [f"{code}__present" for code in concepts]
        + [f"{code}__absent" for code in concepts]
    )

    _require(
        names == expected_names,
        "Candidate feature order is invalid.",
    )
    _require(
        schema.get("negative_evidence_training_supported") is False,
        "Unexpected negative-evidence capability.",
    )

    seen_locations = set()

    for concept, definition in derived.items():
        _require(
            isinstance(definition, dict),
            f"Invalid definition for {concept}.",
        )
        _require(
            definition.get("parent_present_feature") == "E_53"
            and "E_53" in base,
            f"Invalid pain parent for {concept}.",
        )

        locations = definition.get("positive_source_features")

        _require(
            isinstance(locations, list) and len(locations) > 0,
            f"Missing source locations for {concept}.",
        )
        _unique_strings(
            locations,
            len(locations),
            f"source locations for {concept}",
        )
        _require(
            not seen_locations.intersection(locations),
            "Pain-region source locations overlap.",
        )
        seen_locations.update(locations)

    safety_codes = schema.get("safety_only_codes")
    _unique_strings(safety_codes, 3, "safety-only codes")
    _require(
        set(safety_codes) == {"blue_lips", "confusion", "fainting"},
        "Unexpected safety-only codes.",
    )


def validate_policy(
    policy,
    schema,
    classes,
    *,
    model_sha256,
    schema_sha256,
):
    """Check policy identity, dimensions and artifact references.

    Exact release identity is established by the trusted manifest and
    export provenance. This function does not select or tune a policy.
    """

    _require(isinstance(policy, dict), "Policy must be an object.")
    _unique_strings(classes, 21, "class labels")

    _require(
        policy.get("candidate_name") == "rf_group_weighted",
        "Unexpected selected candidate.",
    )
    _require(
        policy.get("policy_id") == "candidate_v1_research_uncertainty_v1",
        "Unexpected policy identity.",
    )
    _require(
        policy.get("classes") == classes,
        "Policy class order does not match classes.json.",
    )
    _require(
        policy.get("feature_names") == schema["feature_names"],
        "Policy feature order does not match the schema.",
    )
    _require(
        policy.get("model_sha256") == model_sha256,
        "Policy refers to a different model.",
    )
    _require(
        policy.get("candidate_schema_sha256") == schema_sha256,
        "Policy refers to a different schema.",
    )
    _require(
        type(policy.get("min_reporting_units")) is int
        and policy["min_reporting_units"] == MIN_REPORTING_UNITS,
        "Unsupported information gate.",
    )
    _require(
        policy.get("mode") == "thresholds",
        "This release requires the selected threshold policy.",
    )
    _require(
        policy.get("probability_representation") == "temperature_scaled",
        "This release requires temperature-scaled probabilities.",
    )
    _require(
        policy.get("deployment_approved") is False,
        "Unexpected deployment approval in research policy.",
    )

    _number(policy.get("min_score"), 0, 1, "Minimum score")
    _number(policy.get("min_margin"), 0, 1, "Minimum margin")
    _number(
        policy.get("temperature"),
        *TEMPERATURE_BOUNDS,
        "Temperature",
    )