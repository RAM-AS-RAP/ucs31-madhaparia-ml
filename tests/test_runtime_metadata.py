"""Metadata validation using committed schema and policy records."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from respiratory_candidate_runtime.errors import BundleValidationError
from respiratory_candidate_runtime.metadata import (
    validate_policy,
    validate_schema,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def metadata():
    schema_bytes = (
        ROOT / "docs/candidate_feature_schema_v1.json"
    ).read_bytes()

    policy_bytes = (
        ROOT
        / "reports/candidate_calibration/recorded_results"
        / "uncertainty_policy/policy.json"
    ).read_bytes()

    schema = json.loads(schema_bytes)
    policy = json.loads(policy_bytes)
    classes = list(policy["classes"])

    return {
        "schema": schema,
        "policy": policy,
        "classes": classes,
        "model_sha256": policy["model_sha256"],
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
    }


def check_policy(data):
    validate_policy(
        data["policy"],
        data["schema"],
        data["classes"],
        model_sha256=data["model_sha256"],
        schema_sha256=data["schema_sha256"],
    )


def test_recorded_metadata_is_accepted_without_changes(metadata):
    before = copy.deepcopy(metadata)

    validate_schema(metadata["schema"])
    check_policy(metadata)

    assert metadata == before


@pytest.mark.parametrize(
    "problem",
    [
        "identity",
        "feature_count",
        "concept_count",
        "duplicate_base",
        "feature_order",
        "missing_region",
        "wrong_parent",
        "overlapping_locations",
        "negative_evidence",
        "missing_safety_code",
    ],
)
def test_invalid_schema_is_rejected(metadata, problem):
    schema = metadata["schema"]

    if problem == "identity":
        schema["schema_id"] = "unknown"
    elif problem == "feature_count":
        schema["feature_count"] = 66
    elif problem == "concept_count":
        schema["symptom_concept_count"] = 32
    elif problem == "duplicate_base":
        schema["base_symptom_features"][1] = (
            schema["base_symptom_features"][0]
        )
    elif problem == "feature_order":
        names = schema["feature_names"]
        names[0], names[1] = names[1], names[0]
    elif problem == "missing_region":
        del schema["derived_features"]["head_region_pain"]
    elif problem == "wrong_parent":
        schema["derived_features"]["head_region_pain"][
            "parent_present_feature"
        ] = "E_89"
    elif problem == "overlapping_locations":
        derived = schema["derived_features"]
        derived["chest_region_pain"]["positive_source_features"] = (
            derived["head_region_pain"]["positive_source_features"].copy()
        )
    elif problem == "negative_evidence":
        schema["negative_evidence_training_supported"] = True
    else:
        schema["safety_only_codes"].remove("blue_lips")

    with pytest.raises(BundleValidationError):
        validate_schema(schema)


@pytest.mark.parametrize(
    "field, value",
    [
        ("candidate_name", "different_model"),
        ("policy_id", "different_policy"),
        ("model_sha256", "0" * 64),
        ("candidate_schema_sha256", "0" * 64),
        ("min_reporting_units", 1),
        ("min_reporting_units", True),
        ("mode", "withhold_all"),
        ("probability_representation", "original"),
        ("deployment_approved", True),
        ("min_score", -0.1),
        ("min_score", True),
        ("min_margin", 1.1),
        ("temperature", 0.1),
        ("temperature", 5.0),
        ("temperature", float("nan")),
        ("temperature", "0.640839"),
    ],
)
def test_incompatible_policy_is_rejected(metadata, field, value):
    metadata["policy"][field] = value

    with pytest.raises(BundleValidationError):
        check_policy(metadata)


def test_class_order_must_match_policy(metadata):
    metadata["classes"].reverse()

    with pytest.raises(BundleValidationError, match="class order"):
        check_policy(metadata)


def test_duplicate_classes_are_rejected(metadata):
    metadata["classes"][1] = metadata["classes"][0]

    with pytest.raises(BundleValidationError):
        check_policy(metadata)


def test_policy_feature_order_must_match_schema(metadata):
    metadata["policy"]["feature_names"] = list(
        reversed(metadata["policy"]["feature_names"])
    )

    with pytest.raises(BundleValidationError, match="feature order"):
        check_policy(metadata)