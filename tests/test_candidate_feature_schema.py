"""Validate the proposed candidate schema for issue #45.

These are specification checks, not model-accuracy tests
or tests of a completed candidate inference pipeline.
"""

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "docs/candidate_feature_schema_v1.json"


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_candidate_is_explicitly_untrained(schema):
    assert schema["schema_id"] == "respiratory_candidate_v1"
    assert schema["status"] == "proposed_not_trained"
    assert schema["compatible_with_existing_v2_model"] is False


def test_feature_order_and_uniqueness(schema):
    concepts = schema["symptom_features"]

    assert len(concepts) == len(set(concepts)) == 34

    expected = (
        ["AGE", "SEX_M"]
        + [f"{code}__present" for code in concepts]
        + [f"{code}__absent" for code in concepts]
    )

    assert schema["feature_names"] == expected
    assert schema["feature_count"] == len(expected) == 70
    assert len(set(expected)) == 70


def test_original_v2_concepts_are_preserved(schema):
    original = json.loads(
        (ROOT / "models/v2_feature_schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["base_symptom_features"] == original["symptom_features"]
    assert schema["symptom_features"][:32] == original["symptom_features"]

    assert schema["symptom_features"][32:] == [
        "head_region_pain",
        "chest_region_pain",
    ]


@pytest.mark.parametrize(
    ("canonical_code", "region"),
    [
        ("headache", "head_region_pain"),
        ("chest_pain", "chest_region_pain"),
    ],
)
def test_broad_symptoms_do_not_invent_exact_locations(
    schema, canonical_code, region
):
    mapping = schema["canonical_mapping_overrides"][canonical_code]

    assert mapping["requires_confirmation_of"] == region
    assert mapping["present_features"] == ["E_53", region]
    assert mapping["exact_location_required"] is False

    # A broad patient report must not assert a precise E_55 value.
    assert not any(
        "_@_" in feature
        for feature in mapping["present_features"]
    )

    assert set(mapping["present_features"]).issubset(
        schema["symptom_features"]
    )


def test_headache_and_chest_pain_have_distinct_representations(schema):
    mappings = schema["canonical_mapping_overrides"]

    headache = set(mappings["headache"]["present_features"])
    chest = set(mappings["chest_pain"]["present_features"])

    assert headache != chest
    assert headache & chest == {"E_53"}
    assert headache - chest == {"head_region_pain"}
    assert chest - headache == {"chest_region_pain"}


def test_region_source_groups_are_distinct(schema):
    derived = schema["derived_features"]

    head = set(derived["head_region_pain"]["positive_source_features"])
    chest = set(derived["chest_region_pain"]["positive_source_features"])

    assert head
    assert chest
    assert head.isdisjoint(chest)

    for region in derived.values():
        assert region["source_evidence"] == "E_55"
        assert region["parent_present_feature"] == "E_53"

        values = region["source_value_meanings"]

        assert set(region["positive_source_features"]) == {
            f"E_55_@_{value}"
            for value in values
        }


def test_symptom_states_distinguish_unknown_from_absent(schema):
    states = schema["symptom_state_encoding"]

    assert states["present"] == {"present": 1, "absent": 0}
    assert states["absent"] == {"present": 0, "absent": 1}
    assert states["unknown"] == {"present": 0, "absent": 0}
    assert states["invalid"] == {"present": 1, "absent": 1}

    assert states["unknown"] != states["absent"]


def test_negative_training_support_is_not_claimed(schema):
    assert schema["negative_evidence_training_supported"] is False
    assert schema["negative_evidence_note"]


def test_safety_and_unsupported_codes_are_not_invented_features(schema):
    assert set(schema["safety_only_codes"]) == {
        "blue_lips",
        "confusion",
        "fainting",
    }
    assert schema["nlp_only_codes"] == ["sneezing"]

    non_model_codes = (
        set(schema["safety_only_codes"])
        | set(schema["nlp_only_codes"])
    )

    assert non_model_codes.isdisjoint(schema["symptom_features"])

    assert not any(
        "duration" in name.lower()
        for name in schema["feature_names"]
    )


def test_existing_v2_schema_matches_recorded_fingerprint(schema):
    relative_path = "models/v2_feature_schema.json"
    current_hash = hashlib.sha256(
        (ROOT / relative_path).read_bytes()
    ).hexdigest()

    assert current_hash == schema["source_sha256"][relative_path]


def test_location_groups_match_original_source_when_available(schema):
    # Raw data is deliberately not committed to Git.
    source_path = ROOT / "data/raw/release_evidences.json"

    if not source_path.exists():
        pytest.skip("Original DDXPlus evidence metadata is not available.")

    current_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    assert current_hash == schema["source_sha256"][
        "data/raw/release_evidences.json"
    ]

    evidence = json.loads(
        source_path.read_text(encoding="utf-8-sig")
    )["E_55"]

    assert evidence["data_type"] == "M"
    assert evidence["code_question"] == "E_53"

    for region in schema["derived_features"].values():
        for value, expected_meaning in region[
            "source_value_meanings"
        ].items():
            assert value in evidence["possible-values"]
            assert value != evidence["default_value"]
            assert evidence["value_meaning"][value]["en"] == expected_meaning