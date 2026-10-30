"""Check masking, pain dependencies and unknown-state preservation."""

from copy import deepcopy

import pytest

from scripts.candidate_feature_builder import (
    derive_candidate_state,
    get_reporting_units,
    load_candidate_schema,
    mask_candidate_state,
)


@pytest.fixture(scope="module")
def schema():
    return load_candidate_schema()


def test_regional_pain_counts_as_one_reporting_unit(schema):
    full = derive_candidate_state(
        ["E_53", "E_55_@_V_89"], schema
    )

    assert get_reporting_units(full, schema) == ["head_region_pain"]

    result = mask_candidate_state(
        full, schema, reported_unit_count=1, seed=42
    )

    assert result["reported_units"] == ["head_region_pain"]
    assert result["reported_unit_count"] == 1
    assert set(result["present"]) == {"E_53", "head_region_pain"}


def test_both_regions_share_the_general_pain_parent(schema):
    full = derive_candidate_state(
        ["E_53", "E_55_@_V_89", "E_55_@_V_101"],
        schema,
    )

    result = mask_candidate_state(
        full, schema, reported_unit_count=2, seed=42
    )

    assert set(result["reported_units"]) == {
        "head_region_pain", "chest_region_pain"
    }
    assert set(result["present"]) == {
        "E_53", "head_region_pain", "chest_region_pain"
    }
    assert result["present"].count("E_53") == 1


def test_general_pain_without_region_remains_reportable(schema):
    full = derive_candidate_state(["E_53"], schema)

    result = mask_candidate_state(
        full, schema, reported_unit_count=1, seed=42
    )

    assert result["reported_units"] == ["E_53"]
    assert result["present"] == ["E_53"]
    assert "head_region_pain" in result["unknown"]
    assert "chest_region_pain" in result["unknown"]


def test_masking_preserves_positive_and_unknown_rules(schema):
    full = derive_candidate_state(
        ["E_53", "E_55_@_V_89", "E_89", "E_91", "E_201"],
        schema,
    )

    for seed in range(20):
        result = mask_candidate_state(
            full, schema, reported_unit_count=2, seed=seed
        )
        present = set(result["present"])
        unknown = set(result["unknown"])

        assert len(result["reported_units"]) == 2
        assert present.issubset(full["present"])
        assert result["absent"] == []
        assert present.isdisjoint(unknown)
        assert present | unknown == set(schema["symptom_features"])

        hidden = set(full["present"]) - present
        assert hidden
        assert hidden.issubset(unknown)

        if "head_region_pain" in present:
            assert "E_53" in present
        else:
            assert "E_53" in unknown

        assert set(result["derived_from"]).issubset(present)


def test_masking_is_reproducible_and_does_not_mutate_source(schema):
    full = derive_candidate_state(
        ["E_53", "E_55_@_V_101", "E_89", "E_91"],
        schema,
    )
    original = deepcopy(full)

    first = mask_candidate_state(
        full, schema, reported_unit_count=2, seed="group-1:variant-2"
    )
    second = mask_candidate_state(
        full, schema, reported_unit_count=2, seed="group-1:variant-2"
    )

    assert first == second
    assert full == original


@pytest.mark.parametrize("count", [0, -1, 2, 1.5, True])
def test_invalid_reporting_count_is_rejected(schema, count):
    full = derive_candidate_state(["E_201"], schema)

    with pytest.raises(ValueError, match="reported_unit_count"):
        mask_candidate_state(
            full, schema, reported_unit_count=count, seed=42
        )


def test_empty_source_cannot_generate_a_positive_scenario(schema):
    full = derive_candidate_state([], schema)

    with pytest.raises(ValueError, match="reported_unit_count"):
        mask_candidate_state(
            full, schema, reported_unit_count=1, seed=42
        )


def test_missing_pain_parent_is_rejected(schema):
    full = derive_candidate_state(
        ["E_53", "E_55_@_V_89"], schema
    )
    full["present"].remove("E_53")
    full["unknown"].append("E_53")

    with pytest.raises(ValueError, match="requires present parent"):
        get_reporting_units(full, schema)


def test_overlapping_states_are_rejected(schema):
    full = derive_candidate_state(["E_201"], schema)
    full["unknown"].append("E_201")

    with pytest.raises(ValueError, match="overlap"):
        get_reporting_units(full, schema)


def test_explicit_absence_requires_a_separate_preparation_path(schema):
    full = derive_candidate_state(["E_201"], schema)
    full["unknown"].remove("E_91")
    full["absent"].append("E_91")

    with pytest.raises(ValueError, match="absent source evidence"):
        get_reporting_units(full, schema)