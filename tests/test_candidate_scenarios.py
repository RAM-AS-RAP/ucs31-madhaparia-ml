"""Tests for controlled full/partial training variants."""

from copy import deepcopy

import pytest

from scripts.candidate_feature_builder import (
    derive_candidate_state,
    get_reporting_units,
    load_candidate_schema,
    make_training_variants,
)


@pytest.fixture(scope="module")
def schema():
    return load_candidate_schema()


def test_one_reporting_unit_produces_only_full_variant(schema):
    # Regional pain activates two features but is one reporting unit.
    full = derive_candidate_state(
        ["E_53", "E_55_@_V_89"], schema
    )

    variants = make_training_variants(
        full, schema, source_group_id="group-one"
    )

    assert len(variants) == 1
    assert variants[0]["variant_id"] == "full"
    assert variants[0]["source_group_weight"] == 1.0
    assert set(variants[0]["present"]) == {
        "E_53", "head_region_pain"
    }


def test_two_units_produce_one_full_and_one_strict_partial(schema):
    full = derive_candidate_state(["E_91", "E_201"], schema)

    variants = make_training_variants(
        full, schema, source_group_id="group-two"
    )

    assert [v["variant_id"] for v in variants] == ["full", "partial"]
    assert variants[0]["reported_unit_count"] == 2
    assert variants[1]["reported_unit_count"] == 1
    assert set(variants[1]["present"]) < set(variants[0]["present"])
    assert sum(v["source_group_weight"] for v in variants) == 1.0
    assert all(v["source_group_id"] == "group-two" for v in variants)


def test_partial_counts_and_parent_rules_hold_across_groups(schema):
    full = derive_candidate_state(
        [
            "E_53", "E_55_@_V_89",
            "E_89", "E_91", "E_201", "E_94", "E_97", "E_144",
        ],
        schema,
    )
    unit_count = len(get_reporting_units(full, schema))

    for number in range(20):
        variants = make_training_variants(
            full, schema, source_group_id=f"group-{number}"
        )
        complete, partial = variants

        assert set(complete["present"]) == set(full["present"])
        assert 1 <= partial["reported_unit_count"] <= min(
            5, unit_count - 1
        )
        assert set(partial["present"]) < set(full["present"])
        assert partial["absent"] == []

        hidden = set(full["present"]) - set(partial["present"])
        assert hidden.issubset(partial["unknown"])

        if "head_region_pain" in partial["present"]:
            assert "E_53" in partial["present"]

        assert sum(v["source_group_weight"] for v in variants) == 1.0


def test_generation_is_reproducible_without_mutating_source(schema):
    full = derive_candidate_state(
        ["E_53", "E_55_@_V_101", "E_89", "E_91"],
        schema,
    )
    original = deepcopy(full)

    first = make_training_variants(
        full, schema, source_group_id="stable-group", seed=42
    )
    second = make_training_variants(
        full, schema, source_group_id="stable-group", seed=42
    )

    assert first == second
    assert full == original


def test_empty_positive_source_is_rejected(schema):
    full = derive_candidate_state([], schema)

    with pytest.raises(ValueError, match="without positive units"):
        make_training_variants(
            full, schema, source_group_id="empty-group"
        )