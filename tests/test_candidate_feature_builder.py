"""Regression tests for deriving candidate concepts before masking."""

import pytest

from scripts.candidate_feature_builder import (
    derive_candidate_state,
    load_candidate_schema,
)


@pytest.fixture(scope="module")
def schema():
    return load_candidate_schema()


@pytest.mark.parametrize(
    "location,expected_region",
    [
        ("E_55_@_V_25", "head_region_pain"),
        ("E_55_@_V_62", "head_region_pain"),
        ("E_55_@_V_89", "head_region_pain"),
        ("E_55_@_V_124", "head_region_pain"),
        ("E_55_@_V_166", "head_region_pain"),
        ("E_55_@_V_167", "head_region_pain"),
        ("E_55_@_V_29", "chest_region_pain"),
        ("E_55_@_V_55", "chest_region_pain"),
        ("E_55_@_V_56", "chest_region_pain"),
        ("E_55_@_V_101", "chest_region_pain"),
        ("E_55_@_V_170", "chest_region_pain"),
        ("E_55_@_V_171", "chest_region_pain"),
    ],
)
def test_source_locations_map_to_expected_region(
    schema, location, expected_region
):
    result = derive_candidate_state(["E_53", location], schema)

    assert set(result["present"]) == {"E_53", expected_region}
    assert result["derived_from"] == {
        expected_region: [location]
    }
    assert result["absent"] == []


def test_tiredness_with_different_pain_regions_stays_distinct(schema):
    head = derive_candidate_state(
        ["E_89", "E_53", "E_55_@_V_89"], schema
    )
    chest = derive_candidate_state(
        ["E_89", "E_53", "E_55_@_V_101"], schema
    )

    assert set(head["present"]) == {
        "E_89", "E_53", "head_region_pain"
    }
    assert set(chest["present"]) == {
        "E_89", "E_53", "chest_region_pain"
    }
    assert "chest_region_pain" in head["unknown"]
    assert "head_region_pain" in chest["unknown"]


def test_both_pain_regions_can_coexist(schema):
    result = derive_candidate_state(
        ["E_53", "E_55_@_V_89", "E_55_@_V_101"],
        schema,
    )

    assert set(result["present"]) == {
        "E_53", "head_region_pain", "chest_region_pain"
    }
    assert result["derived_from"] == {
        "head_region_pain": ["E_55_@_V_89"],
        "chest_region_pain": ["E_55_@_V_101"],
    }


def test_general_pain_does_not_invent_a_location(schema):
    result = derive_candidate_state(["E_53"], schema)

    assert result["present"] == ["E_53"]
    assert "head_region_pain" in result["unknown"]
    assert "chest_region_pain" in result["unknown"]
    assert result["derived_from"] == {}
    assert result["absent"] == []


@pytest.mark.parametrize(
    "location",
    ["E_55_@_V_89", "E_55_@_V_101"],
)
def test_location_without_general_pain_is_rejected(schema, location):
    with pytest.raises(ValueError, match="required parent E_53"):
        derive_candidate_state([location], schema)


def test_empty_evidence_stays_unknown(schema):
    result = derive_candidate_state([], schema)

    assert result["present"] == []
    assert result["absent"] == []
    assert result["unknown"] == schema["symptom_features"]
    assert result["derived_from"] == {}


def test_supported_binary_symptoms_are_preserved(schema):
    result = derive_candidate_state(
        ["E_91", "E_201", "E_89"], schema
    )

    assert set(result["present"]) == {"E_91", "E_201", "E_89"}
    assert result["absent"] == []
    assert set(result["unknown"]) == (
        set(schema["symptom_features"]) - set(result["present"])
    )


def test_unmodelled_evidence_does_not_become_a_symptom(schema):
    # E_10 is outside the candidate symptom feature set.
    assert "E_10" not in schema["symptom_features"]

    result = derive_candidate_state(["E_201", "E_10"], schema)

    assert result["present"] == ["E_201"]
    assert "E_10" not in result["unknown"]
    assert result["absent"] == []


def test_order_and_duplicates_do_not_change_derived_state(schema):
    source = [
        "E_55_@_V_89",
        "E_53",
        "E_89",
        "E_55_@_V_89",
        "E_55_@_V_25",
    ]
    original = source.copy()

    first = derive_candidate_state(source, schema)
    second = derive_candidate_state(
        sorted(set(source)), schema
    )

    assert first == second
    assert source == original
    assert set(first["derived_from"]["head_region_pain"]) == {
        "E_55_@_V_89", "E_55_@_V_25"
    }


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "E_53",
        {"E_53": True},
        [None],
        [123],
        [""],
        ["   "],
    ],
)
def test_malformed_evidence_is_rejected(schema, invalid):
    with pytest.raises(ValueError):
        derive_candidate_state(invalid, schema)