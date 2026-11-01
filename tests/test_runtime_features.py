"""Feature and reporting-unit parity with the frozen implementation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from respiratory_candidate_runtime.errors import CandidateInputError
from respiratory_candidate_runtime.features import (
    get_reporting_units,
    normalize_context_codes,
    validate_candidate_features,
)
from inference.candidate_predictor import (
    normalize_context_codes as reference_normalize,
    validate_candidate_features as reference_validate,
)
from scripts.candidate_feature_builder import (
    get_reporting_units as reference_units,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def schema():
    return json.loads(
        (ROOT / "docs/candidate_feature_schema_v1.json").read_text(
            encoding="utf-8"
        )
    )


def make_features(schema, present=(), absent=()):
    values = {
        name: 0.0
        for name in schema["feature_names"]
    }
    values["AGE"] = 22
    values["SEX_M"] = 0

    for code in present:
        values[f"{code}__present"] = 1

    for code in absent:
        values[f"{code}__absent"] = 1

    return values


@pytest.mark.parametrize(
    "present, expected_units",
    [
        ([], 0),
        (["E_89"], 1),
        (["E_53"], 1),
        (["E_53", "head_region_pain"], 1),
        (["E_53", "chest_region_pain", "E_89"], 2),
        (
            ["E_53", "head_region_pain", "chest_region_pain"],
            2,
        ),
    ],
)
def test_reporting_counts_match_reference(
    schema,
    present,
    expected_units,
):
    features = make_features(schema, present)
    before = copy.deepcopy(features)

    actual = validate_candidate_features(features, schema)
    expected = reference_validate(features, schema)

    assert actual == expected
    assert actual[1] == expected_units
    assert actual[2] == []
    assert list(actual[0]) == schema["feature_names"]
    assert features == before


def test_pain_regions_remain_distinct(schema):
    head = make_features(
        schema,
        ["E_53", "E_89", "head_region_pain"],
    )
    chest = make_features(
        schema,
        ["E_53", "E_89", "chest_region_pain"],
    )

    head_row, head_units, _ = validate_candidate_features(head, schema)
    chest_row, chest_units, _ = validate_candidate_features(chest, schema)

    assert head_row != chest_row
    assert head_units == chest_units == 2
    assert head_row["chest_region_pain__present"] == 0
    assert chest_row["head_region_pain__present"] == 0

    for row in (head_row, chest_row):
        assert all(
            row[f"{code}__absent"] == 0
            for code in schema["symptom_features"]
        )


def test_explicit_negatives_are_preserved_separately(schema):
    features = make_features(
        schema,
        present=["E_89"],
        absent=["E_201", "E_91"],
    )

    actual = validate_candidate_features(features, schema)

    assert actual == reference_validate(features, schema)
    assert actual[1] == 1
    assert actual[2] == ["E_201", "E_91"]
    assert actual[0]["E_201__absent"] == 1


@pytest.mark.parametrize(
    "age",
    [0, 120, 22.5, np.float64(22)],
)
def test_reference_accepted_numeric_ages_remain_accepted(schema, age):
    features = make_features(schema)
    features["AGE"] = age

    assert validate_candidate_features(
        features, schema
    ) == reference_validate(features, schema)


@pytest.mark.parametrize(
    "field, value",
    [
        ("AGE", True),
        ("AGE", np.bool_(False)),
        ("AGE", "22"),
        ("AGE", -1),
        ("AGE", 121),
        ("AGE", np.nan),
        ("AGE", np.inf),
        ("SEX_M", 2),
        ("SEX_M", None),
        ("E_89__present", 0.5),
        ("E_89__absent", -1),
        ("E_89__present", np.nan),
    ],
)
def test_invalid_values_match_reference_rejection(schema, field, value):
    features = make_features(schema)
    features[field] = value

    with pytest.raises(ValueError):
        reference_validate(features, schema)

    with pytest.raises(CandidateInputError):
        validate_candidate_features(features, schema)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_feature",
        "extra_feature",
        "not_dictionary",
        "contradictory",
        "region_without_parent",
    ],
)
def test_invalid_feature_structures_remain_rejected(schema, problem):
    features = make_features(schema)

    if problem == "missing_feature":
        del features["AGE"]
    elif problem == "extra_feature":
        features["invented_feature"] = 1
    elif problem == "not_dictionary":
        features = list(features.values())
    elif problem == "contradictory":
        features["E_89__present"] = 1
        features["E_89__absent"] = 1
    else:
        features["head_region_pain__present"] = 1

    with pytest.raises(ValueError):
        reference_validate(features, schema)

    with pytest.raises(CandidateInputError):
        validate_candidate_features(features, schema)


def test_reporting_unit_order_and_reference_parity(schema):
    state = {
        "present": ["head_region_pain", "E_89", "E_53"],
        "absent": [],
        "unknown": [
            code
            for code in schema["symptom_features"]
            if code not in {"head_region_pain", "E_89", "E_53"}
        ],
    }

    assert get_reporting_units(state, schema) == reference_units(
        state, schema
    )
    assert get_reporting_units(state, schema) == [
        "E_89",
        "head_region_pain",
    ]


@pytest.mark.parametrize(
    "problem",
    ["overlap", "incomplete", "duplicate", "negative"],
)
def test_invalid_reporting_states_match_reference(schema, problem):
    state = {
        "present": [],
        "absent": [],
        "unknown": list(schema["symptom_features"]),
    }

    if problem == "overlap":
        state["present"] = ["E_89"]
    elif problem == "incomplete":
        state["unknown"].remove("E_89")
    elif problem == "duplicate":
        state["unknown"].append("E_89")
    else:
        state["unknown"].remove("E_89")
        state["absent"] = ["E_89"]

    with pytest.raises(ValueError):
        reference_units(state, schema)

    with pytest.raises(CandidateInputError):
        get_reporting_units(state, schema)


def test_context_normalization_matches_reference():
    values = [" sneezing ", "blue_lips", "sneezing", "Blue_Lips"]
    before = values.copy()

    actual = normalize_context_codes(values, "unsupported_symptoms")

    assert actual == reference_normalize(
        values, "unsupported_symptoms"
    )
    assert actual == ["Blue_Lips", "blue_lips", "sneezing"]
    assert values == before


@pytest.mark.parametrize(
    "values",
    [
        "sneezing",
        None,
        [" "],
        [1],
        {"sneezing": True},
    ],
)
def test_invalid_context_matches_reference_rejection(values):
    with pytest.raises(ValueError):
        reference_normalize(values, "unsupported_symptoms")

    with pytest.raises(CandidateInputError):
        normalize_context_codes(values, "unsupported_symptoms")