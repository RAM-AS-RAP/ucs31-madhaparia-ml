"""Artificial-data tests for final-evaluation grouping and cohorts."""

import csv

import pytest

from scripts.audit_candidate_final_source import (
    define_test_cohorts,
    scan_source,
    summarize_cohort,
)
from scripts.candidate_feature_builder import load_candidate_schema


MAPPING = {
    "Allergic sinusitis": "Allergic rhinitis",
    "Bronchitis": "Bronchitis",
}


@pytest.fixture(scope="module")
def schema():
    return load_candidate_schema()


def scan(tmp_path, schema, name, rows):
    path = tmp_path / f"{name}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["AGE", "SEX", "PATHOLOGY", "EVIDENCES"])
        for age, sex, label, evidence in rows:
            writer.writerow([age, sex, label, repr(evidence)])
    return scan_source(path, schema, MAPPING)


def test_normalization_and_first_representative(tmp_path, schema):
    groups, summary = scan(tmp_path, schema, "source", [
        (22, "F", "Bronchitis", ["E_201", "E_91"]),
        (22.0, "F", "Bronchitis", ["E_91", "E_201", "E_201"]),
    ])
    assert len(groups) == 1
    group = next(iter(groups.values()))
    assert group.first_source_row == 1
    assert group.source_rows == 2
    assert summary["counts"]["extra_in_scope_rows_beyond_one_per_group"] == 1


def test_fractional_age_is_preserved(tmp_path, schema):
    groups, _ = scan(tmp_path, schema, "source", [
        (22, "F", "Bronchitis", ["E_201"]),
        (22.5, "F", "Bronchitis", ["E_201"]),
    ])
    assert len(groups) == 2


def test_scope_zero_support_and_label_mapping(tmp_path, schema):
    groups, summary = scan(tmp_path, schema, "source", [
        (22, "F", "Allergic sinusitis", ["E_201"]),
        (23, "F", "Bronchitis", []),
        (24, "F", "Outside scope", ["E_201"]),
    ])
    assert summary["counts"]["all_source_rows"] == 3
    assert summary["counts"]["outside_scope_rows"] == 1
    assert summary["counts"]["unique_zero_supported_groups"] == 1
    assert summary["counts"]["unique_eligible_groups"] == 1
    assert {group.canonical_label for group in groups.values()} == {
        "Allergic rhinitis", "Bronchitis"
    }


@pytest.mark.parametrize("age,sex", [
    (-1, "F"), (121, "F"), ("nan", "F"), ("invalid", "F"), (22, "X")
])
def test_invalid_demographics_stop_audit(tmp_path, schema, age, sex):
    with pytest.raises(ValueError, match="Invalid demographics"):
        scan(tmp_path, schema, "source", [
            (age, sex, "Bronchitis", ["E_201"]),
        ])


@pytest.mark.parametrize("evidence", [
    "not a collection",
    [123],
    ["E_55_@_V_89"],
])
def test_invalid_evidence_stops_audit(tmp_path, schema, evidence):
    with pytest.raises(ValueError, match="Invalid evidence"):
        scan(tmp_path, schema, "source", [
            (22, "F", "Bronchitis", evidence),
        ])


def test_conflicting_source_labels_stop_audit(tmp_path, schema):
    with pytest.raises(ValueError, match="Conflicting labels within"):
        scan(tmp_path, schema, "source", [
            (22, "F", "Bronchitis", ["E_201"]),
            (22, "F", "Allergic sinusitis", ["E_201"]),
        ])


def test_candidate_collisions_are_retained(tmp_path, schema):
    groups, _ = scan(tmp_path, schema, "source", [
        (22, "F", "Bronchitis", ["E_201"]),
        (22, "F", "Allergic sinusitis", ["E_201", "UNMODELLED_TEST_CODE"]),
    ])
    assert len(groups) == 2

    report = summarize_cohort(groups, set(groups), list(MAPPING.values()))
    collisions = report["candidate_inputs_after_source_deduplication"]
    assert collisions["records"] == 2
    assert collisions["unique_input_groups"] == 1
    assert collisions["groups_with_multiple_condition_labels"] == 1


def test_cohorts_use_original_source_matches(tmp_path, schema):
    def row(age):
        return (age, "F", "Bronchitis", ["E_201"])

    train, _ = scan(tmp_path, schema, "train", [row(20), row(22)])
    validation, _ = scan(tmp_path, schema, "validation", [row(21), row(22)])
    test, _ = scan(tmp_path, schema, "test", [
        row(20), row(21), row(22), row(23),
        (24, "F", "Bronchitis", []),
        (20, "F", "Bronchitis", ["E_201", "UNMODELLED_TEST_CODE"]),
    ])

    cohorts, overlap = define_test_cohorts(train, validation, test)
    assert len(cohorts["complete_eligible"]) == 5
    assert len(cohorts["primary"]) == 2
    assert len(cohorts["overlap"]) == 3
    assert overlap == {
        "training_matches": 2,
        "validation_matches": 2,
        "matches_both": 1,
        "training_only_matches": 1,
        "validation_only_matches": 1,
    }


def test_cross_split_label_conflict_stops_audit(tmp_path, schema):
    train, _ = scan(tmp_path, schema, "train", [
        (22, "F", "Bronchitis", ["E_201"]),
    ])
    test, _ = scan(tmp_path, schema, "test", [
        (22, "F", "Allergic sinusitis", ["E_201"]),
    ])
    with pytest.raises(ValueError, match="Conflicting labels across"):
        define_test_cohorts(train, {}, test)


def test_region_and_parent_count_as_one_unit(tmp_path, schema):
    groups, _ = scan(tmp_path, schema, "source", [
        (22, "F", "Bronchitis", ["E_53", "E_55_@_V_89"]),
    ])
    assert next(iter(groups.values())).reporting_units == 1