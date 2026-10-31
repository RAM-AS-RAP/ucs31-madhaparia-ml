"""Test final-view preparation using small artificial source records."""

import csv
from dataclasses import replace

import pandas as pd
import pytest

from scripts.audit_candidate_final_source import define_test_cohorts, scan_source
from scripts.candidate_evaluation import make_validation_view
from scripts.candidate_feature_builder import load_candidate_schema
from scripts.prepare_candidate_final_views import build_final_full_frames


MAPPING = {
    "Allergic sinusitis": "Allergic rhinitis",
    "Bronchitis": "Bronchitis",
}


def write_source(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["AGE", "SEX", "PATHOLOGY", "EVIDENCES"])
        for age, sex, label, evidence in rows:
            writer.writerow([age, sex, label, repr(evidence)])


@pytest.fixture
def prepared(tmp_path):
    schema = load_candidate_schema()
    cough = (20, "F", "Bronchitis", ["E_201"])
    fever = (21, "M", "Bronchitis", ["E_91"])

    paths = {
        name: tmp_path / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    write_source(paths["train"], [cough])
    write_source(paths["validation"], [fever])
    write_source(paths["test"], [
        cough,
        cough,
        fever,
        (22.5, "F", "Allergic sinusitis", ["E_89", "E_53", "E_55_@_V_89"]),
        (23, "F", "Bronchitis", ["E_89", "E_53", "E_55_@_V_101"]),
        (24, "F", "Bronchitis", []),
        (25, "F", "Outside scope", ["E_201"]),
    ])

    groups = {
        name: scan_source(path, schema, MAPPING)[0]
        for name, path in paths.items()
    }
    cohorts, _ = define_test_cohorts(
        groups["train"], groups["validation"], groups["test"]
    )

    return paths["test"], schema, groups["test"], cohorts


def build(prepared):
    path, schema, groups, cohorts = prepared
    return build_final_full_frames(path, schema, MAPPING, groups, cohorts)


def test_selection_provenance_and_zero_supported(prepared):
    full, zero = build(prepared)
    assert len(full) == 4
    assert len(zero) == 1
    assert full["source_group_id"].is_unique
    assert set(full["source_row"]) == {1, 3, 4, 5}
    assert full["evaluation_cohort"].value_counts().to_dict() == {
        "overlap": 2, "primary": 2
    }
    assert full.loc[full["source_row"] == 1, "source_group_size"].item() == 2
    assert zero["source_row"].item() == 6
    assert zero["evaluation_cohort"].item() == "zero_supported"
    assert zero["reported_unit_count"].item() == 0
    assert not set(full["source_group_id"]) & set(zero["source_group_id"])


def test_encoding_keeps_regions_and_fractional_age(prepared):
    full, zero = build(prepared)
    head = full.loc[full["source_row"] == 4].iloc[0]
    chest = full.loc[full["source_row"] == 5].iloc[0]

    assert head["AGE"] == 22.5
    assert head["canonical_label"] == "Allergic rhinitis"
    assert head["source_label"] == "Allergic sinusitis"
    assert head["head_region_pain__present"] == 1
    assert head["chest_region_pain__present"] == 0
    assert chest["head_region_pain__present"] == 0
    assert chest["chest_region_pain__present"] == 1
    assert head["E_53__present"] == chest["E_53__present"] == 1
    assert head["reported_unit_count"] == chest["reported_unit_count"] == 2

    schema = prepared[1]
    absent = [f"{name}__absent" for name in schema["symptom_features"]]
    present = [f"{name}__present" for name in schema["symptom_features"]]
    assert (full[absent].to_numpy() == 0).all()
    assert (zero[absent + present].to_numpy() == 0).all()


def test_sparse_views_preserve_metadata_and_are_reproducible(prepared):
    full, _ = build(prepared)
    schema = prepared[1]

    first = make_validation_view(full, schema, "up_to_1", seed=42)
    second = make_validation_view(full, schema, "up_to_1", seed=42)
    pd.testing.assert_frame_equal(first, second)

    for name in (
        "source_group_id", "canonical_label", "evaluation_cohort",
        "source_row", "source_group_size", "available_reporting_units",
    ):
        assert first[name].equals(full[name])

    assert first["reported_unit_count"].eq(1).all()

    for region in ("head_region_pain", "chest_region_pain"):
        selected = first[f"{region}__present"].eq(1)
        assert first.loc[selected, "E_53__present"].eq(1).all()


def test_incomplete_cohort_is_rejected(prepared):
    path, schema, groups, cohorts = prepared
    changed = {name: set(values) for name, values in cohorts.items()}
    changed["complete_eligible"].pop()

    with pytest.raises(ValueError, match="Complete cohort"):
        build_final_full_frames(path, schema, MAPPING, groups, changed)


def test_intersecting_cohorts_are_rejected(prepared):
    path, schema, groups, cohorts = prepared
    changed = {name: set(values) for name, values in cohorts.items()}
    changed["primary"].add(next(iter(changed["overlap"])))

    with pytest.raises(ValueError, match="intersect"):
        build_final_full_frames(path, schema, MAPPING, groups, changed)


def test_incorrect_source_group_size_is_rejected(prepared):
    path, schema, groups, cohorts = prepared
    changed = groups.copy()
    key = next(iter(changed))
    changed[key] = replace(changed[key], source_rows=99)

    with pytest.raises(ValueError, match="Source-group size"):
        build_final_full_frames(path, schema, MAPPING, changed, cohorts)


def test_missing_representative_is_rejected(prepared):
    path, schema, groups, cohorts = prepared
    changed = groups.copy()
    key = next(iter(changed))
    changed[key] = replace(changed[key], first_source_row=999)

    with pytest.raises(ValueError, match="Missing audited"):
        build_final_full_frames(path, schema, MAPPING, changed, cohorts)