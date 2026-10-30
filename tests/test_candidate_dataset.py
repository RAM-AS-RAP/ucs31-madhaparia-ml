"""Integration tests using small artificial records in temporary folders."""

import csv
from collections import Counter, defaultdict

import pytest

from scripts.audit_candidate_overlap import input_signature
from scripts.audit_candidate_source_records import file_hash
from scripts.build_candidate_dataset import build_candidate_dataset
from scripts.candidate_feature_builder import load_candidate_schema


ALLERGY = "Allergic sinusitis"
BRONCHITIS = "Bronchitis"

MAPPING = {
    ALLERGY: "Allergic rhinitis",
    BRONCHITIS: BRONCHITIS,
}

SHARED = (30, "F", ALLERGY, ["E_181", "E_169"])
RETAINED_ALLERGY = (
    31, "F", ALLERGY,
    ["E_53", "E_55_@_V_89", "E_181"],
)
RETAINED_BRONCHITIS = (40, "M", BRONCHITIS, ["E_201"])
VALIDATION_BRONCHITIS = (
    45, "M", BRONCHITIS, ["E_201", "E_91"]
)


def write_source(path, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["AGE", "SEX", "PATHOLOGY", "EVIDENCES"])
        for age, sex, label, evidence in rows:
            writer.writerow([age, sex, label, repr(evidence)])


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


@pytest.fixture(scope="module")
def schema():
    return load_candidate_schema()


@pytest.fixture
def sources(tmp_path):
    train_rows = [
        SHARED,
        SHARED,
        RETAINED_ALLERGY,
        RETAINED_ALLERGY,
        RETAINED_BRONCHITIS,
        (50, "F", ALLERGY, ["E_10"]),
        (60, "M", "Outside scope", ["E_201"]),
    ]
    validation_rows = [
        SHARED,
        SHARED,
        VALIDATION_BRONCHITIS,
    ]

    train_path = tmp_path / "train_source.csv"
    validation_path = tmp_path / "validation_source.csv"

    write_source(train_path, train_rows)
    write_source(validation_path, validation_rows)

    return {
        "train_path": train_path,
        "validation_path": validation_path,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
    }


def build(sources, schema, output_dir):
    return build_candidate_dataset(
        train_path=sources["train_path"],
        validation_path=sources["validation_path"],
        output_dir=output_dir,
        schema=schema,
        label_mapping=MAPPING,
        seed=42,
    )


def test_overlap_removal_and_exclusions(sources, schema, tmp_path):
    output = tmp_path / "candidate"
    manifest = build(sources, schema, output)

    assert manifest["shared_source_groups_removed_from_training"] == 1
    assert manifest["retained_train_validation_source_overlap"] == 0

    train_totals = manifest["splits"]["train"]["totals"]
    validation_totals = manifest["splits"]["validation"]["totals"]

    assert train_totals["retained_source_groups"] == 2
    assert train_totals["scenario_rows"] == 3
    assert validation_totals["retained_source_groups"] == 2
    assert validation_totals["scenario_rows"] == 2

    exclusions = Counter(
        (row["split"], row["reason"])
        for row in read_csv(output / "excluded_rows.csv")
    )

    assert exclusions == {
        ("train", "source_group_reserved_for_validation"): 2,
        ("train", "repeated_source_input"): 1,
        ("train", "no_supported_positive_concepts"): 1,
        ("validation", "repeated_source_input"): 1,
    }

    train_ids = {
        row["source_group_id"] for row in read_csv(output / "train.csv")
    }
    validation_ids = {
        row["source_group_id"]
        for row in read_csv(output / "validation.csv")
    }
    assert train_ids.isdisjoint(validation_ids)


def test_mapping_provenance_weights_and_feature_encoding(
    sources, schema, tmp_path
):
    output = tmp_path / "candidate"
    build(sources, schema, output)

    groups = read_csv(output / "source_groups.csv")
    train = read_csv(output / "train.csv")
    validation = read_csv(output / "validation.csv")

    allergy_id = input_signature(
        31.0, "F", RETAINED_ALLERGY[3]
    ).hex()

    allergy_group = next(
        row for row in groups if row["source_group_id"] == allergy_id
    )

    assert allergy_group["split"] == "train"
    assert allergy_group["source_row"] == "3"
    assert allergy_group["source_group_size"] == "2"
    assert allergy_group["source_label"] == ALLERGY
    assert allergy_group["canonical_label"] == "Allergic rhinitis"
    assert allergy_group["available_reporting_units"] == "2"

    weights = defaultdict(float)
    for row in train:
        weights[row["source_group_id"]] += float(
            row["source_group_weight"]
        )

    assert len(weights) == 2
    assert all(weight == 1.0 for weight in weights.values())

    full_allergy = next(
        row for row in train
        if row["source_group_id"] == allergy_id
        and row["variant_id"] == "full"
    )

    assert full_allergy["canonical_label"] == "Allergic rhinitis"
    assert full_allergy["head_region_pain__present"] == "1"
    assert full_allergy["E_53__present"] == "1"
    assert full_allergy["chest_region_pain__present"] == "0"
    assert full_allergy["chest_region_pain__absent"] == "0"
    assert float(full_allergy["AGE"]) == 31
    assert full_allergy["SEX_M"] == "0"

    expected_columns = [
        "source_group_id",
        "variant_id",
        "canonical_label",
        "source_group_weight",
        "reported_unit_count",
        *schema["feature_names"],
    ]

    assert list(train[0]) == expected_columns

    for row in train + validation:
        for name in schema["feature_names"]:
            if name.endswith("__absent"):
                assert row[name] == "0"

    assert all(row["variant_id"] == "full" for row in validation)
    assert all(
        float(row["source_group_weight"]) == 1.0
        for row in validation
    )


def test_build_is_reproducible_and_preserves_sources(
    sources, schema, tmp_path
):
    paths = [sources["train_path"], sources["validation_path"]]
    before = [file_hash(path) for path in paths]

    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = build(sources, schema, first)
    second_manifest = build(sources, schema, second)

    assert first_manifest == second_manifest

    for name in (
        "train.csv",
        "validation.csv",
        "source_groups.csv",
        "excluded_rows.csv",
        "manifest.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    assert [file_hash(path) for path in paths] == before

    for name, expected in first_manifest["output_sha256"].items():
        assert file_hash(first / name) == expected


def test_existing_output_directory_is_preserved(
    sources, schema, tmp_path
):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build(sources, schema, output)

    assert marker.read_text(encoding="utf-8") == "preserve me"
    assert list(output.iterdir()) == [marker]


def test_conflicting_labels_within_source_group_stop_build(
    sources, schema, tmp_path
):
    conflict = (
        RETAINED_ALLERGY[0],
        RETAINED_ALLERGY[1],
        BRONCHITIS,
        RETAINED_ALLERGY[3],
    )
    write_source(
        sources["train_path"],
        sources["train_rows"] + [conflict],
    )

    output = tmp_path / "conflict"

    with pytest.raises(ValueError, match="Conflicting labels within"):
        build(sources, schema, output)

    assert not output.exists()


def test_conflicting_labels_across_splits_stop_build(
    sources, schema, tmp_path
):
    conflict = (SHARED[0], SHARED[1], BRONCHITIS, SHARED[3])
    write_source(
        sources["validation_path"],
        [conflict, VALIDATION_BRONCHITIS],
    )

    output = tmp_path / "conflict"

    with pytest.raises(ValueError, match="Conflicting labels across"):
        build(sources, schema, output)

    assert not output.exists()


def test_losing_a_condition_does_not_produce_success_manifest(
    sources, schema, tmp_path
):
    # All allergy training groups would be reserved for validation.
    write_source(
        sources["train_path"],
        [SHARED, RETAINED_BRONCHITIS],
    )

    output = tmp_path / "incomplete"

    with pytest.raises(ValueError, match="lost coverage"):
        build(sources, schema, output)

    assert not (output / "manifest.json").exists()


def test_same_file_cannot_be_used_for_both_splits(
    sources, schema, tmp_path
):
    with pytest.raises(ValueError, match="different files"):
        build_candidate_dataset(
            train_path=sources["train_path"],
            validation_path=sources["train_path"],
            output_dir=tmp_path / "invalid",
            schema=schema,
            label_mapping=MAPPING,
        )