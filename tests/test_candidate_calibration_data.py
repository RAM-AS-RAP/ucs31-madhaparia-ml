"""Tests for isolated loading of reserved full observations."""

import hashlib

import pandas as pd
import pytest

from scripts.candidate_calibration_data import (
    CALIBRATION_PARTITIONS,
    load_calibration_full_rows,
)
from scripts.candidate_feature_builder import load_candidate_schema


def signature(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def example():
    schema = load_candidate_schema()
    partitions = []
    observations = []

    # Include a model-fitting group in the source CSV. It must never
    # appear in any of the three reserved-partition results.
    roles = (*CALIBRATION_PARTITIONS, "fit")

    for number, role in enumerate(roles):
        group_id = signature(role)
        label = "A" if number % 2 == 0 else "B"

        if role != "fit":
            partitions.append({
                "source_group_id": group_id,
                "source_split": "train",
                "canonical_label": label,
                "parent_partition": "calibration_reserved",
                "partition": role,
            })

        for variant in ("full", "partial"):
            row = {
                "source_group_id": group_id,
                "variant_id": variant,
                "canonical_label": label,
                "source_group_weight": 0.5,
                "reported_unit_count": 2 if variant == "full" else 1,
                "AGE": 22 + number,
                "SEX_M": 0,
            }

            present = {"E_89", "E_91"} if variant == "full" else {"E_89"}

            for concept in schema["symptom_features"]:
                row[f"{concept}__present"] = int(concept in present)
                row[f"{concept}__absent"] = 0

            observations.append(row)

    return schema, pd.DataFrame(partitions), pd.DataFrame(observations)


def load_example(tmp_path, example, role="calibration_fit"):
    schema, partitions, observations = example
    path = tmp_path / "train.csv"
    observations.to_csv(path, index=False)

    return load_calibration_full_rows(
        path,
        partitions,
        schema,
        partition=role,
        chunksize=1,
    )


@pytest.mark.parametrize("role", CALIBRATION_PARTITIONS)
def test_loads_only_requested_partition_full_rows(tmp_path, example, role):
    result = load_example(tmp_path, example, role)

    assert result["source_group_id"].tolist() == [signature(role)]
    assert result["variant_id"].tolist() == ["full"]
    assert result["calibration_partition"].tolist() == [role]
    assert result["reported_unit_count"].tolist() == [2]

    assert result["E_89__present"].eq(1).all()
    assert result["E_91__present"].eq(1).all()

    # Original training weights must not leak into calibration weighting.
    assert "source_group_weight" not in result.columns


def test_loader_is_independent_of_csv_row_order(tmp_path, example):
    first = load_example(tmp_path, example)

    schema, partitions, observations = example
    reversed_example = (
        schema,
        partitions.iloc[::-1].copy(),
        observations.iloc[::-1].copy(),
    )
    second = load_example(tmp_path, reversed_example)

    pd.testing.assert_frame_equal(first, second)


def test_missing_full_observation_is_rejected(tmp_path, example):
    schema, partitions, observations = example
    observations = observations.loc[
        ~(
            observations["source_group_id"].eq(signature("calibration_fit"))
            & observations["variant_id"].eq("full")
        )
    ]

    with pytest.raises(ValueError, match="No full observations"):
        load_example(tmp_path, (schema, partitions, observations))


def test_duplicate_full_observation_is_rejected(tmp_path, example):
    schema, partitions, observations = example
    observations = pd.concat(
        [observations, observations.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate full observations"):
        load_example(tmp_path, (schema, partitions, observations))


def test_label_mismatch_is_rejected(tmp_path, example):
    schema, partitions, observations = example
    observations.loc[0, "canonical_label"] = "Wrong label"

    with pytest.raises(ValueError, match="labels disagree"):
        load_example(tmp_path, (schema, partitions, observations))


@pytest.mark.parametrize(
    "column,value",
    [
        ("source_split", "validation"),
        ("parent_partition", "fit"),
        ("partition", "fit"),
    ],
)
def test_invalid_partition_provenance_is_rejected(
    tmp_path, example, column, value
):
    schema, partitions, observations = example
    partitions.loc[0, column] = value

    with pytest.raises(ValueError):
        load_example(tmp_path, (schema, partitions, observations))


def test_duplicate_partition_id_is_rejected(tmp_path, example):
    schema, partitions, observations = example
    partitions.loc[1, "source_group_id"] = partitions.loc[0, "source_group_id"]

    with pytest.raises(ValueError, match="Duplicate source-group IDs"):
        load_example(tmp_path, (schema, partitions, observations))


def test_negative_evidence_is_rejected(tmp_path, example):
    schema, partitions, observations = example
    observations.loc[0, "E_201__absent"] = 1

    with pytest.raises(ValueError, match="positive-only"):
        load_example(tmp_path, (schema, partitions, observations))


def test_incorrect_reporting_unit_count_is_rejected(tmp_path, example):
    schema, partitions, observations = example
    observations.loc[0, "reported_unit_count"] = 3

    with pytest.raises(ValueError, match="reporting-unit count"):
        load_example(tmp_path, (schema, partitions, observations))


def test_model_fitting_partition_cannot_be_requested(tmp_path, example):
    with pytest.raises(ValueError, match="Unknown requested"):
        load_example(tmp_path, example, role="fit")