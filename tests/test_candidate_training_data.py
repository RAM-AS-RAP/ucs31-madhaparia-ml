"""Tests for fitting-only loading and source-group-based weighting."""

import numpy as np
import pandas as pd
import pytest

from scripts.candidate_training_data import (
    load_fitting_rows,
    sample_weights_for_candidate,
)


@pytest.fixture
def fit_groups():
    return pd.DataFrame({
        "source_group_id": ["a", "b", "c"],
        "canonical_label": ["A", "A", "B"],
        "partition": ["fit", "fit", "fit"],
    })


@pytest.fixture
def rows():
    return pd.DataFrame({
        "source_group_id": ["a", "a", "b", "c", "c"],
        "variant_id": ["full", "partial", "full", "full", "partial"],
        "canonical_label": ["A", "A", "A", "B", "B"],
        "source_group_weight": [0.5, 0.5, 1.0, 0.5, 0.5],
    })


def test_group_weighted_candidate_preserves_base_weights(rows, fit_groups):
    weights, factors = sample_weights_for_candidate(
        rows, fit_groups, "rf_group_weighted"
    )

    np.testing.assert_array_equal(weights, [0.5, 0.5, 1.0, 0.5, 0.5])
    assert factors == {"A": 1.0, "B": 1.0}
    assert weights.sum() == 3.0


def test_balancing_uses_groups_not_variant_counts(rows, fit_groups):
    weights, factors = sample_weights_for_candidate(
        rows, fit_groups, "rf_class_balanced"
    )

    # A has two source groups; B has one, regardless of variant counts.
    assert factors == {"A": 0.75, "B": 1.5}
    np.testing.assert_allclose(weights, [0.375, 0.375, 0.75, 0.75, 0.75])

    assert weights[rows["canonical_label"].eq("A")].sum() == 1.5
    assert weights[rows["canonical_label"].eq("B")].sum() == 1.5


def test_reserved_group_is_rejected(rows, fit_groups):
    rows.loc[0, "source_group_id"] = "reserved"

    with pytest.raises(ValueError, match="non-fitting source groups"):
        sample_weights_for_candidate(
            rows, fit_groups, "rf_group_weighted"
        )


def test_reserved_partition_cannot_be_used_as_fit_groups(rows, fit_groups):
    fit_groups.loc[0, "partition"] = "calibration_reserved"

    with pytest.raises(ValueError, match="Only fitting groups"):
        sample_weights_for_candidate(
            rows, fit_groups, "rf_group_weighted"
        )


def test_mismatched_label_is_rejected(rows, fit_groups):
    rows.loc[0, "canonical_label"] = "B"

    with pytest.raises(ValueError, match="disagree with group labels"):
        sample_weights_for_candidate(
            rows, fit_groups, "rf_group_weighted"
        )


def test_duplicate_variant_is_rejected(rows, fit_groups):
    rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate variant"):
        sample_weights_for_candidate(
            rows, fit_groups, "rf_group_weighted"
        )


def test_incorrect_group_weight_is_rejected(rows, fit_groups):
    rows.loc[0, "source_group_weight"] = 1.0

    with pytest.raises(ValueError, match="Invalid source-group weights"):
        sample_weights_for_candidate(
            rows, fit_groups, "rf_group_weighted"
        )


def test_missing_fitting_group_is_rejected(rows, fit_groups):
    rows = rows.loc[rows["source_group_id"].ne("c")]

    with pytest.raises(ValueError, match="no observations"):
        sample_weights_for_candidate(
            rows, fit_groups, "rf_group_weighted"
        )


def test_loader_excludes_reserved_rows_even_across_chunks(
    rows, fit_groups, tmp_path
):
    training = rows.copy()
    training["AGE"] = 22
    training["SEX_M"] = 0

    reserved = pd.DataFrame({
        "source_group_id": ["reserved"],
        "variant_id": ["full"],
        "canonical_label": ["A"],
        "source_group_weight": [1.0],
        "AGE": [99],
        "SEX_M": [1],
    })

    training = pd.concat([training, reserved], ignore_index=True)
    path = tmp_path / "train.csv"
    training.to_csv(path, index=False)

    partitions = pd.concat([
        fit_groups,
        pd.DataFrame({
            "source_group_id": ["reserved"],
            "canonical_label": ["A"],
            "partition": ["calibration_reserved"],
        }),
    ], ignore_index=True)

    loaded = load_fitting_rows(
        path,
        partitions,
        {"feature_names": ["AGE", "SEX_M"]},
        chunksize=2,
    )

    assert len(loaded) == len(rows)
    assert set(loaded["source_group_id"]) == {"a", "b", "c"}
    assert loaded["AGE"].eq(22).all()

    weights, _ = sample_weights_for_candidate(
        loaded, fit_groups, "rf_group_weighted"
    )
    assert weights.sum() == 3.0