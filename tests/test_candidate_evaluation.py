"""Tests for shared observations and canonical probability alignment."""

import numpy as np
import pandas as pd
import pytest

from scripts.candidate_evaluation import (
    harmonize_probabilities,
    make_validation_view,
    project_v2_features,
)
from scripts.candidate_feature_builder import load_candidate_schema


@pytest.fixture(scope="module")
def schema():
    return load_candidate_schema()


@pytest.fixture
def full_frame(schema):
    examples = [
        (
            "group-a",
            {"E_53", "head_region_pain", "E_89", "E_91", "E_201"},
            4,
        ),
        ("group-b", {"E_201"}, 1),
    ]

    rows = []
    for group_id, present, unit_count in examples:
        row = {
            "source_group_id": group_id,
            "variant_id": "full",
            "canonical_label": "Example",
            "source_group_weight": 1.0,
            "reported_unit_count": unit_count,
            "AGE": 22,
            "SEX_M": 0,
        }
        for code in schema["symptom_features"]:
            row[f"{code}__present"] = int(code in present)
            row[f"{code}__absent"] = 0
        rows.append(row)

    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "setting,limit",
    [("up_to_1", 1), ("up_to_2", 2), ("up_to_3", 3), ("up_to_5", 5)],
)
def test_sparse_views_preserve_evidence_and_pain_parents(
    schema, full_frame, setting, limit
):
    result = make_validation_view(full_frame, schema, setting)

    assert result["reported_unit_count"].tolist() == [min(limit, 4), 1]
    assert result["AGE"].tolist() == [22, 22]
    assert result["SEX_M"].tolist() == [0, 0]
    assert result["variant_id"].eq(setting).all()

    for index in result.index:
        for code in schema["symptom_features"]:
            assert (
                result.loc[index, f"{code}__present"]
                <= full_frame.loc[index, f"{code}__present"]
            )
            assert result.loc[index, f"{code}__absent"] == 0

        if result.loc[index, "head_region_pain__present"] == 1:
            assert result.loc[index, "E_53__present"] == 1


def test_full_view_preserves_all_features(schema, full_frame):
    result = make_validation_view(full_frame, schema, "full")

    np.testing.assert_array_equal(
        result[schema["feature_names"]].to_numpy(),
        full_frame[schema["feature_names"]].to_numpy(),
    )


def test_view_is_reproducible_order_independent_and_label_independent(
    schema, full_frame
):
    original = full_frame.copy(deep=True)
    first = make_validation_view(full_frame, schema, "up_to_2")

    changed = full_frame.iloc[::-1].copy()
    changed["canonical_label"] = ["Different", "Labels"]
    second = make_validation_view(changed, schema, "up_to_2")

    columns = schema["feature_names"] + ["reported_unit_count"]

    np.testing.assert_array_equal(
        first.set_index("source_group_id").sort_index()[columns].to_numpy(),
        second.set_index("source_group_id").sort_index()[columns].to_numpy(),
    )
    pd.testing.assert_frame_equal(full_frame, original)


def test_v2_projection_preserves_order_and_general_pain(schema, full_frame):
    view = make_validation_view(full_frame, schema, "full")
    names = ["AGE", "SEX_M", "E_53__present", "E_201__present"]

    projected = project_v2_features(view, {"feature_names": names})

    assert projected.columns.tolist() == names
    assert projected.iloc[0]["E_53__present"] == 1
    assert "head_region_pain__present" not in projected.columns
    assert projected.index.equals(view.index)


def test_allergy_scores_are_combined_and_covid_is_retained():
    source = [
        "COVID-19", "Allergic sinusitis", "Pneumonia", "Allergic rhinitis"
    ]
    targets = ["Allergic rhinitis", "COVID-19", "Pneumonia"]

    result = harmonize_probabilities(
        [[0.25, 0.20, 0.30, 0.25]], source, targets
    )

    np.testing.assert_allclose(result, [[0.45, 0.25, 0.30]])
    assert targets[int(result[0].argmax())] == "Allergic rhinitis"


def test_candidate_alignment_does_not_invent_covid_probability():
    result = harmonize_probabilities(
        [[0.6, 0.4]],
        ["Allergic rhinitis", "Pneumonia"],
        ["Allergic rhinitis", "COVID-19", "Pneumonia"],
    )

    np.testing.assert_allclose(result, [[0.6, 0.0, 0.4]])


def test_unmapped_probability_class_is_rejected():
    with pytest.raises(ValueError, match="No target class"):
        harmonize_probabilities(
            [[0.5, 0.5]],
            ["Pneumonia", "COVID-19"],
            ["Pneumonia"],
        )


@pytest.mark.parametrize(
    "scores",
    [
        [[0.4, 0.4]],
        [[-0.1, 1.1]],
        [[float("nan"), 0.5]],
    ],
)
def test_invalid_probabilities_are_rejected(scores):
    with pytest.raises(ValueError, match="Invalid probability"):
        harmonize_probabilities(scores, ["A", "B"], ["A", "B"])


def test_duplicate_validation_groups_are_rejected(schema, full_frame):
    duplicated = pd.concat([full_frame, full_frame.iloc[[0]]])

    with pytest.raises(ValueError, match="unique strings"):
        make_validation_view(duplicated, schema, "up_to_1")