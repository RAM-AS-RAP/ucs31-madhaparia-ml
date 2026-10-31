"""Tests for the reserved-data subdivision used in issue #49."""

import hashlib
from collections import Counter
from copy import deepcopy

import pytest

from scripts.prepare_candidate_calibration_partitions import (
    assign_calibration_partitions,
)


def signature(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def example_groups(a_count=10, b_count=7):
    return [
        {
            "source_group_id": signature(f"{label}:{number}"),
            "source_split": "train",
            "canonical_label": label,
            "partition": "calibration_reserved",
        }
        for label, count in (("A", a_count), ("B", b_count))
        for number in range(count)
    ]


def test_each_group_assigned_once_with_stratified_counts():
    groups = example_groups()
    rows = assign_calibration_partitions(groups)

    assert len(rows) == 17
    assert len({row["source_group_id"] for row in rows}) == 17
    assert {row["source_group_id"] for row in rows} == {
        row["source_group_id"] for row in groups
    }

    expected = {
        "A": {
            "calibration_fit": 6,
            "policy_development": 2,
            "uncertainty_evaluation": 2,
        },
        "B": {
            "calibration_fit": 4,
            "policy_development": 1,
            "uncertainty_evaluation": 2,
        },
    }

    for label, counts in expected.items():
        assert Counter(
            row["partition"]
            for row in rows
            if row["canonical_label"] == label
        ) == counts

    assert all(row["source_split"] == "train" for row in rows)
    assert all(
        row["parent_partition"] == "calibration_reserved"
        for row in rows
    )


def test_order_independent_and_does_not_modify_inputs():
    groups = example_groups()
    original = deepcopy(groups)

    first = assign_calibration_partitions(groups)
    second = assign_calibration_partitions(list(reversed(groups)))

    assert first == second
    assert groups == original


def test_membership_follows_prespecified_hash_order():
    groups = example_groups(a_count=10, b_count=0)

    ordered = sorted(
        (row["source_group_id"] for row in groups),
        key=lambda group_id: (
            signature("49:calibration_partition:" + group_id),
            group_id,
        ),
    )

    actual = {
        row["source_group_id"]: row["partition"]
        for row in assign_calibration_partitions(groups)
    }

    assert all(actual[group_id] == "calibration_fit"
               for group_id in ordered[:6])
    assert all(actual[group_id] == "policy_development"
               for group_id in ordered[6:8])
    assert all(actual[group_id] == "uncertainty_evaluation"
               for group_id in ordered[8:])


def test_five_groups_keep_all_three_partitions():
    rows = assign_calibration_partitions(
        example_groups(a_count=5, b_count=0)
    )

    assert Counter(row["partition"] for row in rows) == {
        "calibration_fit": 3,
        "policy_development": 1,
        "uncertainty_evaluation": 1,
    }


def test_too_small_class_is_rejected():
    with pytest.raises(ValueError, match="Insufficient reserved groups"):
        assign_calibration_partitions(example_groups(a_count=4))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="No reserved groups"):
        assign_calibration_partitions([])


def test_duplicate_id_across_conditions_is_rejected():
    groups = example_groups()
    groups[-1]["source_group_id"] = groups[0]["source_group_id"]

    with pytest.raises(ValueError, match="Duplicate source-group ID"):
        assign_calibration_partitions(groups)


@pytest.mark.parametrize(
    "field,value",
    [
        ("partition", "fit"),
        ("partition", "selection_primary"),
        ("source_split", "validation"),
    ],
)
def test_non_reserved_or_validation_groups_are_rejected(field, value):
    groups = example_groups()
    groups[0][field] = value

    with pytest.raises(ValueError, match="Only calibration_reserved"):
        assign_calibration_partitions(groups)


@pytest.mark.parametrize("group_id", ["short", "z" * 64, None])
def test_invalid_signatures_are_rejected(group_id):
    groups = example_groups()
    groups[0]["source_group_id"] = group_id

    with pytest.raises(ValueError, match="Invalid source-group signature"):
        assign_calibration_partitions(groups)