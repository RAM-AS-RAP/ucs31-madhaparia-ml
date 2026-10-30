"""Tests for source-group partitioning before candidate fitting."""

import hashlib
from collections import Counter
from copy import deepcopy

import pytest

from scripts.prepare_candidate_partitions import assign_partitions


def signature(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def example_groups(a_count=10, b_count=6):
    groups = []

    for label, count in (("A", a_count), ("B", b_count)):
        for number in range(count):
            groups.append({
                "source_group_id": signature(f"train:{label}:{number}"),
                "split": "train",
                "canonical_label": label,
            })

    for label, number in (("A", 0), ("A", 1), ("B", 0)):
        groups.append({
            "source_group_id": signature(f"validation:{label}:{number}"),
            "split": "validation",
            "canonical_label": label,
        })

    return groups


def test_stratified_reserve_and_validation_cohorts():
    groups = example_groups()
    legacy = {signature("validation:A:0")}

    rows = assign_partitions(groups, legacy)

    assert len(rows) == len(groups)
    assert len({row["source_group_id"] for row in rows}) == len(groups)

    assert Counter(row["partition"] for row in rows) == {
        "fit": 13,
        "calibration_reserved": 3,
        "selection_primary": 2,
        "selection_legacy_overlap": 1,
    }

    reserved = Counter(
        row["canonical_label"]
        for row in rows
        if row["partition"] == "calibration_reserved"
    )
    assert reserved == {"A": 2, "B": 1}

    for row in rows:
        if row["source_split"] == "train":
            assert row["partition"] in {"fit", "calibration_reserved"}
        else:
            assert row["partition"].startswith("selection_")

        assert (
            row["partition"] == "selection_legacy_overlap"
        ) == (row["source_group_id"] in legacy)


def test_assignment_is_order_independent_and_preserves_input():
    groups = example_groups()
    original = deepcopy(groups)
    legacy = {signature("validation:A:0")}

    first = assign_partitions(groups, legacy, seed=42)
    second = assign_partitions(list(reversed(groups)), legacy, seed=42)

    assert first == second
    assert groups == original


def test_small_classes_keep_fitting_and_reserved_groups():
    rows = assign_partitions(example_groups(a_count=2), set())

    counts = Counter(
        row["partition"]
        for row in rows
        if row["source_split"] == "train"
        and row["canonical_label"] == "A"
    )

    assert counts == {"fit": 1, "calibration_reserved": 1}


def test_single_group_class_is_rejected():
    with pytest.raises(ValueError, match="At least two training groups"):
        assign_partitions(example_groups(a_count=1), set())


def test_duplicate_group_id_across_splits_is_rejected():
    groups = example_groups()
    groups[-1]["source_group_id"] = groups[0]["source_group_id"]

    with pytest.raises(ValueError, match="Duplicate source-group ID"):
        assign_partitions(groups, set())


def test_unknown_legacy_group_is_rejected():
    with pytest.raises(ValueError, match="must belong to validation"):
        assign_partitions(example_groups(), {signature("unknown")})


@pytest.mark.parametrize("seed", [-1, True, "42"])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="non-negative integer"):
        assign_partitions(example_groups(), set(), seed=seed)