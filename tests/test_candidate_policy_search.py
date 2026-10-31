"""Tests for the predefined development-only policy search."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.candidate_policy_search import SETTINGS, search_thresholds


def example_views(count=150, score=0.9):
    return {
        setting: {
            "partition": "policy_development",
            "source_group_ids": np.asarray(
                [f"group-{number}" for number in range(count)],
                dtype=str,
            ),
            "classes": ["A", "B"],
            "target_indices": np.zeros(count, dtype=np.int64),
            "probabilities": np.tile([score, 1 - score], (count, 1)),
            "reported_unit_count": np.full(
                count, 1 if setting == "up_to_1" else 2,
                dtype=np.int64,
            ),
        }
        for setting in SETTINGS
    }


def test_search_uses_exact_grid_and_prespecified_threshold_tiebreaks():
    result = search_thresholds(example_views())
    policy = result["policy"]

    assert len(result["grid_summary"]) == 250
    assert len(result["grid_by_view"]) == 1250
    assert policy["mode"] == "thresholds"

    # All retained predictions are correct. Equal coverage and accuracy
    # therefore choose the highest score and margin still retaining them.
    assert policy["min_score"] == 0.9
    assert policy["min_margin"] == 0.2
    assert policy["deployment_approved"] is False

    summaries = {
        row["view"]: row for row in result["selected_view_summary"]
    }
    assert summaries["full"]["coverage"] == 1.0
    assert summaries["up_to_1"]["accepted"] == 0
    assert summaries["up_to_1"]["accepted_accuracy"] is None


def test_higher_coverage_is_preferred_to_a_stricter_threshold():
    views = example_views(count=200)

    for data in views.values():
        data["probabilities"][:100] = [0.99, 0.01]
        data["probabilities"][100:] = [0.75, 0.25]

    result = search_thresholds(views)

    assert result["policy"]["min_score"] == 0.75
    assert result["policy"]["development_mean_coverage"] == 1.0


def test_minimum_accepted_count_is_required_even_when_all_correct():
    result = search_thresholds(example_views(count=99))

    assert result["policy"]["mode"] == "withhold_all"
    assert result["policy"]["min_score"] is None
    assert result["policy"]["qualifying_threshold_pairs"] == 0
    assert all(
        row["accepted"] == 0
        for row in result["selected_view_summary"]
    )


def test_one_failing_required_view_prevents_selection():
    views = example_views()

    # Other views remain perfect. The two-unit view is always wrong.
    views["up_to_2"]["probabilities"][:] = [0.1, 0.9]

    result = search_thresholds(views)

    assert result["policy"]["mode"] == "withhold_all"
    assert result["policy"]["qualifying_threshold_pairs"] == 0


def test_view_row_order_does_not_change_selected_policy():
    views = example_views()
    original = search_thresholds(views)["policy"]

    reordered = deepcopy(views)
    for data in reordered.values():
        for name in (
            "source_group_ids",
            "target_indices",
            "probabilities",
            "reported_unit_count",
        ):
            data[name] = data[name][::-1].copy()

    assert search_thresholds(reordered)["policy"] == original


def test_evaluation_partition_is_rejected():
    views = example_views()
    views["full"]["partition"] = "uncertainty_evaluation"

    with pytest.raises(ValueError, match="Only policy_development"):
        search_thresholds(views)


def test_duplicate_group_is_rejected():
    views = example_views()
    views["full"]["source_group_ids"][1] = (
        views["full"]["source_group_ids"][0]
    )

    with pytest.raises(ValueError, match="unique aligned"):
        search_thresholds(views)


def test_cross_view_label_mismatch_is_rejected():
    views = example_views()
    views["up_to_2"]["target_indices"][0] = 1

    with pytest.raises(ValueError, match="groups or labels differ"):
        search_thresholds(views)


def test_cross_view_class_order_mismatch_is_rejected():
    views = example_views()
    views["up_to_2"]["classes"] = ["B", "A"]

    with pytest.raises(ValueError, match="Class order differs"):
        search_thresholds(views)


def test_missing_view_is_rejected():
    views = example_views()
    del views["up_to_5"]

    with pytest.raises(ValueError, match="all five"):
        search_thresholds(views)


def test_incorrect_view_unit_count_is_rejected():
    views = example_views()
    views["up_to_1"]["reported_unit_count"][0] = 2

    with pytest.raises(ValueError, match="exceeds its view limit"):
        search_thresholds(views)