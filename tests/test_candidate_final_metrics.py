"""Artificial-probability tests for final cohort evaluation."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.candidate_final_metrics import evaluate_final_cohorts


@pytest.fixture
def example():
    return {
        "view": "up_to_2",
        "source_group_ids": ["first", "second", "third"],
        "actual": ["A", "B", "B"],
        "cohort_assignments": ["primary", "primary", "overlap"],
        "reported_unit_counts": np.array([2, 2, 1]),
        "candidate_probabilities": np.array([
            [0.90, 0.10],
            [0.60, 0.40],
            [0.05, 0.95],
        ]),
        "candidate_classes": ["A", "B"],
        "v2_probabilities": np.array([
            [0.10, 0.10, 0.80],
            [0.10, 0.80, 0.10],
            [0.10, 0.80, 0.10],
        ]),
        "v2_classes": ["A", "B", "COVID-19"],
        "temperature": 1.0,
        "policy": {
            "classes": ["A", "B"],
            "mode": "thresholds",
            "min_reporting_units": 2,
            "min_score": 0.78,
            "min_margin": 0.20,
            "minimum_accepted_per_required_view": 100,
            "minimum_accuracy_wilson_lower": 0.90,
            "required_views": ["full", "up_to_2", "up_to_3", "up_to_5"],
            "probability_representation": "temperature_scaled",
            "temperature": 1.0,
        },
    }


def test_cohort_denominators_and_acceptance(example):
    report, arrays = evaluate_final_cohorts(**example)
    cohorts = report["cohorts"]

    primary = cohorts["primary"]["candidate_policy"]["summary"]
    complete = cohorts["complete_eligible"]["candidate_policy"]["summary"]
    overlap = cohorts["overlap"]["candidate_policy"]["summary"]

    assert primary["observations"] == 2
    assert primary["unfiltered_accuracy"] == pytest.approx(0.5)
    assert primary["coverage"] == pytest.approx(0.5)
    assert primary["accepted_accuracy"] == 1.0
    assert primary["required_for_research_target"] is True

    assert complete["observations"] == 3
    assert complete["unfiltered_accuracy"] == pytest.approx(2 / 3)
    assert complete["coverage"] == pytest.approx(1 / 3)
    assert complete["required_for_research_target"] is False

    assert overlap["observations"] == 1
    assert overlap["accepted"] == 0
    assert overlap["accepted_accuracy"] is None

    np.testing.assert_array_equal(
        arrays["candidate_accepted"], [True, False, False]
    )


def test_v2_covid_output_remains_an_error(example):
    report, arrays = evaluate_final_cohorts(**example)
    result = report["cohorts"]["primary"]["v2_classification"]

    assert result["summary"]["accuracy"] == 0.5
    assert result["summary"]["macro_label_count"] == 2
    covid = next(
        row for row in result["per_condition"]
        if row["condition"] == "COVID-19"
    )
    assert covid["predicted_count"] == 1
    assert covid["support"] == 0
    assert covid["included_in_macro_average"] is False
    assert arrays["v2_harmonized_probabilities"].sum(axis=1) == pytest.approx(
        [1, 1, 1]
    )


def test_v2_allergy_scores_are_combined_before_ranking(example):
    changed = deepcopy(example)
    changed["candidate_classes"] = ["Allergic rhinitis", "Bronchitis"]
    changed["policy"]["classes"] = changed["candidate_classes"].copy()
    changed["actual"] = ["Allergic rhinitis", "Bronchitis", "Bronchitis"]
    changed["v2_classes"] = [
        "Allergic sinusitis", "Allergic rhinitis", "Bronchitis", "COVID-19"
    ]
    changed["v2_probabilities"] = np.array([
        [0.30, 0.25, 0.35, 0.10],
        [0.05, 0.05, 0.80, 0.10],
        [0.05, 0.05, 0.80, 0.10],
    ])

    report, arrays = evaluate_final_cohorts(**changed)
    assert report["cohorts"]["primary"]["v2_classification"]["summary"]["accuracy"] == 1

    names = arrays["v2_harmonized_classes"].tolist()
    assert arrays["v2_harmonized_probabilities"][
        0, names.index("Allergic rhinitis")
    ] == pytest.approx(0.55)
    assert arrays["v2_harmonized_probabilities"][
        0, names.index("COVID-19")
    ] == pytest.approx(0.10)


def test_empty_primary_is_not_replaced_by_complete_cohort(example):
    example["cohort_assignments"] = ["overlap"] * 3
    report, _ = evaluate_final_cohorts(**example)

    primary = report["cohorts"]["primary"]
    assert primary["status"] == "not_assessable"
    assert primary["observations"] == 0
    assert primary["required_for_research_target"] is True
    assert "candidate_policy" not in primary
    assert report["cohorts"]["complete_eligible"]["status"] == "evaluated"


def test_one_unit_view_withholds_all_cases(example):
    example["view"] = "up_to_1"
    example["reported_unit_counts"] = np.ones(3, dtype=int)
    report, arrays = evaluate_final_cohorts(**example)

    assert not arrays["candidate_accepted"].any()
    summary = report["cohorts"]["primary"]["candidate_policy"]["summary"]
    assert summary["accepted_accuracy"] is None
    assert summary["required_for_research_target"] is False


def test_withheld_true_cases_remain_in_recall_denominator(example):
    report, _ = evaluate_final_cohorts(**example)
    rows = report["cohorts"]["complete_eligible"]["candidate_policy"]["per_condition"]
    condition = next(row for row in rows if row["condition"] == "B")

    assert condition["support_all"] == 2
    assert condition["withheld_true_cases"] == 2
    assert condition["correct_accepted_recall_all_cases"] == 0.0


@pytest.mark.parametrize("assignments", [
    ["primary", "unknown", "overlap"],
    ["primary"],
])
def test_invalid_cohorts_are_rejected(example, assignments):
    example["cohort_assignments"] = assignments
    with pytest.raises(ValueError):
        evaluate_final_cohorts(**example)


def test_duplicate_group_ids_are_rejected(example):
    example["source_group_ids"] = ["same", "same", "third"]
    with pytest.raises(ValueError, match="unique"):
        evaluate_final_cohorts(**example)


@pytest.mark.parametrize("units", [
    [0, 2, 1],
    [2, 3, 1],
    [2.0, 2.0, 1.0],
    [True, True, True],
])
def test_invalid_reporting_unit_counts_are_rejected(example, units):
    example["reported_unit_counts"] = np.asarray(units)
    with pytest.raises(ValueError):
        evaluate_final_cohorts(**example)


def test_changed_temperature_is_rejected(example):
    example["temperature"] = 0.64
    with pytest.raises(ValueError, match="Policy temperature differs"):
        evaluate_final_cohorts(**example)


def test_reordered_candidate_classes_are_rejected(example):
    example["candidate_classes"] = ["B", "A"]
    with pytest.raises(ValueError, match="Candidate class"):
        evaluate_final_cohorts(**example)


def test_unknown_actual_label_is_rejected(example):
    example["actual"][0] = "Outside scope"
    with pytest.raises(ValueError, match="actual labels"):
        evaluate_final_cohorts(**example)


def test_input_policy_and_arrays_are_not_modified(example):
    original = deepcopy(example)
    evaluate_final_cohorts(**example)

    assert example["policy"] == original["policy"]
    for key in (
        "candidate_probabilities",
        "v2_probabilities",
        "reported_unit_counts",
    ):
        np.testing.assert_array_equal(example[key], original[key])