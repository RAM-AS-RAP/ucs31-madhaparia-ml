"""Check metric calculations against small, known examples."""

import pytest

from scripts.candidate_comparison_metrics import score_predictions


def score(actual, predicted):
    return score_predictions(
        actual,
        predicted,
        ground_truth_labels=["A", "B"],
        output_labels=["A", "B", "COVID-19"],
    )


def test_known_accuracy_precision_recall_and_f1():
    result = score(
        ["A", "A", "B", "B"],
        ["A", "COVID-19", "B", "A"],
    )
    summary = result["summary"]

    assert summary["observations"] == 4
    assert summary["accuracy"] == 0.5
    assert summary["macro_precision"] == pytest.approx(0.75)
    assert summary["macro_recall"] == pytest.approx(0.5)
    assert summary["macro_f1"] == pytest.approx(7 / 12)
    assert summary["weighted_f1"] == pytest.approx(7 / 12)

    assert result["confusion_matrix"] == [
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 0],
    ]

    covid = next(
        row for row in result["per_condition"]
        if row["condition"] == "COVID-19"
    )
    assert covid["support"] == 0
    assert covid["predicted_count"] == 1
    assert covid["precision"] == 0.0
    assert covid["recall"] is None
    assert covid["f1"] is None
    assert covid["included_in_macro_average"] is False


def test_covid_predictions_are_not_dropped_from_accuracy():
    result = score(["A", "B"], ["COVID-19", "COVID-19"])

    assert result["summary"]["observations"] == 2
    assert result["summary"]["accuracy"] == 0.0
    assert result["summary"]["macro_f1"] == 0.0
    assert sum(map(sum, result["confusion_matrix"])) == 2


def test_fixed_macro_labels_and_support_are_explicit():
    result = score(["A", "A"], ["A", "A"])

    assert result["summary"]["accuracy"] == 1.0
    assert result["summary"]["macro_f1"] == 0.5
    assert result["summary"]["weighted_f1"] == 1.0
    assert result["summary"]["macro_label_count"] == 2
    assert result["summary"]["supported_ground_truth_classes"] == 1


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([], []),
        (["A"], []),
        (["A", "B"], ["A"]),
    ],
)
def test_empty_or_misaligned_arrays_are_rejected(actual, predicted):
    with pytest.raises(ValueError, match="aligned prediction arrays"):
        score(actual, predicted)


def test_unexpected_ground_truth_is_rejected():
    with pytest.raises(ValueError, match="ground-truth label"):
        score(["COVID-19"], ["A"])


def test_unexpected_prediction_is_rejected():
    with pytest.raises(ValueError, match="predicted label"):
        score(["A"], ["Unknown condition"])