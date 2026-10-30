"""Classification metrics with explicit ground-truth and output label sets."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def score_predictions(
    actual,
    predicted,
    *,
    ground_truth_labels,
    output_labels,
):
    """Calculate unweighted metrics for one observation per source group.

    Macro metrics use the fixed ground-truth label set, with zero_division=0.
    Extra model outputs remain errors when they do not match ground truth.
    """

    actual = np.asarray(actual, dtype=object)
    predicted = np.asarray(predicted, dtype=object)
    truth_labels = list(ground_truth_labels)
    outputs = list(output_labels)

    if (
        actual.ndim != 1
        or predicted.ndim != 1
        or len(actual) == 0
        or len(actual) != len(predicted)
    ):
        raise ValueError("Expected non-empty, aligned prediction arrays")

    if (
        not truth_labels
        or len(truth_labels) != len(set(truth_labels))
        or len(outputs) != len(set(outputs))
        or not set(truth_labels).issubset(outputs)
    ):
        raise ValueError("Invalid evaluation label definitions")

    if not set(actual).issubset(truth_labels):
        raise ValueError("Unexpected ground-truth label")

    if not set(predicted).issubset(outputs):
        raise ValueError("Unexpected predicted label")

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=outputs,
        zero_division=0,
    )

    matrix = confusion_matrix(actual, predicted, labels=outputs)
    if int(matrix.sum()) != len(actual):
        raise ValueError("Confusion matrix lost observations")

    predicted_counts = matrix.sum(axis=0)
    positions = {label: index for index, label in enumerate(outputs)}
    truth_positions = [positions[label] for label in truth_labels]

    truth_support = support[truth_positions]

    summary = {
        "observations": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision[truth_positions].mean()),
        "macro_recall": float(recall[truth_positions].mean()),
        "macro_f1": float(f1[truth_positions].mean()),
        "weighted_f1": float(
            np.average(f1[truth_positions], weights=truth_support)
        ),
        "macro_label_count": len(truth_labels),
        "supported_ground_truth_classes": int(
            np.count_nonzero(truth_support)
        ),
    }

    per_condition = []

    for index, label in enumerate(outputs):
        has_support = support[index] > 0
        has_predictions = predicted_counts[index] > 0

        per_condition.append({
            "condition": label,
            "support": int(support[index]),
            "predicted_count": int(predicted_counts[index]),
            "precision": (
                float(precision[index]) if has_predictions else None
            ),
            "recall": float(recall[index]) if has_support else None,
            "f1": float(f1[index]) if has_support else None,
            "included_in_macro_average": label in truth_labels,
        })

    return {
        "summary": summary,
        "per_condition": per_condition,
        "confusion_labels": outputs,
        "confusion_matrix": matrix.tolist(),
        "notes": [
            "Summary scores are proportions between 0 and 1.",
            "Macro metrics use the fixed ground-truth label set.",
            "Undefined class contributions to macro metrics use zero.",
            "Null per-condition values indicate unavailable denominators.",
            "Recall is not established for classes without ground-truth cases.",
        ],
    }