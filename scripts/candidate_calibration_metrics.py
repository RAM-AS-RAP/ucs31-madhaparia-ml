"""Probability and classification metrics for one observation view."""

import numpy as np

from scripts.candidate_comparison_metrics import score_predictions
from scripts.candidate_probability_calibration import (
    negative_log_likelihood,
    validate_probabilities,
)


SETTINGS = ("full", "up_to_1", "up_to_2", "up_to_3", "up_to_5")
MIN_LOG_LOSS_IMPROVEMENT = 1e-6


def evaluate_probability_view(actual, probabilities, classes):
    """Evaluate one view with one observation per source group.

    Brier score uses the sum over classes, without dividing by class count.
    Reliability bins describe top-label confidence and correctness.
    """

    values = validate_probabilities(probabilities)
    classes = list(classes)
    actual = np.asarray(actual, dtype=object)

    if (
        len(classes) != values.shape[1]
        or any(not isinstance(label, str) or not label.strip()
               for label in classes)
        or len(classes) != len(set(classes))
    ):
        raise ValueError("Invalid probability class definitions")

    if actual.shape != (len(values),):
        raise ValueError("Actual labels must match probability rows")

    positions = {label: index for index, label in enumerate(classes)}

    if any(
        not isinstance(label, str) or label not in positions
        for label in actual
    ):
        raise ValueError("Unexpected actual label")

    targets = np.asarray(
        [positions[label] for label in actual],
        dtype=np.int64,
    )
    predicted_indices = values.argmax(axis=1)
    predicted = np.asarray(classes, dtype=object)[predicted_indices]

    result = score_predictions(
        actual,
        predicted,
        ground_truth_labels=classes,
        output_labels=classes,
    )

    expected = np.zeros_like(values)
    expected[np.arange(len(values)), targets] = 1.0

    brier = float(
        np.mean(np.sum((values - expected) ** 2, axis=1))
    )

    top_scores = values.max(axis=1)
    correct = predicted_indices == targets

    # Bins are [0.0, 0.1), ..., [0.9, 1.0].
    # A score of exactly 1 belongs to the final bin.
    bin_indices = np.minimum(
        np.floor(top_scores * 10).astype(np.int64),
        9,
    )

    reliability_bins = []
    expected_calibration_error = 0.0

    for index in range(10):
        selected = bin_indices == index
        count = int(selected.sum())
        correct_count = int(correct[selected].sum())

        if count:
            mean_score = float(top_scores[selected].mean())
            accuracy = correct_count / count
            gap = abs(accuracy - mean_score)
            expected_calibration_error += count / len(values) * gap
        else:
            mean_score = None
            accuracy = None
            gap = None

        reliability_bins.append({
            "bin_index": index,
            "lower_bound": index / 10,
            "upper_bound": (index + 1) / 10,
            "upper_bound_inclusive": index == 9,
            "count": count,
            "correct_count": correct_count,
            "mean_top_score": mean_score,
            "accuracy": accuracy,
            "absolute_gap": gap,
        })

    if sum(row["count"] for row in reliability_bins) != len(values):
        raise ValueError("Reliability bins lost observations")

    result["summary"].update({
        "log_loss": negative_log_likelihood(values, targets),
        "multiclass_brier_score": brier,
        "top_label_ece": float(expected_calibration_error),
        "mean_top_score": float(top_scores.mean()),
    })
    result["reliability_bins"] = reliability_bins
    result["notes"].extend([
        "Brier score is not divided by the number of classes.",
        "Top-label ECE uses 10 equal-width confidence bins.",
        "Empty reliability bins have null accuracy and mean score.",
        "Calibration metrics apply to this observation distribution.",
        "Different views must not be counted as independent patients.",
    ])

    return result


def select_probability_representation(original_losses, scaled_losses):
    """Apply the prespecified policy-development log-loss criterion."""

    for name, losses in (
        ("original", original_losses),
        ("temperature_scaled", scaled_losses),
    ):
        if set(losses) != set(SETTINGS):
            raise ValueError(f"Expected all five views for {name}")

        values = np.asarray(
            [losses[setting] for setting in SETTINGS],
            dtype=float,
        )
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid log losses for {name}")

    original_mean = float(np.mean([
        original_losses[setting] for setting in SETTINGS
    ]))
    scaled_mean = float(np.mean([
        scaled_losses[setting] for setting in SETTINGS
    ]))
    improvement = original_mean - scaled_mean

    selected = (
        "temperature_scaled"
        if improvement > MIN_LOG_LOSS_IMPROVEMENT
        else "original"
    )

    return {
        "selected_representation": selected,
        "criterion": (
            "Temperature scaling must improve mean policy-development "
            "log loss across five views by more than 1e-6"
        ),
        "original_mean_log_loss": original_mean,
        "temperature_scaled_mean_log_loss": scaled_mean,
        "log_loss_improvement": improvement,
        "required_improvement_exclusive": MIN_LOG_LOSS_IMPROVEMENT,
        "deployment_approved": False,
    }