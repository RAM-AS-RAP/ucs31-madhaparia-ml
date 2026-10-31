"""Evaluate a frozen abstention policy without selecting new thresholds."""

import numpy as np

from scripts.candidate_comparison_metrics import score_predictions
from scripts.candidate_uncertainty_policy import (
    MIN_REPORTING_UNITS,
    assess_probability_rows,
    summarize_acceptance,
    wilson_interval,
)


def evaluate_policy_view(
    actual,
    probabilities,
    classes,
    reported_unit_counts,
    policy,
):
    """Return a report and acceptance mask for one observation view.

    Probabilities must already use the policy's selected representation.
    This function does not fit calibration or search thresholds.
    """

    classes = list(classes)
    actual = np.asarray(actual, dtype=object)

    if classes != list(policy["classes"]):
        raise ValueError("Class order differs from the frozen policy")

    if policy["min_reporting_units"] != MIN_REPORTING_UNITS:
        raise ValueError("Information gate differs from the protocol")

    mode = policy["mode"]
    if mode not in {"thresholds", "withhold_all"}:
        raise ValueError("Unknown policy mode")

    if mode == "withhold_all":
        if policy["min_score"] is not None or policy["min_margin"] is not None:
            raise ValueError("Withhold-all policy must not contain thresholds")
        min_score, min_margin = 0.0, 0.0
    else:
        min_score = policy["min_score"]
        min_margin = policy["min_margin"]

    decision = assess_probability_rows(
        probabilities,
        reported_unit_counts,
        min_score=min_score,
        min_margin=min_margin,
    )

    predicted = np.asarray(classes, dtype=object)[decision["top_indices"]]

    unfiltered = score_predictions(
        actual,
        predicted,
        ground_truth_labels=classes,
        output_labels=classes,
    )

    accepted = decision["accepted"].copy()
    if mode == "withhold_all":
        accepted[:] = False

    correct = actual == predicted
    summary = summarize_acceptance(correct, accepted)

    coverage_lower, coverage_upper = wilson_interval(
        summary["accepted"], summary["observations"]
    )
    summary.update({
        "unfiltered_accuracy": unfiltered["summary"]["accuracy"],
        "coverage_wilson_lower": coverage_lower,
        "coverage_wilson_upper": coverage_upper,
    })

    minimum_count = policy["minimum_accepted_per_required_view"]
    minimum_lower = policy["minimum_accuracy_wilson_lower"]

    if minimum_count != 100 or minimum_lower != 0.90:
        raise ValueError("Research targets differ from the frozen protocol")

    lower = summary["accepted_accuracy_wilson_lower"]
    summary["meets_count_and_accuracy_targets"] = bool(
        summary["accepted"] >= minimum_count
        and lower is not None
        and lower >= minimum_lower
    )

    if accepted.any():
        accepted_classification = score_predictions(
            actual[accepted],
            predicted[accepted],
            ground_truth_labels=classes,
            output_labels=classes,
        )
        accepted_by_class = {
            row["condition"]: row
            for row in accepted_classification["per_condition"]
        }
    else:
        accepted_classification = None
        accepted_by_class = {}

    unfiltered_by_class = {
        row["condition"]: row
        for row in unfiltered["per_condition"]
    }

    per_condition = []

    for condition in classes:
        true_cases = actual == condition
        accepted_true = true_cases & accepted

        support = int(true_cases.sum())
        accepted_support = int(accepted_true.sum())
        correct_accepted = int((accepted_true & correct).sum())
        accepted_predictions = int(
            ((predicted == condition) & accepted).sum()
        )

        recall_lower, recall_upper = wilson_interval(
            correct_accepted, support
        )
        conditional = accepted_by_class.get(condition)

        per_condition.append({
            "condition": condition,
            "support_all": support,
            "accepted_true_cases": accepted_support,
            "withheld_true_cases": support - accepted_support,
            "acceptance_rate_for_true_cases": (
                accepted_support / support if support else None
            ),
            "accepted_predictions_of_condition": accepted_predictions,
            "correct_accepted": correct_accepted,
            "unfiltered_recall": (
                unfiltered_by_class[condition]["recall"]
            ),
            "accepted_precision": (
                conditional["precision"] if conditional else None
            ),
            "accepted_recall": (
                conditional["recall"] if conditional else None
            ),
            "accepted_f1": (
                conditional["f1"] if conditional else None
            ),
            "correct_accepted_recall_all_cases": (
                correct_accepted / support if support else None
            ),
            "correct_accepted_recall_wilson_lower": recall_lower,
            "correct_accepted_recall_wilson_upper": recall_upper,
        })

    units = np.asarray(reported_unit_counts)
    by_reporting_units = []

    for count in np.unique(units):
        selected = units == count
        by_reporting_units.append({
            "reported_unit_count": int(count),
            **summarize_acceptance(
                correct[selected],
                accepted[selected],
            ),
        })

    report = {
        "summary": summary,
        "per_condition": per_condition,
        "by_reporting_units": by_reporting_units,
        "unfiltered_classification": unfiltered,
        "accepted_classification": accepted_classification,
        "notes": [
            "Accepted accuracy excludes withheld observations.",
            "Accepted recall uses only accepted true cases as denominator.",
            "Correct-accepted recall over all cases includes withheld cases "
            "in its denominator.",
            "Undefined rates are null, not zero or perfect performance.",
            "Wilson intervals use their explicitly reported denominators.",
            "The count and accuracy target applies only to required views.",
            "No calibration parameters or policy thresholds were changed.",
        ],
    }

    return report, accepted