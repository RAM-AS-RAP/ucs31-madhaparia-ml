"""Search the prespecified abstention thresholds on development data."""

import numpy as np

from scripts.candidate_probability_calibration import (
    validate_targets_and_weights,
)
from scripts.candidate_uncertainty_policy import (
    MIN_REPORTING_UNITS,
    assess_probability_rows,
    summarize_acceptance,
)


SETTINGS = ("full", "up_to_1", "up_to_2", "up_to_3", "up_to_5")
REQUIRED_SETTINGS = ("full", "up_to_2", "up_to_3", "up_to_5")
SCORE_THRESHOLDS = tuple(value / 100 for value in range(50, 100))
MARGIN_THRESHOLDS = (0.00, 0.05, 0.10, 0.15, 0.20)

MIN_ACCEPTED_GROUPS = 100
MIN_ACCURACY_WILSON_LOWER = 0.90


def search_thresholds(views):
    """Return the threshold grid, selected policy and per-view summaries."""

    if set(views) != set(SETTINGS):
        raise ValueError("Expected all five development views")

    prepared = {}
    reference_labels = None
    reference_classes = None

    for setting in SETTINGS:
        data = views[setting]

        if data.get("partition") != "policy_development":
            raise ValueError("Only policy_development views are permitted")

        probabilities = np.asarray(data["probabilities"], dtype=float)
        units = np.asarray(data["reported_unit_count"])

        decision = assess_probability_rows(
            probabilities,
            units,
            min_score=0.0,
            min_margin=0.0,
        )
        targets, _, _ = validate_targets_and_weights(
            probabilities,
            data["target_indices"],
            None,
        )

        classes = list(data["classes"])
        if (
            len(classes) != probabilities.shape[1]
            or any(not isinstance(label, str) or not label.strip()
                   for label in classes)
            or len(classes) != len(set(classes))
        ):
            raise ValueError("Invalid class definitions")

        if reference_classes is None:
            reference_classes = classes
        elif classes != reference_classes:
            raise ValueError("Class order differs between views")

        ids = np.asarray(data["source_group_ids"], dtype=object)
        if (
            ids.shape != (len(probabilities),)
            or any(not isinstance(value, str) or not value.strip()
                   for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Source-group IDs must be unique aligned strings")

        labels_by_id = dict(zip(ids.tolist(), targets.tolist()))

        if reference_labels is None:
            reference_labels = labels_by_id
        elif labels_by_id != reference_labels:
            raise ValueError("Source groups or labels differ between views")

        if setting != "full":
            cap = int(setting.rsplit("_", 1)[1])
            if (units > cap).any():
                raise ValueError("Reporting-unit count exceeds its view limit")

        prepared[setting] = {
            "enough_information": decision["enough_reporting_units"],
            "top_scores": decision["top_scores"],
            "margins": decision["score_margins"],
            "correct": decision["top_indices"] == targets,
        }

    grid_summary = []
    grid_by_view = []
    best = None
    best_key = None
    best_view_summaries = None

    for min_score in SCORE_THRESHOLDS:
        for min_margin in MARGIN_THRESHOLDS:
            summaries = {}

            for setting in SETTINGS:
                data = prepared[setting]
                accepted = (
                    data["enough_information"]
                    & (data["top_scores"] >= min_score)
                    & (data["margins"] >= min_margin)
                )

                summary = summarize_acceptance(data["correct"], accepted)
                summaries[setting] = summary

                grid_by_view.append({
                    "min_score": min_score,
                    "min_margin": min_margin,
                    "view": setting,
                    **summary,
                })

            eligible = all(
                summaries[setting]["accepted"] >= MIN_ACCEPTED_GROUPS
                and summaries[setting]["accepted_accuracy_wilson_lower"]
                is not None
                and summaries[setting]["accepted_accuracy_wilson_lower"]
                >= MIN_ACCURACY_WILSON_LOWER
                for setting in REQUIRED_SETTINGS
            )

            mean_coverage = float(np.mean([
                summaries[setting]["coverage"]
                for setting in REQUIRED_SETTINGS
            ]))

            accuracies = [
                summaries[setting]["accepted_accuracy"]
                for setting in REQUIRED_SETTINGS
            ]
            mean_accuracy = (
                float(np.mean(accuracies))
                if all(value is not None for value in accuracies)
                else None
            )

            candidate = {
                "min_score": min_score,
                "min_margin": min_margin,
                "eligible": eligible,
                "mean_coverage": mean_coverage,
                "mean_accepted_accuracy": mean_accuracy,
            }
            grid_summary.append(candidate)

            if not eligible:
                continue

            # Prespecified order:
            # coverage, accepted accuracy, score threshold, margin threshold.
            key = (
                mean_coverage,
                mean_accuracy,
                min_score,
                min_margin,
            )

            if best_key is None or key > best_key:
                best = candidate
                best_key = key
                best_view_summaries = summaries

    if best is None:
        policy = {
            "status": "no_qualifying_thresholds",
            "mode": "withhold_all",
            "min_score": None,
            "min_margin": None,
        }
        selected_summaries = {
            setting: summarize_acceptance(
                data["correct"],
                np.zeros(len(data["correct"]), dtype=bool),
            )
            for setting, data in prepared.items()
        }
    else:
        policy = {
            "status": "selected_on_policy_development",
            "mode": "thresholds",
            "min_score": best["min_score"],
            "min_margin": best["min_margin"],
            "development_mean_coverage": best["mean_coverage"],
            "development_mean_accepted_accuracy": (
                best["mean_accepted_accuracy"]
            ),
        }
        selected_summaries = best_view_summaries

    policy.update({
        "selection_partition": "policy_development",
        "min_reporting_units": MIN_REPORTING_UNITS,
        "required_views": list(REQUIRED_SETTINGS),
        "minimum_accepted_per_required_view": MIN_ACCEPTED_GROUPS,
        "minimum_accuracy_wilson_lower": MIN_ACCURACY_WILSON_LOWER,
        "wilson_interval": "two-sided 95%; lower endpoint",
        "threshold_pairs_evaluated": len(grid_summary),
        "qualifying_threshold_pairs": sum(
            row["eligible"] for row in grid_summary
        ),
        "deployment_approved": False,
    })

    return {
        "policy": policy,
        "grid_summary": grid_summary,
        "grid_by_view": grid_by_view,
        "selected_view_summary": [
            {"view": setting, **selected_summaries[setting]}
            for setting in SETTINGS
        ],
    }