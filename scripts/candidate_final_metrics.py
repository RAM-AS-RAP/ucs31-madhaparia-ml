"""Evaluate frozen models on shared final-test observations."""

import numpy as np

from scripts.candidate_calibration_metrics import evaluate_probability_view
from scripts.candidate_comparison_metrics import score_predictions
from scripts.candidate_evaluation import VIEW_LIMITS, harmonize_probabilities
from scripts.candidate_probability_calibration import (
    TEMPERATURE_BOUNDS,
    temperature_scale,
    validate_probabilities,
)
from scripts.candidate_selective_evaluation import evaluate_policy_view


def evaluate_final_cohorts(
    *,
    view,
    source_group_ids,
    actual,
    cohort_assignments,
    reported_unit_counts,
    candidate_probabilities,
    candidate_classes,
    v2_probabilities,
    v2_classes,
    temperature,
    policy,
):
    """Return cohort reports and derived arrays without fitting anything."""
    if view not in VIEW_LIMITS:
        raise ValueError("Unknown observation view")

    original = validate_probabilities(candidate_probabilities)
    v2_native = validate_probabilities(v2_probabilities)
    count = len(original)

    ids = np.asarray(source_group_ids)
    labels = np.asarray(actual, dtype=object)
    assignments = np.asarray(cohort_assignments, dtype=object)
    units = np.asarray(reported_unit_counts)
    candidate_classes = list(candidate_classes)
    v2_classes = list(v2_classes)

    if (
        ids.shape != (count,)
        or any(not isinstance(value, str) or not value.strip() for value in ids)
        or len(set(ids.tolist())) != count
    ):
        raise ValueError("Expected unique, non-empty source-group IDs")

    if labels.shape != (count,) or assignments.shape != (count,):
        raise ValueError("Labels and cohorts must match probability rows")

    if not set(assignments.tolist()).issubset({"primary", "overlap"}):
        raise ValueError("Unknown cohort assignment")

    if (
        units.shape != (count,)
        or not np.issubdtype(units.dtype, np.integer)
        or np.issubdtype(units.dtype, np.bool_)
        or (units < 1).any()
    ):
        raise ValueError("Expected positive integer reporting-unit counts")

    limit = VIEW_LIMITS[view]
    if limit is not None and (units > limit).any():
        raise ValueError("Reporting-unit counts exceed the view limit")

    if (
        candidate_classes != list(policy["classes"])
        or len(candidate_classes) != len(set(candidate_classes))
        or original.shape[1] != len(candidate_classes)
        or not set(labels.tolist()).issubset(candidate_classes)
    ):
        raise ValueError("Candidate class definitions or actual labels differ")

    if (
        len(v2_native) != count
        or v2_native.shape[1] != len(v2_classes)
        or len(v2_classes) != len(set(v2_classes))
    ):
        raise ValueError("V2 rows or class definitions differ")

    v2_output_classes = sorted(set(candidate_classes) | {"COVID-19"})
    canonical_v2 = {
        "Allergic rhinitis" if name == "Allergic sinusitis" else name
        for name in v2_classes
    }
    if canonical_v2 != set(v2_output_classes):
        raise ValueError("V2 canonical outputs differ from the expected scope")

    if (
        isinstance(temperature, (bool, np.bool_))
        or not np.isscalar(temperature)
        or not np.isfinite(temperature)
        or not TEMPERATURE_BOUNDS[0] <= temperature <= TEMPERATURE_BOUNDS[1]
    ):
        raise ValueError("Invalid fitted temperature")

    representation = policy["probability_representation"]
    if representation == "temperature_scaled":
        if policy["temperature"] != temperature:
            raise ValueError("Policy temperature differs from fitted temperature")
    elif representation == "original":
        if policy["temperature"] is not None:
            raise ValueError("Original representation must not apply temperature")
    else:
        raise ValueError("Unknown probability representation")

    scaled = temperature_scale(original, temperature)
    np.testing.assert_array_equal(
        original.argmax(axis=1), scaled.argmax(axis=1)
    )

    v2_harmonized = harmonize_probabilities(
        v2_native, v2_classes, v2_output_classes
    )
    v2_predicted = np.asarray(v2_output_classes, dtype=object)[
        v2_harmonized.argmax(axis=1)
    ]
    selected_probabilities = (
        scaled if representation == "temperature_scaled" else original
    )

    masks = {
        "primary": assignments == "primary",
        "complete_eligible": np.ones(count, dtype=bool),
        "overlap": assignments == "overlap",
    }

    reports = {}
    acceptance = {}

    for cohort, mask in masks.items():
        cohort_count = int(mask.sum())
        required = cohort == "primary" and view in policy["required_views"]

        if cohort_count == 0:
            reports[cohort] = {
                "status": "not_assessable",
                "observations": 0,
                "required_for_research_target": required,
                "reason": "No observations in this cohort",
            }
            continue

        v2_report = score_predictions(
            labels[mask],
            v2_predicted[mask],
            ground_truth_labels=candidate_classes,
            output_labels=v2_output_classes,
        )

        probability_reports = {
            name: evaluate_probability_view(
                labels[mask], probabilities[mask], candidate_classes
            )
            for name, probabilities in (
                ("original", original),
                ("temperature_scaled", scaled),
            )
        }

        policy_report, accepted = evaluate_policy_view(
            labels[mask],
            selected_probabilities[mask],
            candidate_classes,
            units[mask],
            policy,
        )
        policy_report["summary"]["required_for_research_target"] = required
        acceptance[cohort] = accepted

        reports[cohort] = {
            "status": "evaluated",
            "observations": cohort_count,
            "required_for_research_target": required,
            "v2_classification": v2_report,
            "candidate_probability_metrics": probability_reports,
            "candidate_policy": policy_report,
        }

    combined = acceptance["complete_eligible"]
    for cohort in ("primary", "overlap"):
        if masks[cohort].any():
            np.testing.assert_array_equal(
                combined[masks[cohort]], acceptance[cohort]
            )

    if view == "up_to_1" and combined.any():
        raise AssertionError("One-unit observations received accepted predictions")

    report = {
        "view": view,
        "cohorts": reports,
        "notes": [
            "Models use identical source observations and cohort assignments.",
            "Complete eligible includes primary and overlap groups.",
            "COVID-19 remains a possible incorrect V2 output.",
            "V2 allergy scores are combined without dropping probability mass.",
            "Candidate uncertainty thresholds are not applied to V2.",
            "Candidate probability metrics precede withholding.",
            "Accepted accuracy excludes withheld observations.",
            "Correct-accepted recall over all cases includes withheld true cases.",
            "Empty cohorts are not assessable, not successful evaluations.",
            "No model, temperature or threshold selection was performed.",
        ],
    }

    arrays = {
        "candidate_temperature_scaled_probabilities": scaled,
        "candidate_accepted": combined,
        "v2_harmonized_probabilities": v2_harmonized,
        "v2_harmonized_classes": np.asarray(v2_output_classes, dtype=str),
    }

    return report, arrays