"""Evaluate frozen V2 and the selected candidate on official test views."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.candidate_predictor import predict_candidate
from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_evaluation import VIEW_LIMITS, project_v2_features
from scripts.candidate_final_inputs import (
    assert_inputs_unchanged,
    prepare_final_inputs,
)
from scripts.candidate_final_metrics import evaluate_final_cohorts


OUTPUT = ROOT / "data/candidate_experiment_v1/final_test/evaluation"
BATCH_SIZE = 5000


def predict_batches(model, frame):
    started = time.perf_counter()
    probabilities = np.vstack([
        model.predict_proba(frame.iloc[start:start + BATCH_SIZE])
        for start in range(0, len(frame), BATCH_SIZE)
    ])
    return probabilities, time.perf_counter() - started


def append_tables(view, report, representation, tables):
    for cohort, result in report["cohorts"].items():
        context = {"view": view, "cohort": cohort}

        tables["cohort_status.csv"].append({
            **context,
            "status": result["status"],
            "observations": result["observations"],
            "required_for_research_target": result["required_for_research_target"],
        })

        if result["status"] != "evaluated":
            continue

        candidate_reports = result["candidate_probability_metrics"]
        selected_report = candidate_reports[representation]
        v2_report = result["v2_classification"]

        for model_name, classification in (
            ("frozen_v2_harmonized", v2_report),
            ("rf_group_weighted", selected_report),
        ):
            model_context = {**context, "model": model_name}

            # Use the same classification-summary fields for both models.
            tables["classification_summary.csv"].append({
                **model_context,
                **{
                    key: classification["summary"][key]
                    for key in v2_report["summary"]
                },
            })
            tables["classification_per_condition.csv"].extend(
                {**model_context, **row}
                for row in classification["per_condition"]
            )

            names = classification["confusion_labels"]
            matrix = classification["confusion_matrix"]

            for true_index, true_label in enumerate(names):
                support = sum(matrix[true_index])
                for predicted_index, predicted_label in enumerate(names):
                    count = matrix[true_index][predicted_index]
                    tables["confusion_counts.csv"].append({
                        **model_context,
                        "actual_condition": true_label,
                        "predicted_condition": predicted_label,
                        "count": count,
                        "true_condition_total": support,
                        "normalized_by_true_condition": (
                            count / support if support else None
                        ),
                    })

        for name, probability_report in candidate_reports.items():
            probability_context = {**context, "representation": name}

            tables["candidate_probability_summary.csv"].append({
                **probability_context,
                **probability_report["summary"],
            })
            tables["candidate_probability_per_condition.csv"].extend(
                {**probability_context, **row}
                for row in probability_report["per_condition"]
            )
            tables["reliability_bins.csv"].extend(
                {**probability_context, **row}
                for row in probability_report["reliability_bins"]
            )

        policy_report = result["candidate_policy"]
        tables["policy_summary.csv"].append({
            **context, **policy_report["summary"]
        })
        tables["policy_per_condition.csv"].extend(
            {**context, **row} for row in policy_report["per_condition"]
        )
        tables["policy_by_reporting_units.csv"].extend(
            {**context, **row} for row in policy_report["by_reporting_units"]
        )


def percentage(value):
    return "undefined" if value is None else f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check inputs without calculating predictions or saving files.",
    )
    args = parser.parse_args()

    if OUTPUT.exists():
        raise FileExistsError(
            "Final evaluation directory already exists. "
            "Preserve the existing run; do not overwrite it."
        )

    runner_hash = file_hash(Path(__file__))
    context = prepare_final_inputs()
    context["input_sha256"][
        Path(__file__).resolve().relative_to(ROOT).as_posix()
    ] = runner_hash
    assert_inputs_unchanged(context)

    policy = context["policy"]
    representation = policy["probability_representation"]
    candidate_classes = list(context["candidate_model"].classes_)
    v2_classes = list(context["v2_model"].classes_)

    print("\nFINAL EVALUATION INPUTS")
    for name, count in context["view_manifest"]["cohort_counts"].items():
        print(f"  {name}: {count:,} groups")
    print("Probability representation:", representation)
    print("Frozen temperature:", context["temperature"])
    print("Policy mode:", policy["mode"])
    print("Minimum score:", policy["min_score"])
    print("Minimum margin:", policy["min_margin"])

    if args.check_only:
        print("\nPASS: preflight completed.")
        print("No predictions calculated and no files saved.")
        return

    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "predictions").mkdir()
    output_hashes = {}

    def save_json(filename, content):
        path = OUTPUT / filename
        path.write_text(
            json.dumps(
                content, indent=2, ensure_ascii=False, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )
        output_hashes[filename] = file_hash(path)

    save_json("run_started.json", {
        "issue": 50,
        "status": "started_before_final_predictions",
        "input_sha256": context["input_sha256"],
        "cohort_counts": context["view_manifest"]["cohort_counts"],
        "notes": [
            "A completed evaluation_manifest.json is required for a valid run.",
            "An interrupted run must not be silently overwritten.",
        ],
    })

    # These inputs should be blocked before model prediction.
    zero_responses = []
    for row in context["zero_supported"].to_dict("records"):
        features = {
            name: row[name] for name in context["schema"]["feature_names"]
        }
        response = predict_candidate(features)

        if (
            response["status"] != "insufficient_information"
            or response["prediction_performed"] is not False
            or response["predicted_condition"] is not None
            or response["model_score"] is not None
            or response["reported_unit_count"] != 0
            or response["policy_id"] != policy["policy_id"]
        ):
            raise AssertionError("Zero-supported input was not safely withheld")

        zero_responses.append({
            "source_group_id": row["source_group_id"],
            "source_row": int(row["source_row"]),
            "canonical_label": row["canonical_label"],
            "public_response": response,
        })

    save_json("zero_supported_responses.json", {
        "groups": len(zero_responses),
        "cases": zero_responses,
        "notes": [
            "These groups are outside eligible classifier metric denominators.",
            "This checks direct-feature handling, not full application safety.",
        ],
    })

    tables = {
        filename: []
        for filename in (
            "cohort_status.csv",
            "classification_summary.csv",
            "classification_per_condition.csv",
            "confusion_counts.csv",
            "candidate_probability_summary.csv",
            "candidate_probability_per_condition.csv",
            "reliability_bins.csv",
            "policy_summary.csv",
            "policy_per_condition.csv",
            "policy_by_reporting_units.csv",
            "inference_times.csv",
        )
    }
    results = {}

    for view in VIEW_LIMITS:
        frame = context["frames"][view]

        print(f"\nPredicting frozen V2 / {view}...", flush=True)
        v2_features = project_v2_features(
            frame, context["v2_schema"]
        ).astype(np.float32)
        v2_probabilities, v2_seconds = predict_batches(
            context["v2_model"], v2_features
        )
        del v2_features

        print(f"Predicting selected candidate / {view}...", flush=True)
        candidate_features = frame.loc[
            :, context["schema"]["feature_names"]
        ].astype(np.float32)
        candidate_probabilities, candidate_seconds = predict_batches(
            context["candidate_model"], candidate_features
        )
        del candidate_features

        for model_name, duration in (
            ("frozen_v2_harmonized", v2_seconds),
            ("rf_group_weighted", candidate_seconds),
        ):
            tables["inference_times.csv"].append({
                "model": model_name,
                "view": view,
                "observations": len(frame),
                "batch_size": BATCH_SIZE,
                "prediction_seconds": duration,
            })

        ids = frame["source_group_id"].to_numpy(dtype=str)
        actual = frame["canonical_label"].to_numpy(dtype=str)
        assignments = frame["evaluation_cohort"].to_numpy(dtype=str)
        units = frame["reported_unit_count"].to_numpy(dtype=np.int64)

        report, derived = evaluate_final_cohorts(
            view=view,
            source_group_ids=ids,
            actual=actual,
            cohort_assignments=assignments,
            reported_unit_counts=units,
            candidate_probabilities=candidate_probabilities,
            candidate_classes=candidate_classes,
            v2_probabilities=v2_probabilities,
            v2_classes=v2_classes,
            temperature=context["temperature"],
            policy=policy,
        )
        results[view] = report
        append_tables(view, report, representation, tables)

        filename = f"predictions/{view}.npz"
        path = OUTPUT / filename
        np.savez_compressed(
            path,
            source_group_ids=ids,
            actual=actual,
            cohort_assignments=assignments,
            reported_unit_count=units,
            candidate_classes=np.asarray(candidate_classes, dtype=str),
            candidate_original_probabilities=candidate_probabilities,
            v2_native_classes=np.asarray(v2_classes, dtype=str),
            v2_native_probabilities=v2_probabilities,
            **derived,
        )
        output_hashes[filename] = file_hash(path)

        primary = report["cohorts"]["primary"]
        if primary["status"] != "evaluated":
            print("  Primary cohort: not assessable.")
            continue

        v2_summary = primary["v2_classification"]["summary"]
        candidate_summary = primary[
            "candidate_probability_metrics"
        ][representation]["summary"]
        policy_summary = primary["candidate_policy"]["summary"]

        print(
            "  Primary accuracy: "
            f"V2={percentage(v2_summary['accuracy'])} | "
            f"candidate={percentage(candidate_summary['accuracy'])}"
        )
        print(
            "  Primary macro F1: "
            f"V2={percentage(v2_summary['macro_f1'])} | "
            f"candidate={percentage(candidate_summary['macro_f1'])}"
        )
        print(
            f"  Coverage={percentage(policy_summary['coverage'])} | "
            f"accepted accuracy={percentage(policy_summary['accepted_accuracy'])} | "
            f"Wilson lower={percentage(policy_summary['accepted_accuracy_wilson_lower'])}"
        )

        if view in policy["required_views"]:
            print(
                "  Research targets met:",
                policy_summary["meets_count_and_accuracy_targets"],
            )
        else:
            print("  One-unit view: all predictions must be withheld.")

    failed_views = []
    unavailable_views = []

    for view in policy["required_views"]:
        primary = results[view]["cohorts"]["primary"]
        if primary["status"] != "evaluated":
            unavailable_views.append(view)
        elif not primary["candidate_policy"]["summary"][
            "meets_count_and_accuracy_targets"
        ]:
            failed_views.append(view)

    if unavailable_views:
        assessment = "not_assessable"
        targets_met = None
    else:
        targets_met = not failed_views
        assessment = "met" if targets_met else "not_met"

    decision = {
        "issue": 50,
        "assessment_cohort": "primary",
        "required_views": policy["required_views"],
        "research_target_assessment": assessment,
        "research_targets_met_on_all_required_views": targets_met,
        "failed_required_views": failed_views,
        "not_assessable_required_views": unavailable_views,
        "model_retrained": False,
        "temperature_refitted": False,
        "thresholds_changed": False,
        "candidate_reselected": False,
        "deployment_approved": False,
        "interpretation": (
            "Official-test-split research evaluation with documented earlier "
            "project-level test exposure. Aggregate targets do not establish "
            "condition-specific adequacy or clinical performance for Kenyan users."
        ),
    }

    assert_inputs_unchanged(context)

    for filename, rows in tables.items():
        path = OUTPUT / filename
        pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
        output_hashes[filename] = file_hash(path)

    save_json("per_view_results.json", results)
    save_json("evaluation_decision.json", decision)

    assert_inputs_unchanged(context)

    manifest = {
        "issue": 50,
        "status": "frozen_official_test_evaluation_completed",
        "candidate_name": "rf_group_weighted",
        "baseline_name": "frozen_v2_harmonized",
        "settings": list(VIEW_LIMITS),
        "cohort_counts": context["view_manifest"]["cohort_counts"],
        "cohort_support": context["view_manifest"]["cohort_support"],
        "zero_supported_groups": len(zero_responses),
        "probability_representation": representation,
        "temperature": context["temperature"],
        "policy_id": policy["policy_id"],
        "input_sha256": context["input_sha256"],
        "output_sha256": output_hashes,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "feature_matrix_dtype": "float32",
        "prediction_batch_size": BATCH_SIZE,
        "notes": [
            "Both models received the same prepared observations.",
            "Primary and overlap cohorts are reported separately.",
            "Complete eligible results include both cohorts.",
            "Zero-supported groups are excluded from classifier metrics.",
            "V2 retains its COVID-19 output after allergy harmonization.",
            "Candidate thresholds were not applied to V2.",
            "Unfiltered candidate accuracy is distinct from accepted accuracy.",
            "Repeated views contain the same groups, not independent patients.",
            "No model, calibration parameter or threshold was fitted.",
            "The protocol documents prior project-level test exposure.",
            "Per-condition limitations must accompany aggregate results.",
            "The deployed application was not changed.",
        ],
    }

    (OUTPUT / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nFROZEN FINAL EVALUATION COMPLETE")
    print("Primary-cohort research target assessment:", assessment)
    print("Failed required views:", failed_views)
    print("Not-assessable required views:", unavailable_views)
    print("Zero-supported groups correctly withheld:", len(zero_responses))
    print("Input/model/policy fingerprints unchanged.")
    print("No model or thresholds changed.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()