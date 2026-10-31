"""Verify final-test results using saved predictions, without model inference."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_evaluation import VIEW_LIMITS
from scripts.candidate_final_metrics import evaluate_final_cohorts
from scripts.evaluate_candidate_final_test import append_tables
from scripts.verify_candidate_calibration import check_equal


BASE = ROOT / "data/candidate_experiment_v1"
EVALUATION = BASE / "final_test/evaluation"
VIEWS = BASE / "final_test/views"
OUTPUT = ROOT / "reports/candidate_final_evaluation/metrics_verification.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if OUTPUT.exists():
        raise FileExistsError("Final metric verification already exists.")

    checked = {}

    def remember(path, expected=None):
        path = path.resolve()
        actual = file_hash(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Fingerprint mismatch: {path}")
        if path in checked and checked[path] != actual:
            raise RuntimeError(f"Input changed during verification: {path}")
        checked[path] = actual

    print("Checking final-evaluation inputs and outputs...", flush=True)

    manifest_path = EVALUATION / "evaluation_manifest.json"
    remember(manifest_path)
    manifest = read_json(manifest_path)

    if manifest["status"] != "frozen_official_test_evaluation_completed":
        raise ValueError("Final evaluation is incomplete")
    if manifest["settings"] != list(VIEW_LIMITS):
        raise ValueError("Unexpected evaluated views")

    for relative, expected in manifest["input_sha256"].items():
        remember(ROOT / relative, expected)
    for relative, expected in manifest["output_sha256"].items():
        remember(EVALUATION / relative, expected)

    remember(Path(__file__))
    remember(ROOT / "scripts/verify_candidate_calibration.py")

    policy_path = BASE / "calibration/uncertainty_policy/policy.json"
    fit_path = BASE / "calibration/temperature_fit/temperature_fit.json"
    remember(policy_path)
    remember(fit_path)
    policy = read_json(policy_path)
    fitted = read_json(fit_path)

    check_equal(
        manifest["probability_representation"],
        policy["probability_representation"],
    )
    check_equal(manifest["temperature"], fitted["fit"]["temperature"])
    check_equal(manifest["policy_id"], policy["policy_id"])

    saved_results = read_json(EVALUATION / "per_view_results.json")
    decision = read_json(EVALUATION / "evaluation_decision.json")
    if set(saved_results) != set(VIEW_LIMITS):
        raise ValueError("Saved results contain unexpected views")

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
        )
    }
    failed = []
    unavailable = []
    reference_v2_classes = None

    for view in VIEW_LIMITS:
        frame = pd.read_csv(VIEWS / f"{view}.csv")

        with np.load(
            EVALUATION / "predictions" / f"{view}.npz",
            allow_pickle=False,
        ) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}

        for archive_name, column in (
            ("source_group_ids", "source_group_id"),
            ("actual", "canonical_label"),
            ("cohort_assignments", "evaluation_cohort"),
            ("reported_unit_count", "reported_unit_count"),
        ):
            np.testing.assert_array_equal(
                arrays[archive_name], frame[column].to_numpy()
            )

        if len(frame) != manifest["cohort_counts"]["complete_eligible"]:
            raise AssertionError(f"Observation count differs: {view}")

        classes = arrays["candidate_classes"].tolist()
        v2_classes = arrays["v2_native_classes"].tolist()
        check_equal(classes, policy["classes"])

        if reference_v2_classes is None:
            reference_v2_classes = v2_classes
        else:
            check_equal(v2_classes, reference_v2_classes)

        report, derived = evaluate_final_cohorts(
            view=view,
            source_group_ids=arrays["source_group_ids"],
            actual=arrays["actual"],
            cohort_assignments=arrays["cohort_assignments"],
            reported_unit_counts=arrays["reported_unit_count"],
            candidate_probabilities=arrays["candidate_original_probabilities"],
            candidate_classes=classes,
            v2_probabilities=arrays["v2_native_probabilities"],
            v2_classes=v2_classes,
            temperature=fitted["fit"]["temperature"],
            policy=policy,
        )

        check_equal(report, saved_results[view], view)

        for name in (
            "candidate_temperature_scaled_probabilities",
            "v2_harmonized_probabilities",
        ):
            np.testing.assert_allclose(
                derived[name], arrays[name], rtol=0, atol=1e-12
            )

        if arrays["candidate_accepted"].dtype != np.dtype(bool):
            raise AssertionError("Acceptance mask must be boolean")

        for name in ("candidate_accepted", "v2_harmonized_classes"):
            np.testing.assert_array_equal(derived[name], arrays[name])

        for cohort, result in report["cohorts"].items():
            check_equal(
                result["observations"],
                manifest["cohort_counts"][cohort],
                f"{view}.{cohort}.observations",
            )

        append_tables(
            view, report, policy["probability_representation"], tables
        )

        if view in policy["required_views"]:
            primary = report["cohorts"]["primary"]
            if primary["status"] != "evaluated":
                unavailable.append(view)
            elif not primary["candidate_policy"]["summary"][
                "meets_count_and_accuracy_targets"
            ]:
                failed.append(view)

        print(f"PASS: {view}, all cohorts and saved acceptance decisions.")

    for filename, rows in tables.items():
        expected = pd.DataFrame(rows)
        expected = expected.where(pd.notna(expected), np.nan)
        recorded = pd.read_csv(EVALUATION / filename)

        pd.testing.assert_frame_equal(
            recorded,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-10,
        )
        print(f"PASS: {filename} ({len(rows)} rows).")

    expected_assessment = (
        "not_assessable" if unavailable else "not_met" if failed else "met"
    )
    check_equal(decision["required_views"], policy["required_views"])
    check_equal(decision["assessment_cohort"], "primary")
    check_equal(decision["failed_required_views"], failed)
    check_equal(decision["not_assessable_required_views"], unavailable)
    check_equal(decision["research_target_assessment"], expected_assessment)
    check_equal(
        decision["research_targets_met_on_all_required_views"],
        None if unavailable else not failed,
    )

    for name in (
        "model_retrained",
        "temperature_refitted",
        "thresholds_changed",
        "candidate_reselected",
        "deployment_approved",
    ):
        check_equal(decision[name], False, name)

    zero = pd.read_csv(VIEWS / "zero_supported.csv")
    zero_report = read_json(EVALUATION / "zero_supported_responses.json")
    check_equal(zero_report["groups"], len(zero))
    check_equal(len(zero_report["cases"]), len(zero))
    check_equal(len(zero), manifest["zero_supported_groups"])

    if len({case["source_group_id"] for case in zero_report["cases"]}) != len(zero):
        raise AssertionError("Duplicate zero-supported responses")

    zero_by_id = zero.set_index("source_group_id")
    if {case["source_group_id"] for case in zero_report["cases"]} != set(zero_by_id.index):
        raise AssertionError("Zero-supported response groups differ")

    for case in zero_report["cases"]:
        row = zero_by_id.loc[case["source_group_id"]]
        response = case["public_response"]
        check_equal(case["canonical_label"], row["canonical_label"])
        check_equal(case["source_row"], int(row["source_row"]))
        check_equal(response["status"], "insufficient_information")
        check_equal(response["prediction_performed"], False)
        check_equal(response["predicted_condition"], None)
        check_equal(response["model_score"], None)
        check_equal(response["reported_unit_count"], 0)
        check_equal(response["policy_id"], policy["policy_id"])

    for path, expected in checked.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during verification: {path}")

    verification = {
        "issue": 50,
        "status": "final_metrics_verified_against_saved_predictions",
        "views_verified": list(VIEW_LIMITS),
        "cohort_counts": manifest["cohort_counts"],
        "csv_rows_verified": {
            filename: len(rows) for filename, rows in tables.items()
        },
        "research_target_assessment": expected_assessment,
        "zero_supported_responses_checked": len(zero),
        "input_sha256": {
            path.relative_to(ROOT.resolve()).as_posix(): expected
            for path, expected in sorted(checked.items())
        },
        "notes": [
            "Metrics were recomputed from saved native probability arrays.",
            "Source IDs, labels, cohorts and unit counts match saved views.",
            "V2 harmonization and candidate temperature scaling were checked.",
            "Acceptance masks and structured result reports were checked.",
            "All ten metric CSV tables were checked.",
            "Recorded zero-supported responses were checked without rerunning inference.",
            "Inference-time measurements were fingerprint-checked, not remeasured.",
            "Existing tested metric and table-building helpers were reused.",
            "No models were loaded for prediction and no parameters were fitted.",
            "This is a consistency verification, not independent clinical validation.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(verification, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nPASS: final results match saved predictions.")
    print("PASS: zero-supported responses and frozen decision checked.")
    print("Input fingerprints unchanged.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()