"""Verify recorded evaluation metrics against saved prediction arrays."""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_calibration_metrics import evaluate_probability_view
from scripts.candidate_probability_calibration import temperature_scale
from scripts.candidate_selective_evaluation import evaluate_policy_view


REPORTS = ROOT / "reports/candidate_calibration"
RECORDED = REPORTS / "recorded_results"
EVALUATION = RECORDED / "uncertainty_evaluation"
CALIBRATION = ROOT / "data/candidate_experiment_v1/calibration"
OUTPUT = REPORTS / "metrics_verification.json"
VIEWS = ("full", "up_to_1", "up_to_2", "up_to_3", "up_to_5")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_equal(actual, expected, location="result"):
    """Compare counts and structure exactly, with tolerance for float metrics."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"Dictionary fields differ: {location}")
        for key in expected:
            check_equal(actual[key], expected[key], f"{location}.{key}")

    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"List structure differs: {location}")
        for index, (first, second) in enumerate(zip(actual, expected)):
            check_equal(first, second, f"{location}[{index}]")

    elif isinstance(expected, float):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)
        ):
            raise AssertionError(f"Numerical metric differs: {location}")

    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f"Value differs: {location}")


def main():
    if OUTPUT.exists():
        raise FileExistsError("Metric verification report already exists.")

    checked = {}

    def remember(path, expected=None):
        path = path.resolve()
        actual = file_hash(path)
        if expected is not None and actual != expected:
            raise RuntimeError(f"Fingerprint mismatch: {path}")
        if path in checked and checked[path] != actual:
            raise RuntimeError(f"File changed during verification: {path}")
        checked[path] = actual

    print("Checking exported results, prediction archives and documentation...")

    export_path = REPORTS / "results_export_manifest.json"
    documentation_path = REPORTS / "documentation_manifest.json"

    for manifest_path in (export_path, documentation_path):
        remember(manifest_path)
        manifest = read_json(manifest_path)

        input_key = (
            "source_sha256" if manifest_path == export_path else "input_sha256"
        )
        for relative, expected in manifest[input_key].items():
            remember(ROOT / relative, expected)

        for relative, expected in manifest["output_sha256"].items():
            remember(REPORTS / relative, expected)

    evaluation_manifest = read_json(EVALUATION / "evaluation_manifest.json")

    # Check the metric implementations against their evaluation fingerprints.
    for key, filename in {
        "metrics_helper": "candidate_calibration_metrics.py",
        "classification_metrics_helper": "candidate_comparison_metrics.py",
        "calibration_helper": "candidate_probability_calibration.py",
        "uncertainty_helper": "candidate_uncertainty_policy.py",
        "selective_evaluation_helper": "candidate_selective_evaluation.py",
    }.items():
        remember(
            ROOT / "scripts" / filename,
            evaluation_manifest["input_sha256"][key],
        )

    remember(Path(__file__))

    policy_path = RECORDED / "uncertainty_policy/policy.json"
    policy = read_json(policy_path)
    fitted = read_json(RECORDED / "temperature_fit/temperature_fit.json")
    saved_results = read_json(EVALUATION / "per_view_results.json")
    decision = read_json(EVALUATION / "evaluation_decision.json")

    if file_hash(policy_path) != decision["policy_sha256"]:
        raise AssertionError("Evaluation decision references another policy")

    if set(saved_results) != set(VIEWS):
        raise AssertionError("Unexpected saved evaluation views")

    partitions_path = CALIBRATION / "partitions.csv"
    remember(
        partitions_path,
        evaluation_manifest["input_sha256"]["calibration_partitions"],
    )
    partitions = pd.read_csv(partitions_path, dtype=str)

    if not partitions["source_group_id"].is_unique:
        raise AssertionError("Duplicate source groups in partitions")

    reserved = partitions[
        partitions["partition"] == "uncertainty_evaluation"
    ].set_index("source_group_id")

    expected_count = evaluation_manifest["source_groups"]
    if len(reserved) != expected_count:
        raise AssertionError("Evaluation group count differs")

    tables = {
        "probability_summary.csv": [],
        "probability_per_condition.csv": [],
        "reliability_bins.csv": [],
        "policy_summary.csv": [],
        "per_condition.csv": [],
        "by_reporting_units.csv": [],
    }
    failed_views = []

    for view in VIEWS:
        archive_path = (
            CALIBRATION / "uncertainty_evaluation/predictions" / f"{view}.npz"
        )
        remember(
            archive_path,
            evaluation_manifest["output_sha256"][f"predictions/{view}.npz"],
        )

        with np.load(archive_path, allow_pickle=False) as archive:
            ids = archive["source_group_ids"].copy()
            actual = archive["actual"].copy()
            classes = archive["classes"].tolist()
            original = archive["original_probabilities"].copy()
            scaled = archive["temperature_scaled_probabilities"].copy()
            units = archive["reported_unit_count"].copy()
            saved_accepted = archive["accepted"].copy()

        if (
            ids.shape != (expected_count,)
            or len(set(ids.tolist())) != expected_count
            or set(ids.tolist()) != set(reserved.index)
        ):
            raise AssertionError(f"Wrong evaluation groups: {view}")

        np.testing.assert_array_equal(
            actual,
            reserved.loc[ids, "canonical_label"].to_numpy(dtype=str),
        )
        check_equal(classes, policy["classes"], f"{view}.classes")

        # Check archive provenance against the corresponding observation view.
        view_path = CALIBRATION / "views/uncertainty_evaluation" / f"{view}.csv"
        remember(
            view_path,
            evaluation_manifest["input_sha256"][f"evaluation_view_{view}"],
        )
        frame = pd.read_csv(
            view_path,
            usecols=["source_group_id", "canonical_label", "reported_unit_count"],
        )
        np.testing.assert_array_equal(ids, frame["source_group_id"].to_numpy())
        np.testing.assert_array_equal(actual, frame["canonical_label"].to_numpy())
        np.testing.assert_array_equal(units, frame["reported_unit_count"].to_numpy())

        if original.shape != (expected_count, len(classes)):
            raise AssertionError(f"Probability shape differs: {view}")

        recalculated_scaled = temperature_scale(
            original, fitted["fit"]["temperature"]
        )
        np.testing.assert_allclose(
            scaled, recalculated_scaled, rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(
            original.argmax(axis=1), scaled.argmax(axis=1)
        )

        context = {"partition": "uncertainty_evaluation", "view": view}

        for representation, probabilities in (
            ("original", original),
            ("temperature_scaled", scaled),
        ):
            report = evaluate_probability_view(actual, probabilities, classes)
            check_equal(
                report,
                saved_results[view]["probability_metrics"][representation],
                f"{view}.{representation}",
            )

            probability_context = {**context, "representation": representation}
            tables["probability_summary.csv"].append(
                {**probability_context, **report["summary"]}
            )
            tables["probability_per_condition.csv"].extend(
                {**probability_context, **row}
                for row in report["per_condition"]
            )
            tables["reliability_bins.csv"].extend(
                {**probability_context, **row}
                for row in report["reliability_bins"]
            )

        representation = policy["probability_representation"]
        if representation not in {"original", "temperature_scaled"}:
            raise AssertionError("Unknown selected probability representation")

        selected = scaled if representation == "temperature_scaled" else original
        report, accepted = evaluate_policy_view(
            actual, selected, classes, units, policy
        )

        required = view in policy["required_views"]
        report["summary"]["required_for_research_target"] = required

        if saved_accepted.dtype != np.dtype(bool):
            raise AssertionError(f"Acceptance archive is not boolean: {view}")
        np.testing.assert_array_equal(accepted, saved_accepted)
        check_equal(
            report, saved_results[view]["policy_metrics"], f"{view}.policy"
        )

        tables["policy_summary.csv"].append({**context, **report["summary"]})
        tables["per_condition.csv"].extend(
            {**context, **row} for row in report["per_condition"]
        )
        tables["by_reporting_units.csv"].extend(
            {**context, **row} for row in report["by_reporting_units"]
        )

        if required and not report["summary"]["meets_count_and_accuracy_targets"]:
            failed_views.append(view)

        print(f"PASS: {view} probabilities, acceptance decisions and metrics.")

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

    check_equal(failed_views, decision["failed_required_views"])
    check_equal(
        not failed_views,
        decision["research_targets_met_on_all_required_views"],
    )
    check_equal(False, decision["thresholds_changed"])
    check_equal(False, decision["temperature_refitted"])

    for path, expected in checked.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during verification: {path}")

    verification = {
        "issue": 49,
        "status": "evaluation_metrics_verified_against_saved_predictions",
        "partition": "uncertainty_evaluation",
        "groups_per_view": expected_count,
        "views_verified": list(VIEWS),
        "probability_summaries_verified": 10,
        "policy_summaries_verified": 5,
        "csv_rows_verified": {
            filename: len(rows) for filename, rows in tables.items()
        },
        "failed_required_views": failed_views,
        "input_sha256": {
            path.relative_to(ROOT.resolve()).as_posix(): expected
            for path, expected in sorted(checked.items())
        },
        "notes": [
            "Recomputed evaluation metrics from saved prediction arrays.",
            "Checked source-group identities, labels and reporting-unit counts.",
            "Checked temperature transformation and exact acceptance masks.",
            "Checked nested JSON results and all six evaluation CSV tables.",
            "Checked exported documentation fingerprints.",
            "Used the existing tested metric implementations.",
            "Did not rerun model inference, fit temperature or search thresholds.",
            "Development and temperature-fitting metrics were not recomputed.",
            "Final-test data was not inspected.",
        ],
    }
    OUTPUT.write_text(
        json.dumps(verification, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nPASS: recorded evaluation metrics match saved predictions.")
    print("PASS: input and documentation fingerprints unchanged.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()