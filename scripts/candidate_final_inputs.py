"""Verify frozen final-evaluation inputs without calculating predictions."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.candidate_predictor import (
    load_candidate_resources,
    validate_candidate_features,
)
from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_evaluation import VIEW_LIMITS, make_validation_view
from scripts.prepare_candidate_final_views import METADATA


EXPERIMENT = ROOT / "data/candidate_experiment_v1"
VIEWS = EXPERIMENT / "final_test/views"
CALIBRATION_REPORTS = ROOT / "reports/candidate_calibration"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_inputs_unchanged(context):
    for relative, expected in context["input_sha256"].items():
        if file_hash(ROOT / relative) != expected:
            raise RuntimeError(f"Final-evaluation input changed: {relative}")


def prepare_final_inputs():
    """Return verified models, views and metadata for the final runner."""
    checked = {}

    def remember(path, expected=None):
        path = Path(path).resolve()
        actual = file_hash(path)

        if expected is not None and actual != expected:
            raise ValueError(f"Fingerprint mismatch: {path}")

        if path in checked and checked[path] != actual:
            raise RuntimeError(f"Input changed during preflight: {path}")

        checked[path] = actual

    print("Checking final-view provenance...", flush=True)

    manifest_path = VIEWS / "view_manifest.json"
    remember(manifest_path)
    manifest = read_json(manifest_path)

    if manifest["status"] != "final_observation_views_prepared_no_predictions":
        raise ValueError("Unexpected final-view preparation status")
    if manifest["seed"] != 42 or set(manifest["views"]) != set(VIEW_LIMITS):
        raise ValueError("Unexpected view configuration")

    view_inputs = {
        "train_source": ROOT / "data/raw/release_train_patients",
        "validation_source": ROOT / "data/raw/release_validate_patients",
        "test_source": ROOT / "data/raw/release_test_patients",
        "dataset_manifest": ROOT / "data/candidate_v1/manifest.json",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "protocol": ROOT / "docs/CANDIDATE_FINAL_EVALUATION_PROTOCOL.md",
        "candidate_feature_builder": ROOT / "scripts/candidate_feature_builder.py",
        "source_audit_helpers": ROOT / "scripts/audit_candidate_source_records.py",
        "overlap_audit_helpers": ROOT / "scripts/audit_candidate_overlap.py",
        "policy": EXPERIMENT / "calibration/uncertainty_policy/policy.json",
        "model": EXPERIMENT / "models/rf_group_weighted/model.joblib",
        "audit_script": ROOT / "scripts/audit_candidate_final_source.py",
        "source_audit": ROOT / "reports/candidate_final_evaluation/source_audit.json",
        "dataset_builder": ROOT / "scripts/build_candidate_dataset.py",
        "evaluation_helper": ROOT / "scripts/candidate_evaluation.py",
        "previous_view_manifest": (
            ROOT / "reports/candidate_model_comparison/view_manifest.json"
        ),
        "view_preparation_script": ROOT / "scripts/prepare_candidate_final_views.py",
    }

    if set(manifest["input_sha256"]) != set(view_inputs):
        raise ValueError("Unexpected final-view dependency list")

    for name, expected in manifest["input_sha256"].items():
        remember(view_inputs[name], expected)

    expected_outputs = {
        "groups.csv",
        "zero_supported.csv",
        *[f"{view}.csv" for view in VIEW_LIMITS],
    }
    if set(manifest["output_sha256"]) != expected_outputs:
        raise ValueError("Unexpected final-view output list")

    for filename, expected in manifest["output_sha256"].items():
        remember(VIEWS / filename, expected)

    print("Checking frozen calibration and metric dependencies...", flush=True)

    probes_path = CALIBRATION_REPORTS / "inference_probes.json"
    remember(probes_path)
    probes = read_json(probes_path)

    evaluation_manifest_path = (
        CALIBRATION_REPORTS
        / "recorded_results/uncertainty_evaluation/evaluation_manifest.json"
    )
    remember(
        evaluation_manifest_path,
        probes["source_sha256"]["evaluation_manifest"],
    )
    earlier_evaluation = read_json(evaluation_manifest_path)

    temperature_path = (
        EXPERIMENT / "calibration/temperature_fit/temperature_fit.json"
    )
    remember(
        temperature_path,
        earlier_evaluation["input_sha256"]["temperature_fit"],
    )
    fitted = read_json(temperature_path)

    remember(view_inputs["policy"], earlier_evaluation["input_sha256"]["policy"])
    remember(view_inputs["model"], earlier_evaluation["input_sha256"]["model"])

    for name, filename in {
        "metrics_helper": "candidate_calibration_metrics.py",
        "classification_metrics_helper": "candidate_comparison_metrics.py",
        "calibration_helper": "candidate_probability_calibration.py",
        "uncertainty_helper": "candidate_uncertainty_policy.py",
        "selective_evaluation_helper": "candidate_selective_evaluation.py",
    }.items():
        remember(
            ROOT / "scripts" / filename,
            earlier_evaluation["input_sha256"][name],
        )

    for name, filename in {
        "predictor": "candidate_predictor.py",
        "assessment": "candidate_assessment.py",
    }.items():
        remember(
            ROOT / "inference" / filename,
            probes["source_sha256"][name],
        )

    remember(ROOT / "scripts/candidate_final_metrics.py")
    remember(Path(__file__))

    print("Inspecting frozen model metadata...", flush=True)

    load_candidate_resources.cache_clear()
    candidate, schema, policy = load_candidate_resources()

    if schema["feature_names"] != manifest["feature_names"]:
        raise ValueError("Final views use a different candidate feature order")
    if sorted(policy["classes"]) != manifest["classes"]:
        raise ValueError("Final views use a different candidate condition scope")
    if list(candidate.classes_) != policy["classes"]:
        raise ValueError("Candidate class order differs from policy")
    if fitted["classes"] != policy["classes"]:
        raise ValueError("Temperature-fit class order differs from policy")
    if fitted["feature_names"] != schema["feature_names"]:
        raise ValueError("Temperature-fit feature order differs")

    temperature = fitted["fit"]["temperature"]
    if policy["probability_representation"] == "temperature_scaled":
        if policy["temperature"] != temperature:
            raise ValueError("Frozen temperature differs from selected policy")
    elif policy["probability_representation"] == "original":
        if policy["temperature"] is not None:
            raise ValueError("Original probabilities must not apply temperature")
    else:
        raise ValueError("Unknown selected probability representation")

    comparison_path = (
        ROOT / "reports/candidate_model_comparison/comparison_manifest.json"
    )
    remember(comparison_path)
    comparison = read_json(comparison_path)

    v2_path = ROOT / "models/rf_v2_23class_prototype.joblib"
    v2_schema_path = ROOT / "models/v2_feature_schema.json"
    remember(v2_path, comparison["input_sha256"]["frozen_v2_model"])
    remember(v2_schema_path, comparison["input_sha256"]["frozen_v2_schema"])
    remember(view_inputs["model"], comparison["input_sha256"]["rf_group_weighted_model"])

    v2_schema = read_json(v2_schema_path)
    v2 = joblib.load(v2_path)

    for name, model, feature_names in (
        ("candidate", candidate, schema["feature_names"]),
        ("V2", v2, v2_schema["feature_names"]),
    ):
        if model.n_features_in_ != len(feature_names):
            raise ValueError(f"{name} feature count differs")
        if (
            hasattr(model, "feature_names_in_")
            and list(model.feature_names_in_) != feature_names
        ):
            raise ValueError(f"{name} feature order differs")

    canonical_v2 = {
        "Allergic rhinitis" if label == "Allergic sinusitis" else label
        for label in v2.classes_
    }
    if canonical_v2 != set(policy["classes"]) | {"COVID-19"}:
        raise ValueError("Unexpected V2 canonical output scope")

    if len(schema["feature_names"]) != 70 or len(v2_schema["feature_names"]) != 66:
        raise ValueError("Unexpected model feature counts")

    full = pd.read_csv(VIEWS / "full.csv")
    groups = pd.read_csv(VIEWS / "groups.csv")
    zero = pd.read_csv(VIEWS / "zero_supported.csv")

    expected_columns = METADATA + schema["feature_names"]
    if list(full.columns) != expected_columns or list(zero.columns) != expected_columns:
        raise ValueError("Unexpected final-observation columns")

    pd.testing.assert_frame_equal(
        groups, full[METADATA], check_dtype=False, check_exact=True
    )

    if not full["source_group_id"].is_unique:
        raise ValueError("Duplicate eligible source-group IDs")
    if not zero["source_group_id"].is_unique:
        raise ValueError("Duplicate zero-supported source-group IDs")
    if set(full["source_group_id"]) & set(zero["source_group_id"]):
        raise ValueError("Eligible and zero-supported groups intersect")

    if len(full) != manifest["cohort_counts"]["complete_eligible"]:
        raise ValueError("Complete eligible cohort count differs")
    if len(zero) != manifest["zero_supported_groups"]:
        raise ValueError("Zero-supported group count differs")

    if not full["evaluation_cohort"].isin(["primary", "overlap"]).all():
        raise ValueError("Unexpected eligible cohort assignment")
    if not zero["evaluation_cohort"].eq("zero_supported").all():
        raise ValueError("Unexpected zero-supported cohort assignment")

    for cohort in ("primary", "complete_eligible", "overlap"):
        selected = (
            full
            if cohort == "complete_eligible"
            else full.loc[full["evaluation_cohort"].eq(cohort)]
        )
        if len(selected) != manifest["cohort_counts"][cohort]:
            raise ValueError(f"Cohort count differs: {cohort}")

        counts = selected["canonical_label"].value_counts()
        support = {label: int(counts.get(label, 0)) for label in policy["classes"]}
        if support != manifest["cohort_support"][cohort]:
            raise ValueError(f"Cohort condition support differs: {cohort}")

    for row in zero.to_dict("records"):
        features = {name: row[name] for name in schema["feature_names"]}
        _, unit_count, absent = validate_candidate_features(features, schema)
        if unit_count != 0 or absent or row["reported_unit_count"] != 0:
            raise ValueError("Invalid zero-supported observation")

    frames = {}
    for view in VIEW_LIMITS:
        print(f"Checking saved {view} observations...", flush=True)
        details = manifest["views"][view]

        if (
            details["filename"] != f"{view}.csv"
            or details["sha256"] != manifest["output_sha256"][f"{view}.csv"]
            or details["requested_maximum_reporting_units"] != VIEW_LIMITS[view]
        ):
            raise ValueError(f"View manifest details differ: {view}")

        frame = full if view == "full" else pd.read_csv(VIEWS / f"{view}.csv")
        expected = make_validation_view(full, schema, view, seed=42)

        pd.testing.assert_frame_equal(
            frame, expected, check_dtype=False, check_exact=True
        )

        if len(frame) != details["rows"]:
            raise ValueError(f"View row count differs: {view}")

        unit_counts = {
            str(int(value)): int(count)
            for value, count in frame["reported_unit_count"].value_counts().items()
        }
        if unit_counts != details["actual_reporting_unit_counts"]:
            raise ValueError(f"Reporting-unit distribution differs: {view}")

        frames[view] = frame

    context = {
        "candidate_model": candidate,
        "v2_model": v2,
        "schema": schema,
        "v2_schema": v2_schema,
        "policy": policy,
        "temperature": temperature,
        "view_manifest": manifest,
        "frames": frames,
        "zero_supported": zero,
        "input_sha256": {
            path.relative_to(ROOT.resolve()).as_posix(): expected
            for path, expected in checked.items()
        },
    }
    assert_inputs_unchanged(context)
    return context


def main():
    context = prepare_final_inputs()

    print("\nFINAL EVALUATION PREFLIGHT PASSED")
    for name, count in context["view_manifest"]["cohort_counts"].items():
        print(f"  {name}: {count:,} groups")
    print("Candidate features:", len(context["schema"]["feature_names"]))
    print("V2 features:", len(context["v2_schema"]["feature_names"]))
    print("Candidate conditions:", len(context["policy"]["classes"]))
    print("Probability representation:", context["policy"]["probability_representation"])
    print("Frozen temperature:", context["temperature"])
    print("Zero-supported groups:", len(context["zero_supported"]))
    print("PASS: saved views match deterministic reconstruction.")
    print("PASS: frozen-input fingerprints unchanged.")
    print("No predictions calculated and no files saved.")


if __name__ == "__main__":
    main()