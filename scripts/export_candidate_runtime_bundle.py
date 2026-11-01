"""Export the frozen candidate and verify portable inference parity."""

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from respiratory_candidate_runtime import __version__
from respiratory_candidate_runtime.errors import CandidateInputError
from respiratory_candidate_runtime.integrity import (
    decode_json_bytes,
    read_verified_bundle,
    sha256_bytes,
)
from respiratory_candidate_runtime.loader import (
    current_environment,
    load_runtime,
)
from respiratory_candidate_runtime.metadata import (
    validate_policy,
    validate_schema,
)


EXPERIMENT = "data/candidate_experiment_v1"
CALIBRATION_REPORTS = "reports/candidate_calibration"
RECORDED_CALIBRATION = f"{CALIBRATION_REPORTS}/recorded_results"
FINAL_REPORTS = "reports/candidate_final_evaluation"
RECORDED_FINAL = f"{FINAL_REPORTS}/recorded_results/evaluation"
COMPARISON_REPORTS = "reports/candidate_model_comparison"

DESTINATION = (
    ROOT / EXPERIMENT / "portable_runtime_v1"
)

SOURCES = {
    "model.joblib": (
        f"{EXPERIMENT}/models/rf_group_weighted/model.joblib"
    ),
    "feature_schema.json": "docs/candidate_feature_schema_v1.json",
    "policy.json": (
        f"{EXPERIMENT}/calibration/uncertainty_policy/policy.json"
    ),
    "reference_cases.json": (
        f"{CALIBRATION_REPORTS}/inference_probes.json"
    ),
    "provenance/training_manifest.json": (
        f"{COMPARISON_REPORTS}/rf_group_weighted_training_manifest.json"
    ),
    "provenance/model_selection.json": (
        f"{COMPARISON_REPORTS}/selection.json"
    ),
    "provenance/model_comparison.json": (
        f"{COMPARISON_REPORTS}/comparison_manifest.json"
    ),
    "provenance/temperature_fit.json": (
        f"{RECORDED_CALIBRATION}/temperature_fit/temperature_fit.json"
    ),
    "provenance/calibration_selection.json": (
        f"{RECORDED_CALIBRATION}/policy_development_comparison/"
        "selection.json"
    ),
    "provenance/policy_search.json": (
        f"{RECORDED_CALIBRATION}/uncertainty_policy/search_manifest.json"
    ),
    "provenance/uncertainty_evaluation.json": (
        f"{RECORDED_CALIBRATION}/uncertainty_evaluation/"
        "evaluation_manifest.json"
    ),
    "provenance/uncertainty_decision.json": (
        f"{RECORDED_CALIBRATION}/uncertainty_evaluation/"
        "evaluation_decision.json"
    ),
    "provenance/final_evaluation.json": (
        f"{RECORDED_FINAL}/evaluation_manifest.json"
    ),
    "provenance/final_decision.json": (
        f"{RECORDED_FINAL}/evaluation_decision.json"
    ),
    "provenance/final_verification.json": (
        f"{FINAL_REPORTS}/metrics_verification.json"
    ),
}

EXPECTED_CASE_IDS = {
    "tiredness_headache",
    "tiredness_chest_pain",
    "fever_cough",
    "tiredness_only",
    "headache_only",
    "no_confirmed_symptoms",
    "unsupported_only",
    "supported_plus_unsupported",
    "safety_only_symptom",
    "supplied_safety_flag",
    "explicit_negative_evidence",
    "contradictory_symptom",
    "region_without_parent",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def prepare_payloads(root=ROOT):
    """Check linked frozen records without loading a model."""

    root = Path(root)
    payloads = {
        name: (root / source).read_bytes()
        for name, source in SOURCES.items()
    }
    digests = {
        name: sha256_bytes(data)
        for name, data in payloads.items()
    }
    snapshot = {
        source: digests[name]
        for name, source in SOURCES.items()
    }

    def document(name):
        return decode_json_bytes(payloads[name], name)

    def check_digest(name, expected):
        require(
            digests[name] == expected,
            f"Recorded fingerprint mismatch: {name}",
        )

    schema = document("feature_schema.json")
    policy = document("policy.json")
    probes = document("reference_cases.json")
    training = document("provenance/training_manifest.json")
    selection = document("provenance/model_selection.json")
    comparison = document("provenance/model_comparison.json")
    temperature = document("provenance/temperature_fit.json")
    calibration_selection = document(
        "provenance/calibration_selection.json"
    )
    search = document("provenance/policy_search.json")
    uncertainty = document("provenance/uncertainty_evaluation.json")
    uncertainty_decision = document(
        "provenance/uncertainty_decision.json"
    )
    final = document("provenance/final_evaluation.json")
    final_decision = document("provenance/final_decision.json")
    verification = document("provenance/final_verification.json")

    require(
        verification["status"]
        == "final_metrics_verified_against_saved_predictions",
        "Completed final metric verification is required.",
    )
    require(
        final["status"] == "frozen_official_test_evaluation_completed",
        "Completed frozen final evaluation is required.",
    )

    verified_inputs = verification["input_sha256"]

    check_digest(
        "provenance/final_evaluation.json",
        verified_inputs[
            f"{EXPERIMENT}/final_test/evaluation/evaluation_manifest.json"
        ],
    )
    check_digest(
        "provenance/final_decision.json",
        verified_inputs[
            f"{EXPERIMENT}/final_test/evaluation/evaluation_decision.json"
        ],
    )
    check_digest(
        "reference_cases.json",
        verified_inputs[SOURCES["reference_cases.json"]],
    )
    check_digest(
        "provenance/final_decision.json",
        final["output_sha256"]["evaluation_decision.json"],
    )

    for name in (
        "model.joblib",
        "feature_schema.json",
        "policy.json",
        "provenance/model_comparison.json",
        "provenance/uncertainty_evaluation.json",
    ):
        check_digest(
            name,
            final["input_sha256"][SOURCES[name]],
        )

    for name, input_key in (
        ("model.joblib", "model"),
        ("feature_schema.json", "candidate_schema"),
        ("policy.json", "policy"),
        ("provenance/policy_search.json", "search_manifest"),
        ("provenance/temperature_fit.json", "temperature_fit"),
        ("provenance/calibration_selection.json", "calibration_selection"),
    ):
        check_digest(name, uncertainty["input_sha256"][input_key])

    check_digest(
        "provenance/uncertainty_decision.json",
        uncertainty["output_sha256"]["evaluation_decision.json"],
    )
    check_digest(
        "policy.json",
        search["output_sha256"]["policy.json"],
    )
    check_digest(
        "provenance/training_manifest.json",
        temperature["input_sha256"]["archived_training_manifest"],
    )
    check_digest(
        "provenance/model_selection.json",
        temperature["input_sha256"]["selection"],
    )
    check_digest(
        "provenance/model_selection.json",
        comparison["report_sha256"]["selection.json"],
    )
    check_digest("model.joblib", training["model_sha256"])
    check_digest(
        "feature_schema.json",
        training["input_sha256"]["candidate_schema"],
    )
    check_digest(
        "provenance/temperature_fit.json",
        policy["temperature_fit_sha256"],
    )
    check_digest(
        "provenance/calibration_selection.json",
        policy["calibration_selection_sha256"],
    )

    validate_schema(schema)
    validate_policy(
        policy,
        schema,
        policy["classes"],
        model_sha256=digests["model.joblib"],
        schema_sha256=digests["feature_schema.json"],
    )

    require(
        selection["selected_candidate"] == "rf_group_weighted",
        "The selected research candidate has changed.",
    )
    require(
        training["classes"] == policy["classes"]
        and training["feature_names"] == schema["feature_names"],
        "Training metadata does not match the runtime metadata.",
    )
    require(
        calibration_selection["selected_representation"]
        == policy["probability_representation"]
        == "temperature_scaled",
        "Calibration representation mismatch.",
    )
    require(
        temperature["fit"]["temperature"]
        == calibration_selection["selected_temperature"]
        == policy["temperature"]
        == final["temperature"],
        "Frozen temperatures disagree.",
    )
    require(
        uncertainty_decision["policy_sha256"] == digests["policy.json"],
        "Reserved evaluation refers to another policy.",
    )
    require(
        uncertainty_decision[
            "research_targets_met_on_all_required_views"
        ] is True,
        "Reserved uncertainty evaluation did not meet its targets.",
    )
    require(
        final_decision["research_target_assessment"] == "met",
        "Final evaluation did not meet its research targets.",
    )

    for field in (
        "model_retrained",
        "temperature_refitted",
        "thresholds_changed",
        "candidate_reselected",
        "deployment_approved",
    ):
        require(
            final_decision[field] is False,
            f"Unexpected final evaluation flag: {field}",
        )

    require(
        probes["status"] == "unlabelled_inference_probes_completed",
        "Completed inference probes are required.",
    )
    cases = probes["cases"]
    require(
        len(cases) == len(EXPECTED_CASE_IDS)
        and {case["case_id"] for case in cases} == EXPECTED_CASE_IDS,
        "Reference cases are missing or duplicated.",
    )

    for name, probe_key in (
        ("model.joblib", "candidate_model"),
        ("feature_schema.json", "candidate_schema"),
        ("policy.json", "policy"),
    ):
        check_digest(name, probes["source_sha256"][probe_key])

    # Freeze the evaluated Python dependencies without reading datasets
    # or regenerating saved final-test predictions.
    for source, expected in final["input_sha256"].items():
        if source.endswith(".py"):
            actual = sha256_bytes((root / source).read_bytes())
            require(actual == expected, f"Evaluated code changed: {source}")
            snapshot[source] = actual

    reference_sources = {
        "predictor": "inference/candidate_predictor.py",
        "assessment": "inference/candidate_assessment.py",
        "calibration_helper": "scripts/candidate_probability_calibration.py",
        "uncertainty_helper": "scripts/candidate_uncertainty_policy.py",
        "feature_builder": "scripts/candidate_feature_builder.py",
        "hash_helper": "scripts/audit_candidate_source_records.py",
    }

    for key, source in reference_sources.items():
        actual = sha256_bytes((root / source).read_bytes())
        require(
            actual == probes["source_sha256"][key],
            f"Reference inference code changed: {source}",
        )
        snapshot[source] = actual

    for path in sorted(
        (root / "respiratory_candidate_runtime").rglob("*.py")
    ):
        snapshot[path.relative_to(root).as_posix()] = sha256_bytes(
            path.read_bytes()
        )

    exporter = root / "scripts/export_candidate_runtime_bundle.py"
    snapshot[exporter.relative_to(root).as_posix()] = sha256_bytes(
        exporter.read_bytes()
    )

    expected_environment = {
        "python": training["versions"]["python"],
        "packages": {
            "numpy": training["versions"]["numpy"],
            "pandas": training["versions"]["pandas"],
            "scipy": temperature["versions"]["scipy"],
            "scikit-learn": training["versions"]["scikit_learn"],
            "joblib": training["versions"]["joblib"],
        },
    }
    require(
        current_environment() == expected_environment,
        "Active Python/dependency versions differ from the recorded "
        "training and calibration environment.",
    )

    payloads["classes.json"] = json_bytes(policy["classes"])
    return payloads, snapshot, expected_environment


def check_sources_unchanged(snapshot, root=ROOT):
    for source, expected in snapshot.items():
        require(
            sha256_bytes((Path(root) / source).read_bytes()) == expected,
            f"Source changed during export: {source}",
        )


def compare_values(actual, expected, location="response"):
    """Keep decisions exact; allow only tiny numeric score differences."""

    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and actual.keys() == expected.keys(),
            f"Different dictionary fields: {location}",
        )
        for key in expected:
            compare_values(actual[key], expected[key], f"{location}.{key}")
        return

    if isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            f"Different list structure: {location}",
        )
        for index, (first, second) in enumerate(zip(actual, expected)):
            compare_values(first, second, f"{location}[{index}]")
        return

    if type(expected) is float:
        require(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            f"Numeric parity failed: {location}",
        )
        return

    require(
        type(actual) is type(expected) and actual == expected,
        f"Exact parity failed: {location}",
    )


def verify_cases(predict, cases, input_error):
    """Check public and diagnostic calls against recorded outcomes."""

    for case in cases:
        for diagnostics in (False, True):
            features = copy.deepcopy(case["encoded_features"])
            context = copy.deepcopy(case["supplied_context"])

            if case["audit_status"] == "rejected_invalid_input":
                try:
                    predict(
                        features,
                        **context,
                        include_diagnostics=diagnostics,
                    )
                except input_error as error:
                    require(
                        str(error) == case["validation_error"],
                        f"Different validation error: {case['case_id']}",
                    )
                else:
                    raise AssertionError(
                        f"Invalid input was accepted: {case['case_id']}"
                    )
                continue

            expected = copy.deepcopy(case["public_response"])
            if diagnostics and case["research_diagnostics"] is not None:
                expected["diagnostics"] = copy.deepcopy(
                    case["research_diagnostics"]
                )

            actual = predict(
                features,
                **context,
                include_diagnostics=diagnostics,
            )
            compare_values(actual, expected, case["case_id"])

    return {
        "cases_verified": len(cases),
        "calls_verified": len(cases) * 2,
        "absolute_float_tolerance": 1e-12,
        "relative_float_tolerance": 0,
        "decisions_compared_exactly": True,
    }


def export_bundle(payloads, snapshot, environment):
    require(
        not DESTINATION.exists(),
        "Portable export already exists; refusing to overwrite it.",
    )

    # Import the original research implementation only in the exporter.
    # The portable runtime does not depend on these imports.
    from inference.candidate_predictor import (
        load_candidate_resources,
        predict_candidate,
    )

    cases = decode_json_bytes(
        payloads["reference_cases.json"],
        "reference_cases.json",
    )["cases"]

    print("Checking the frozen reference against the saved cases...")
    load_candidate_resources.cache_clear()
    reference_parity = verify_cases(
        predict_candidate,
        cases,
        ValueError,
    )

    policy = decode_json_bytes(payloads["policy.json"], "policy.json")
    schema = decode_json_bytes(
        payloads["feature_schema.json"],
        "feature_schema.json",
    )

    manifest = {
        "bundle_format_version": 1,
        "bundle_id": f"respiratory_candidate_v1-runtime-{__version__}",
        "runtime_version": __version__,
        "model_version": "candidate_v1_research",
        "schema_id": schema["schema_id"],
        "policy_id": policy["policy_id"],
        "environment": environment,
        "files": {
            name: {
                "sha256": sha256_bytes(data),
                "size_bytes": len(data),
            }
            for name, data in sorted(payloads.items())
        },
        "source_sha256": snapshot,
        "research_target_assessment": "met",
        "deployment_approved": False,
        "limitations": [
            "Evaluated on simulated respiratory-condition data.",
            "Scores are not validated individual clinical risks.",
            "Historical test exposure cannot be ruled out.",
            "This export does not activate any application.",
            "Wheel packaging and isolated installation remain pending.",
        ],
    }
    manifest_data = json_bytes(manifest)
    manifest_digest = sha256_bytes(manifest_data)

    # The temporary directory is confined to the ignored experiment
    # directory. Only a fully checked export receives the final name.
    with tempfile.TemporaryDirectory(
        prefix=".runtime-export-",
        dir=DESTINATION.parent,
    ) as temporary:
        staged = Path(temporary) / "completed_export"
        bundle = staged / "bundle"
        bundle.mkdir(parents=True)

        for name, data in payloads.items():
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        (bundle / "manifest.json").write_bytes(manifest_data)

        print("Loading the verified portable model...")
        runtime = load_runtime(
            bundle,
            expected_manifest_sha256=manifest_digest,
        )
        print("Checking portable public and diagnostic responses...")
        portable_parity = verify_cases(
            runtime.predict,
            cases,
            CandidateInputError,
        )

        read_verified_bundle(bundle, manifest_digest)
        check_sources_unchanged(snapshot)

        export_record = {
            "issue": 58,
            "status": "data_bundle_verified_wheel_packaging_pending",
            "bundle_id": manifest["bundle_id"],
            "manifest_sha256": manifest_digest,
            "runtime_version": __version__,
            "environment": environment,
            "reference_parity": reference_parity,
            "portable_parity": portable_parity,
            "source_sha256": snapshot,
            "source_fingerprints_unchanged": True,
            "model_retrained": False,
            "temperature_refitted": False,
            "thresholds_changed": False,
            "deployment_approved": False,
        }
        (staged / "export_record.json").write_bytes(
            json_bytes(export_record)
        )

        require(
            not DESTINATION.exists(),
            "Destination appeared during export; refusing to overwrite.",
        )
        staged.rename(DESTINATION)

    return manifest_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check sources and metadata without predictions or writes.",
    )
    arguments = parser.parse_args()

    print("Checking frozen artifacts, provenance and runtime environment...")
    payloads, snapshot, environment = prepare_payloads()
    check_sources_unchanged(snapshot)

    print(f"Payload files: {len(payloads)}")
    print("Reference cases: 13")
    print("Features: 70")
    print("Conditions: 21")
    print(f"Runtime version: {__version__}")
    print("PASS: linked training, calibration and evaluation fingerprints.")

    if arguments.check_only:
        print("\nPASS: export preflight completed.")
        print("No model loaded, predictions calculated or files saved.")
        return

    digest = export_bundle(payloads, snapshot, environment)

    print("\nPORTABLE DATA BUNDLE EXPORTED")
    print("PASS: frozen-reference and portable-runtime parity.")
    print("PASS: source fingerprints unchanged.")
    print(f"Manifest SHA-256: {digest}")
    print(f"Saved to: {DESTINATION}")
    print("Wheel packaging and isolated installation remain pending.")


if __name__ == "__main__":
    main()