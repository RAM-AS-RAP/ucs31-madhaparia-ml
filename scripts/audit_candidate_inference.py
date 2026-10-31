"""Audit unlabelled examples through the real candidate inference interface."""

import csv
import json
import sys
from pathlib import Path
import math


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.candidate_predictor import (
    load_candidate_resources,
    predict_candidate,
)
from scripts.audit_candidate_source_records import file_hash


EXPERIMENT = ROOT / "data/candidate_experiment_v1"
CALIBRATION = EXPERIMENT / "calibration"
REPORTS = ROOT / "reports/candidate_calibration"


def main():
    json_output = REPORTS / "inference_probes.json"
    csv_output = REPORTS / "inference_probes.csv"

    if json_output.exists() or csv_output.exists():
        raise FileExistsError("Inference-probe reports already exist")

    paths = {
        "candidate_model": EXPERIMENT / "models/rf_group_weighted/model.joblib",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "policy": CALIBRATION / "uncertainty_policy/policy.json",
        "search_manifest": (
            CALIBRATION / "uncertainty_policy/search_manifest.json"
        ),
        "evaluation_manifest": (
            CALIBRATION / "uncertainty_evaluation/evaluation_manifest.json"
        ),
        "evaluation_decision": (
            CALIBRATION / "uncertainty_evaluation/evaluation_decision.json"
        ),
        "view_manifest": CALIBRATION / "views/view_manifest.json",
        "predictor": ROOT / "inference/candidate_predictor.py",
        "assessment": ROOT / "inference/candidate_assessment.py",
        "calibration_helper": (
            ROOT / "scripts/candidate_probability_calibration.py"
        ),
        "uncertainty_helper": ROOT / "scripts/candidate_uncertainty_policy.py",
        "feature_builder": ROOT / "scripts/candidate_feature_builder.py",
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
        "frozen_v2_model": ROOT / "models/rf_v2_23class_prototype.joblib",
        "frozen_v2_schema": ROOT / "models/v2_feature_schema.json",
        "audit_script": Path(__file__).resolve(),
    }

    def fingerprints():
        return {name: file_hash(path) for name, path in paths.items()}

    before = fingerprints()

    load_candidate_resources.cache_clear()
    _, schema, policy = load_candidate_resources()

    overrides = schema["canonical_mapping_overrides"]
    head = overrides["headache"]["present_features"]
    chest = overrides["chest_pain"]["present_features"]

    cases = [
        {
            "case_id": "tiredness_headache",
            "present": ["E_89", *head],
        },
        {
            "case_id": "tiredness_chest_pain",
            "present": ["E_89", *chest],
        },
        {
            "case_id": "fever_cough",
            "present": ["E_91", "E_201"],
        },
        {
            "case_id": "tiredness_only",
            "present": ["E_89"],
            "expected_status": "insufficient_information",
        },
        {
            "case_id": "headache_only",
            "present": head,
            "expected_status": "insufficient_information",
        },
        {
            "case_id": "no_confirmed_symptoms",
            "present": [],
            "expected_status": "insufficient_information",
        },
        {
            "case_id": "unsupported_only",
            "present": [],
            "unsupported": ["sneezing"],
            "expected_status": "unsupported_input",
        },
        {
            "case_id": "supported_plus_unsupported",
            "present": ["E_91", "E_201"],
            "unsupported": ["sneezing"],
            "expected_status": "unsupported_input",
        },
        {
            "case_id": "safety_only_symptom",
            "present": [],
            "unsupported": ["blue_lips"],
            "expected_status": "safety_review_required",
        },
        {
            "case_id": "supplied_safety_flag",
            "present": ["E_91", "E_201"],
            "safety_flags": ["confusion"],
            "expected_status": "safety_review_required",
        },
        {
            "case_id": "explicit_negative_evidence",
            "present": ["E_91", "E_201"],
            "absent": ["E_89"],
            "expected_status": "unsupported_input",
        },
        {
            "case_id": "contradictory_symptom",
            "present": ["E_91", "E_201"],
            "absent": ["E_91"],
            "expected_error": "Conflicting",
        },
        {
            "case_id": "region_without_parent",
            "present": ["E_89", "head_region_pain"],
            "expected_error": "requires present parent",
        },
    ]

    records = []
    csv_rows = []
    encoded_examples = {}

    print("UNLABELLED CANDIDATE INFERENCE PROBES — age 22, female")
    print("Direct feature inputs; no NLP or complete application safety assessment.")
    print("These examples do not measure diagnostic accuracy.")

    for case in cases:
        case_id = case["case_id"]
        present = set(case["present"])
        absent = set(case.get("absent", []))

        supported = set(schema["symptom_features"])
        if not (present | absent).issubset(supported):
            raise ValueError(f"Unknown concept in audit definition: {case_id}")

        features = {"AGE": 22, "SEX_M": 0}
        for concept in schema["symptom_features"]:
            features[f"{concept}__present"] = int(concept in present)
            features[f"{concept}__absent"] = int(concept in absent)

        encoded_examples[case_id] = features.copy()

        arguments = {
            "safety_flags": case.get("safety_flags", []),
            "unsupported_symptoms": case.get("unsupported", []),
        }

        public = None
        diagnostics = None
        error = None

        try:
            public = predict_candidate(features, **arguments)
        except ValueError as caught:
            expected_error = case.get("expected_error")
            if expected_error is None or expected_error not in str(caught):
                raise
            error = str(caught)
        else:
            if "expected_error" in case:
                raise AssertionError(f"Invalid input was accepted: {case_id}")

            if "diagnostics" in public:
                raise AssertionError("Default response exposed diagnostic rankings")

            audited = predict_candidate(
                features,
                **arguments,
                include_diagnostics=True,
            )
            diagnostics = audited.pop("diagnostics", None)

            public_fields = public.copy()
            audited_fields = audited.copy()

            public_score = public_fields.pop("model_score")
            audited_score = audited_fields.pop("model_score")

            if public_fields != audited_fields:
                raise AssertionError(
                    f"Diagnostics changed public response fields: {case_id}"
                )

            if public_score is None or audited_score is None:
                scores_match = public_score is None and audited_score is None
            else:
                scores_match = (
                    type(public_score) is float
                    and type(audited_score) is float
                    and math.isclose(
                        public_score,
                        audited_score,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )

            if not scores_match:
                raise AssertionError(
                    f"Diagnostics changed the public score: {case_id}"
                )

            expected_status = case.get("expected_status")
            if expected_status is not None and public["status"] != expected_status:
                raise AssertionError(f"Unexpected input-handling status: {case_id}")

            if public["status"] != "prediction_available":
                if (
                    public["predicted_condition"] is not None
                    or public["model_score"] is not None
                ):
                    raise AssertionError("Withheld response exposed a conclusion")

            if public["safety_flags"]:
                if (
                    public["status"] != "safety_review_required"
                    or public["prediction_performed"]
                ):
                    raise AssertionError("Safety flag did not block prediction")

            if expected_status is not None and public["prediction_performed"]:
                raise AssertionError(f"Blocked input reached prediction: {case_id}")

        status = (
            "rejected_invalid_input" if error else public["status"]
        )

        diagnostic_leader = (
            diagnostics["ranked_conditions"][0]
            if diagnostics is not None
            else None
        )

        print(f"\n{case_id}")
        print("  Status:", status)

        if error:
            print("  Validation error:", error)
        else:
            print("  Reporting units:", public["reported_unit_count"])
            print("  Public condition:", public["predicted_condition"])
            print("  Public score:", public["model_score"])
            print("  Reasons:", public["uncertainty_reasons"])

            if diagnostic_leader:
                print(
                    "  Research diagnostic leader:",
                    diagnostic_leader["condition"],
                    f"{100 * diagnostic_leader['assessed_model_score']:.2f}%",
                )

        records.append({
            "case_id": case_id,
            "age": 22,
            "sex": "F",
            "ground_truth": None,
            "encoded_features": features,
            "supplied_context": arguments,
            "audit_status": status,
            "public_response": public,
            "research_diagnostics": diagnostics,
            "validation_error": error,
        })

        csv_rows.append({
            "case_id": case_id,
            "status": status,
            "reported_unit_count": (
                public["reported_unit_count"] if public else None
            ),
            "predicted_condition": (
                public["predicted_condition"] if public else None
            ),
            "public_model_score": (
                public["model_score"] if public else None
            ),
            "prediction_performed": (
                public["prediction_performed"] if public else False
            ),
            "safety_flags": (
                "|".join(public["safety_flags"]) if public else ""
            ),
            "research_diagnostic_leader": (
                diagnostic_leader["condition"] if diagnostic_leader else None
            ),
            "research_diagnostic_score": (
                diagnostic_leader["assessed_model_score"]
                if diagnostic_leader else None
            ),
            "validation_error": error,
        })

    if (
        encoded_examples["tiredness_headache"]
        == encoded_examples["tiredness_chest_pain"]
    ):
        raise AssertionError("Head and chest inputs collided")

    if before != fingerprints():
        raise RuntimeError("An input or model changed during the audit")

    report = {
        "issue": 49,
        "status": "unlabelled_inference_probes_completed",
        "policy_id": policy["policy_id"],
        "source_sha256": before,
        "cases": records,
        "notes": [
            "Original examples have no ground-truth diagnoses.",
            "No preferred disease or acceptance outcome was imposed on them.",
            "Thresholds and calibration were not changed.",
            "Default responses do not contain diagnostic rankings.",
            "Safety examples test propagation of supplied flags.",
            "These are direct confirmed-feature probes, not NLP tests.",
            "Full application safety checks remain a separate responsibility.",
            "The existing V2 model was not modified.",
        ],
    }

    REPORTS.mkdir(parents=True, exist_ok=True)

    json_output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with csv_output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=list(csv_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\nPASS: input handling and public-response checks.")
    print("PASS: head-region and chest-region inputs remain distinct.")
    print("Source/model/policy fingerprints unchanged.")
    print("Reports saved to:", REPORTS)


if __name__ == "__main__":
    main()