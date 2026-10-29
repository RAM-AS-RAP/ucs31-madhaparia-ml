"""Audit the ML repository's text-to-V2 prediction path.

Confirmation is simulated: only present patient symptoms
are selected. This does not test the frontend or backend.

Examples are unlabelled behaviour probes, not accuracy tests.
"""

import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.v2_predictor import load_v2_resources
from nlp.symptom_mapper import build_confirmation_output
from scripts.audit_v2_baseline import (
    SOURCE_FILES,
    audit_case,
    file_hash,
    git_commit,
)


OUTPUT_PATH = ROOT / "reports/v2_baseline_audit/nlp_baseline.json"

TEXT_CASES = [
    ("tiredness_headache", "I have tiredness and headache"),
    ("fever_cough", "I have fever and cough"),
    ("mixed_language", "I have fever na kikohozi"),
    ("spelling_errors", "I have a couhg and feaver"),
    ("negated_cough", "I have fever but no cough"),
    ("duration", "I have a cough for three weeks"),
    ("sneezing_only", "I am sneezing"),
    ("unsupported_symptom", "I have a skin rash"),
    ("other_person", "My sister has a cough"),
    ("positive_red_flag", "I am coughing up blood"),
    ("negated_red_flag", "I have fever but no coughing up blood"),
    ("unrelated_text", "The weather is nice today"),
]

AUDIT_FILES = list(dict.fromkeys(
    SOURCE_FILES + [
        "nlp/symptom_mapper.py",
        "nlp/english_aliases.json",
        "nlp/swahili_aliases.json",
        "scripts/audit_v2_nlp_baseline.py",
    ]
))


def main():
    hashes_before = {
        name: file_hash(ROOT / name)
        for name in AUDIT_FILES
    }

    schema = json.loads(
        (ROOT / "models/v2_feature_schema.json").read_text(
            encoding="utf-8"
        )
    )

    canonical = {
        entry["code"]: entry
        for entry in json.loads(
            (ROOT / "nlp/canonical_symptoms.json").read_text(
                encoding="utf-8"
            )
        )
    }

    model, feature_names = load_v2_resources()
    results = []

    for case_id, text in TEXT_CASES:
        mapping = build_confirmation_output(text)

        confirmed_codes = sorted({
            symptom["code"]
            for symptom in mapping["recognised_symptoms"]
            if symptom["status"] == "present"
            and symptom["reported_subject"] != "other_person"
        })

        # Honour the mapper's safety flag before model inference.
        # This includes uncertain red flags requiring review.
        if mapping["has_red_flag"]:
            prediction_audit = {
                "case_id": case_id,
                "age": 22,
                "sex": "F",
                "confirmed_codes": confirmed_codes,
                "audit_status": "skipped_red_flag",
                "ground_truth_condition": None,
            }
        else:
            prediction_audit = audit_case(
                case_id,
                confirmed_codes,
                canonical,
                schema,
                model,
                feature_names,
            )

        results.append({
            "case_id": case_id,
            "input_text": text,
            "nlp_output": mapping,
            "simulated_confirmed_codes": confirmed_codes,
            "prediction_audit": prediction_audit,
        })

    hashes_after = {
        name: file_hash(ROOT / name)
        for name in AUDIT_FILES
    }

    if hashes_before != hashes_after:
        raise RuntimeError("Source files changed during the audit.")

    report = {
        "audit_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "source_sha256": hashes_before,
        "python_version": sys.version,
        "package_versions": {
            package: version(package)
            for package in [
                "scikit-learn",
                "numpy",
                "pandas",
                "scipy",
                "joblib",
                "rapidfuzz",
            ]
        },
        "confirmation_policy": (
            "Select only present symptoms attributed to the patient. "
            "Respect the NLP red-flag result before inference."
        ),
        "limitations": [
            "These examples have no verified disease labels.",
            "This tests the ML mapper, not the deployed backend mapper.",
            "User confirmation is simulated.",
            "Negated symptoms are recorded but not encoded as absent "
            "in this positive-confirmation baseline.",
            "Duration is recorded by NLP but is not a V2 model input.",
            "Skip statuses belong to the audit harness.",
        ],
        "cases": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print("\nML NLP BASELINE AUDIT — age 22, female")

    for result in results:
        print("\n" + result["case_id"])
        print("  Text:", result["input_text"])

        for symptom in result["nlp_output"]["recognised_symptoms"]:
            print(
                f"  Recognised: {symptom['code']}"
                f" | status={symptom['status']}"
                f" | subject={symptom['reported_subject']}"
                f" | duration_days={symptom.get('duration_min_days')}"
            )

        print("  Confirmed:", result["simulated_confirmed_codes"])
        print(
            "  Unmatched:",
            result["nlp_output"]["unmatched_wording"],
        )

        prediction = result["prediction_audit"]
        print("  Status:", prediction["audit_status"])

        if prediction["audit_status"] == "prediction_recorded":
            leading = prediction["ranked_conditions"][0]
            print(
                f"  Leading: {leading['condition']} "
                f"({leading['model_score_percent']:.2f}%)"
            )
            print(
                "  Uncertain:",
                prediction["assessment"]["uncertain"],
            )
            print("  Dropped:", prediction["dropped_features"])

    print("\nSource/model fingerprints unchanged.")
    print("Report saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()