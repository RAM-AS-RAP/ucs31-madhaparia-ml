"""Record V2 behaviour for manually confirmed symptom examples.

These examples have no verified disease labels.
Results describe model behaviour, not diagnostic accuracy.

This script does not modify model artifacts or training data.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from inference.backend_adapter import build_v2_backend_features
from inference.prediction_confidence import assess_prediction
from inference.v2_predictor import load_v2_resources


OUTPUT_PATH = ROOT / "reports/v2_baseline_audit/baseline.json"

CASES = [
    ("tiredness_headache", ["tiredness", "headache"]),
    ("fever_cough", ["fever", "cough"]),
    ("tiredness_chest_pain", ["tiredness", "chest_pain"]),
    ("tiredness_only", ["tiredness"]),
    ("headache_only", ["headache"]),
    ("fever_only", ["fever"]),
    ("cough_only", ["cough"]),
    ("sneezing_only", ["sneezing"]),
    ("no_confirmed_symptoms", []),
]

SOURCE_FILES = [
    "models/rf_v2_23class_prototype.joblib",
    "models/v2_feature_schema.json",
    "nlp/canonical_symptoms.json",
    "inference/backend_adapter.py",
    "inference/v2_predictor.py",
    "inference/prediction_confidence.py",
    "scripts/v2_feature_encoder.py",
    "scripts/audit_v2_baseline.py",
]


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def audit_case(case_id, codes, canonical, schema, model, feature_names):
    symptoms = [canonical[code] for code in codes]

    mapped_features = sorted({
        feature
        for symptom in symptoms
        for feature in symptom["features"]
    })

    supported = set(schema["symptom_features"])
    retained = sorted(set(mapped_features) & supported)
    dropped = sorted(set(mapped_features) - supported)

    result = {
        "case_id": case_id,
        "age": 22,
        "sex": "F",
        "input_mode": "manually_confirmed_canonical_symptoms",
        "confirmed_codes": codes,
        "canonical_symptoms": symptoms,
        "mapped_features": mapped_features,
        "retained_v2_features": retained,
        "dropped_features": dropped,
        "ground_truth_condition": None,
    }

    # This is an audit guard, not a replacement safety service.
    if any(symptom["is_red_flag"] for symptom in symptoms):
        result["audit_status"] = "skipped_red_flag"
        return result

    if not retained:
        result["audit_status"] = "skipped_no_supported_features"
        return result

    # Explicit absences are not supplied in these examples.
    # All other supported symptoms remain unknown.
    feature_values = build_v2_backend_features(
        age=22,
        sex="F",
        confirmed_features=retained,
        explicitly_absent_features=[],
    )

    frame = pd.DataFrame(
        [feature_values],
        columns=feature_names,
    )
    scores = model.predict_proba(frame)[0]

    assessment = assess_prediction(
        classes=model.classes_,
        probabilities=scores,
    )

    ranking = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index),
    )

    result.update({
        "audit_status": "prediction_recorded",
        "encoded_features": feature_values,
        "assessment": assessment,
        "ranked_conditions": [
            {
                "rank": rank,
                "condition": str(model.classes_[index]),
                "model_score": float(scores[index]),
                "model_score_percent": float(scores[index]) * 100,
            }
            for rank, index in enumerate(ranking, start=1)
        ],
    })

    return result


def main():
    hashes_before = {
        name: file_hash(ROOT / name)
        for name in SOURCE_FILES
    }

    schema = json.loads(
        (ROOT / "models/v2_feature_schema.json").read_text(
            encoding="utf-8"
        )
    )
    canonical_entries = json.loads(
        (ROOT / "nlp/canonical_symptoms.json").read_text(
            encoding="utf-8"
        )
    )
    canonical = {
        entry["code"]: entry
        for entry in canonical_entries
    }

    model, feature_names = load_v2_resources()

    results = [
        audit_case(
            case_id,
            codes,
            canonical,
            schema,
            model,
            feature_names,
        )
        for case_id, codes in CASES
    ]

    hashes_after = {
        name: file_hash(ROOT / name)
        for name in SOURCE_FILES
    }
    if hashes_before != hashes_after:
        raise RuntimeError("Source files changed during the audit.")

    report = {
        "audit_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "source_sha256": hashes_before,
        "python_version": platform.python_version(),
        "package_versions": {
            package: version(package)
            for package in [
                "scikit-learn",
                "numpy",
                "pandas",
                "scipy",
                "joblib",
            ]
        },
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "class_count": len(model.classes_),
        "schema": schema,
        "limitations": [
            "Cases are unlabelled behaviour probes, not accuracy tests.",
            "Canonical symptoms are supplied manually; NLP is not tested.",
            "Unsupported features are recorded and excluded explicitly.",
            "Scores are uncalibrated model outputs.",
            "Skip statuses are audit decisions, not new model behaviour.",
        ],
        "cases": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print("\nV2 BASELINE AUDIT — age 22, female")
    print("Unlabelled examples: these are not accuracy measurements.\n")

    for result in results:
        print(result["case_id"])
        print("  Retained:", result["retained_v2_features"])
        print("  Dropped:", result["dropped_features"])

        if result["audit_status"] != "prediction_recorded":
            print("  Status:", result["audit_status"])
            continue

        for prediction in result["ranked_conditions"][:3]:
            print(
                f"  {prediction['rank']}. "
                f"{prediction['condition']}: "
                f"{prediction['model_score_percent']:.2f}%"
            )

        print("  Uncertain:", result["assessment"]["uncertain"])

    print("\nSource/model fingerprints unchanged.")
    print("Report saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()