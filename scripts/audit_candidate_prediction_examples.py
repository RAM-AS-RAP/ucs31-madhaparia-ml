"""Record unlabelled regression probes for frozen V2 and both candidates."""

import csv
import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_evaluation import (
    harmonize_probabilities,
    project_v2_features,
)
from scripts.candidate_feature_builder import get_reporting_units


EXPERIMENT = ROOT / "data/candidate_experiment_v1"
REPORTS = ROOT / "reports/candidate_model_comparison"

CANDIDATES = ["rf_group_weighted", "rf_class_balanced"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ranked_scores(scores, classes):
    pairs = sorted(
        zip(classes, scores),
        key=lambda item: (-float(item[1]), str(item[0])),
    )
    return [
        {
            "rank": rank,
            "condition": str(condition),
            "model_score": float(score),
        }
        for rank, (condition, score) in enumerate(pairs, start=1)
    ]


def main():
    json_output = REPORTS / "prediction_examples.json"
    csv_output = REPORTS / "prediction_examples_top3.csv"

    if json_output.exists() or csv_output.exists():
        raise FileExistsError("Prediction-example reports already exist")

    paths = {
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "dataset_manifest": ROOT / "data/candidate_v1/manifest.json",
        "frozen_v2_schema": ROOT / "models/v2_feature_schema.json",
        "frozen_v2_model": ROOT / "models/rf_v2_23class_prototype.joblib",
        "evaluation_helper": ROOT / "scripts/candidate_evaluation.py",
        "feature_builder": ROOT / "scripts/candidate_feature_builder.py",
        "comparison_manifest": EXPERIMENT / "comparison/comparison_manifest.json",
        "probe_script": Path(__file__).resolve(),
    }

    for name in CANDIDATES:
        folder = EXPERIMENT / "models" / name
        paths[f"{name}_model"] = folder / "model.joblib"
        paths[f"{name}_manifest"] = folder / "training_manifest.json"

    def fingerprints():
        return {name: file_hash(path) for name, path in paths.items()}

    before = fingerprints()
    comparison = read_json(paths["comparison_manifest"])

    for name, expected in comparison["input_sha256"].items():
        if name in before and before[name] != expected:
            raise ValueError(f"Comparison input changed: {name}")

    schema = read_json(paths["candidate_schema"])
    v2_schema = read_json(paths["frozen_v2_schema"])
    dataset_manifest = read_json(paths["dataset_manifest"])

    if schema != dataset_manifest["schema"]:
        raise ValueError("Candidate schema differs from dataset")

    overrides = schema["canonical_mapping_overrides"]
    cases = {
        "tiredness_headache": [
            "E_89", *overrides["headache"]["present_features"]
        ],
        "tiredness_chest_pain": [
            "E_89", *overrides["chest_pain"]["present_features"]
        ],
        "fever_cough": ["E_91", "E_201"],
    }

    concepts = schema["symptom_features"]
    rows = []
    case_metadata = []

    for case_id, positive_codes in cases.items():
        present = set(positive_codes)
        state = {
            "present": [code for code in concepts if code in present],
            "absent": [],
            "unknown": [code for code in concepts if code not in present],
        }
        units = get_reporting_units(state, schema)

        row = {"AGE": 22, "SEX_M": 0}
        row.update({
            f"{code}__present": int(code in present) for code in concepts
        })
        row.update({f"{code}__absent": 0 for code in concepts})

        rows.append(row)
        case_metadata.append({
            "case_id": case_id,
            "age": 22,
            "sex": "F",
            "positive_concepts": state["present"],
            "reporting_units": units,
            "ground_truth": None,
        })

    candidate_frame = pd.DataFrame(
        rows, columns=schema["feature_names"]
    )
    v2_frame = project_v2_features(candidate_frame, v2_schema)

    if np.array_equal(
        candidate_frame.iloc[0].to_numpy(),
        candidate_frame.iloc[1].to_numpy(),
    ):
        raise ValueError("Candidate lost the pain-region distinction")

    if not np.array_equal(
        v2_frame.iloc[0].to_numpy(),
        v2_frame.iloc[1].to_numpy(),
    ):
        raise ValueError("Unexpected change to the V2 collision")

    target_classes = sorted(
        set(dataset_manifest["label_mapping"].values()) | {"COVID-19"}
    )

    results = []
    csv_rows = []

    print("UNLABELLED PREDICTION PROBES — age 22, female")
    print("Scores are not accuracy measurements or calibrated clinical risks.")

    for name in ["frozen_v2_harmonized", *CANDIDATES]:
        if name == "frozen_v2_harmonized":
            model = joblib.load(paths["frozen_v2_model"])
            X = v2_frame
        else:
            model = joblib.load(paths[f"{name}_model"])
            X = candidate_frame

        if model.feature_names_in_.tolist() != X.columns.tolist():
            raise ValueError(f"Feature order mismatch: {name}")

        model.set_params(n_jobs=2)
        native = model.predict_proba(X)
        canonical = harmonize_probabilities(
            native, model.classes_, target_classes
        )

        if name == "frozen_v2_harmonized":
            if not np.allclose(native[0], native[1], rtol=0, atol=1e-12):
                raise ValueError("Identical V2 inputs produced different scores")

        print(f"\n{name}")

        for index, metadata in enumerate(case_metadata):
            native_ranking = ranked_scores(native[index], model.classes_)
            canonical_ranking = ranked_scores(
                canonical[index], target_classes
            )

            results.append({
                "model": name,
                **metadata,
                "encoded_model_input": X.iloc[index].to_dict(),
                "native_ranking": native_ranking,
                "canonical_ranking": canonical_ranking,
            })

            print(f"  {metadata['case_id']}:")
            for item in canonical_ranking[:3]:
                print(
                    f"    {item['rank']}. {item['condition']}: "
                    f"{100 * item['model_score']:.2f}%"
                )
                csv_rows.append({
                    "model": name,
                    "case_id": metadata["case_id"],
                    **item,
                })

        del model
        gc.collect()

    if before != fingerprints():
        raise RuntimeError("An input changed during the probes")

    report = {
        "issue": 48,
        "status": "unlabelled_regression_probes_completed",
        "input_sha256": before,
        "results": results,
        "notes": [
            "These examples have no ground-truth diagnoses.",
            "No accuracy or clinical correctness is established.",
            "Inputs assume confirmation of the source-defined concepts.",
            "Regional pain uses broad confirmed concepts, not invented locations.",
            "This is a feature-input audit, not an NLP or end-to-end test.",
            "Native and harmonized rankings are both preserved.",
            "No model, symptom mapping or deployed application was changed.",
            "Final uncertainty handling belongs to issue #49.",
        ],
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    with csv_output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["model", "case_id", "rank", "condition", "model_score"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\nPASS: candidate inputs preserve distinct pain regions.")
    print("PASS: frozen V2 retains the original input collision.")
    print("Input fingerprints unchanged.")
    print("Reports saved to:", REPORTS)


if __name__ == "__main__":
    main()