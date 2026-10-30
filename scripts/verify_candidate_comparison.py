"""Verify documentation hashes and recompute summaries from saved predictions."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_comparison_metrics import score_predictions
from scripts.candidate_evaluation import harmonize_probabilities


EXPERIMENT = ROOT / "data/candidate_experiment_v1"
SOURCE = EXPERIMENT / "comparison"
REPORTS = ROOT / "reports/candidate_model_comparison"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    comparison = read_json(SOURCE / "comparison_manifest.json")
    documentation = read_json(REPORTS / "documentation_manifest.json")

    assert file_hash(SOURCE / "comparison_manifest.json") == (
        documentation["comparison_manifest_sha256"]
    )

    for filename, expected in documentation["output_sha256"].items():
        assert file_hash(REPORTS / filename) == expected, filename

    for filename, expected in comparison["report_sha256"].items():
        assert file_hash(SOURCE / filename) == expected, filename

    for name, script in (
        ("metrics_helper", "candidate_comparison_metrics.py"),
        ("evaluation_helper", "candidate_evaluation.py"),
    ):
        assert file_hash(ROOT / "scripts" / script) == (
            comparison["input_sha256"][name]
        )

    assert file_hash(
        ROOT / "models/rf_v2_23class_prototype.joblib"
    ) == comparison["input_sha256"]["frozen_v2_model"]

    for name in ("rf_group_weighted", "rf_class_balanced"):
        assert file_hash(
            EXPERIMENT / "models" / name / "model.joblib"
        ) == comparison["input_sha256"][f"{name}_model"]

    summary = pd.read_csv(SOURCE / "summary.csv")
    checked = set()

    for filename, expected_hash in comparison[
        "prediction_archive_sha256"
    ].items():
        path = SOURCE / "predictions" / filename
        assert file_hash(path) == expected_hash, filename

        model, setting = path.stem.split("__", 1)

        with np.load(path, allow_pickle=False) as archive:
            classes = archive["output_classes"]
            probabilities = archive["probabilities"]
            actual = archive["actual_labels"]
            cohorts = archive["evaluation_cohorts"]
            group_ids = archive["source_group_ids"]

            assert len(set(group_ids)) == len(group_ids)
            assert classes.tolist() == comparison["harmonized_output_labels"]

            aligned = harmonize_probabilities(
                archive["native_probabilities"],
                archive["native_classes"],
                classes,
            )
            np.testing.assert_allclose(
                aligned, probabilities, rtol=0, atol=1e-12
            )

            predicted = classes[probabilities.argmax(axis=1)]

            for cohort in (
                "selection_primary",
                "all_validation",
                "selection_legacy_overlap",
            ):
                mask = (
                    np.ones(len(actual), dtype=bool)
                    if cohort == "all_validation"
                    else cohorts == cohort
                )

                result = score_predictions(
                    actual[mask],
                    predicted[mask],
                    ground_truth_labels=comparison["ground_truth_labels"],
                    output_labels=classes,
                )["summary"]

                saved = summary.loc[
                    summary["model"].eq(model)
                    & summary["view"].eq(setting)
                    & summary["cohort"].eq(cohort)
                ]
                assert len(saved) == 1

                for metric, value in result.items():
                    assert np.isclose(
                        float(saved.iloc[0][metric]),
                        float(value),
                        rtol=0,
                        atol=1e-12,
                    ), (model, setting, cohort, metric)

                checked.add((model, setting, cohort))

        print(f"PASS: {model}, {setting}", flush=True)

    assert len(checked) == len(summary) == 45

    report = {
        "status": "passed",
        "summary_rows_recomputed": len(checked),
        "prediction_archives_checked": len(
            comparison["prediction_archive_sha256"]
        ),
        "documentation_files_checked": len(
            documentation["output_sha256"]
        ),
        "comparison_manifest_sha256": file_hash(
            SOURCE / "comparison_manifest.json"
        ),
        "verification_script_sha256": file_hash(Path(__file__).resolve()),
        "model_fingerprints_unchanged": True,
        "notes": [
            "Summary metrics were recomputed from saved prediction archives.",
            "Native-to-canonical probability alignment was checked.",
            "Documentation file fingerprints matched.",
            "No model predictions were rerun and no model was modified.",
        ],
    }

    output = REPORTS / "comparison_verification.json"
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("\nPASS: all 45 summary rows match the saved predictions.")
    print("Verification saved to:", output)


if __name__ == "__main__":
    main()