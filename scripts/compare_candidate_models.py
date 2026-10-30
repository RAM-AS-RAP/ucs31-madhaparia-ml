"""Compare frozen V2 and two candidates on the shared validation views."""

import gc
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_comparison_metrics import score_predictions
from scripts.candidate_evaluation import (
    VIEW_LIMITS,
    harmonize_probabilities,
    project_v2_features,
)


DATA = ROOT / "data/candidate_v1"
EXPERIMENT = ROOT / "data/candidate_experiment_v1"
VIEWS = EXPERIMENT / "validation_views"
OUTPUT = EXPERIMENT / "comparison"

CANDIDATES = ["rf_group_weighted", "rf_class_balanced"]
MODEL_NAMES = ["frozen_v2_harmonized", *CANDIDATES]
SPARSE_SETTINGS = ["up_to_1", "up_to_2", "up_to_3", "up_to_5"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_documentation_table(summary, selection):
    lines = [
        "# Candidate Model Validation Comparison",
        "",
        "These are validation results, not final-test or clinical results.",
        "Scores below are percentages.",
        "",
        "## Primary comparison cohort",
        "",
        "This cohort excludes normalized original-input matches with the",
        "original DDXPlus training source pool.",
        "",
        "| Model | View | Groups | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    primary = summary.loc[summary["cohort"].eq("selection_primary")]

    for row in primary.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['view']} | "
            f"{row['observations']:,} | "
            f"{100 * row['accuracy']:.2f}% | "
            f"{100 * row['macro_precision']:.2f}% | "
            f"{100 * row['macro_recall']:.2f}% | "
            f"{100 * row['macro_f1']:.2f}% | "
            f"{100 * row['weighted_f1']:.2f}% |"
        )

    lines.extend([
        "",
        "## Prespecified selection score",
        "",
        "Unweighted mean macro F1 across the four sparse settings",
        "on the primary comparison cohort.",
        "",
        "| Model | Mean sparse macro F1 |",
        "| --- | ---: |",
    ])

    for name in MODEL_NAMES:
        value = selection["mean_sparse_macro_f1"][name]
        lines.append(f"| {name} | {100 * value:.2f}% |")

    lines.extend([
        "",
        f"Candidate selected for further investigation: "
        f"**{selection['selected_candidate']}**.",
        "",
        "Selection does not authorize deployment or establish improvement",
        "for every condition.",
        "",
        "## Interpretation",
        "",
        "- Macro metrics use the fixed set of 21 ground-truth conditions.",
        "- Undefined contributions to macro averages use zero.",
        "- Per-condition CSV values are blank where a denominator is unavailable.",
        "- V2 allergy probabilities are combined before canonical ranking.",
        "- V2's COVID-19 score is retained; an incorrect COVID-19 prediction",
        "  remains an error.",
        "- COVID-19 sensitivity cannot be measured without positive cases.",
        "- Full and sparse views are separate settings, not independent patients.",
        "- Full-validation and legacy-overlap results are in summary.csv.",
        "- No calibration-reserved or final-test observations were evaluated.",
        "",
    ])

    (OUTPUT / "comparison_table.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Comparison directory already exists: {OUTPUT}")

    paths = {
        "dataset_manifest": DATA / "manifest.json",
        "training_csv": DATA / "train.csv",
        "validation_csv": DATA / "validation.csv",
        "partition_manifest": EXPERIMENT / "partition_manifest.json",
        "partitions_csv": EXPERIMENT / "partitions.csv",
        "view_manifest": VIEWS / "view_manifest.json",
        "protocol": ROOT / "docs/CANDIDATE_MODEL_EXPERIMENT_PROTOCOL.md",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "feature_builder": ROOT / "scripts/candidate_feature_builder.py",
        "evaluation_helper": ROOT / "scripts/candidate_evaluation.py",
        "metrics_helper": ROOT / "scripts/candidate_comparison_metrics.py",
        "view_preparation_script": (
            ROOT / "scripts/prepare_candidate_validation_views.py"
        ),
        "partition_script": ROOT / "scripts/prepare_candidate_partitions.py",
        "training_data_helper": ROOT / "scripts/candidate_training_data.py",
        "training_script": ROOT / "scripts/train_candidate_model.py",
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
        "comparison_script": Path(__file__).resolve(),
        "frozen_v2_model": ROOT / "models/rf_v2_23class_prototype.joblib",
        "frozen_v2_schema": ROOT / "models/v2_feature_schema.json",
        "frozen_v2_manifest": (
            ROOT / "models/rf_v2_23class_prototype_manifest.json"
        ),
    }

    for name in CANDIDATES:
        folder = EXPERIMENT / "models" / name
        paths[f"{name}_model"] = folder / "model.joblib"
        paths[f"{name}_manifest"] = folder / "training_manifest.json"

    for setting in VIEW_LIMITS:
        paths[f"view_{setting}"] = VIEWS / f"{setting}.csv"

    def fingerprints():
        return {name: file_hash(path) for name, path in paths.items()}

    print("Verifying models, views and dependencies...", flush=True)
    before = fingerprints()

    dataset_manifest = read_json(paths["dataset_manifest"])
    view_manifest = read_json(paths["view_manifest"])
    partition_manifest = read_json(paths["partition_manifest"])
    schema = read_json(paths["candidate_schema"])
    v2_schema = read_json(paths["frozen_v2_schema"])
    v2_manifest = read_json(paths["frozen_v2_manifest"])

    if schema != dataset_manifest["schema"]:
        raise ValueError("Candidate schema differs from dataset manifest")

    for key, expected in view_manifest["input_sha256"].items():
        if before[key] != expected:
            raise ValueError(f"Validation-view dependency changed: {key}")

    if before["partitions_csv"] != partition_manifest["partitions_sha256"]:
        raise ValueError("Partition fingerprint mismatch")

    for setting, details in view_manifest["views"].items():
        if before[f"view_{setting}"] != details["sha256"]:
            raise ValueError(f"Validation view changed: {setting}")

    training_manifests = {}
    for name in CANDIDATES:
        report = read_json(paths[f"{name}_manifest"])
        training_manifests[name] = report

        if before[f"{name}_model"] != report["model_sha256"]:
            raise ValueError(f"Candidate model changed: {name}")

        for key, expected in report["input_sha256"].items():
            if before[key] != expected:
                raise ValueError(f"Training dependency changed: {name}, {key}")

    truth_labels = sorted(set(dataset_manifest["label_mapping"].values()))
    if len(truth_labels) != 21 or "COVID-19" in truth_labels:
        raise ValueError("Unexpected ground-truth condition scope")

    output_labels = sorted([*truth_labels, "COVID-19"])
    output_array = np.asarray(output_labels, dtype=str)

    partitions = pd.read_csv(paths["partitions_csv"])
    if partitions["source_group_id"].duplicated().any():
        raise ValueError("Duplicate partition group IDs")

    validation_groups = partitions.loc[
        partitions["source_split"].eq("validation")
    ].set_index("source_group_id")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    prediction_dir = OUTPUT / "predictions"
    prediction_dir.mkdir()

    summaries = []
    condition_rows = []
    confusion_rows = []
    inference_times = []
    prediction_hashes = {}

    reference_ids = None
    reference_actual = None
    reference_cohorts = None

    for name in MODEL_NAMES:
        print(f"\nLoading {name}...", flush=True)

        if name == "frozen_v2_harmonized":
            model = joblib.load(paths["frozen_v2_model"])
            expected_features = v2_schema["feature_names"]
            expected_classes = sorted(v2_manifest["classes"])
        else:
            model = joblib.load(paths[f"{name}_model"])
            expected_features = schema["feature_names"]
            expected_classes = truth_labels

        if model.feature_names_in_.tolist() != expected_features:
            raise ValueError(f"Unexpected feature order: {name}")

        if model.classes_.tolist() != expected_classes:
            raise ValueError(f"Unexpected model classes: {name}")

        # Limit inference parallelism in memory; do not rewrite model files.
        model.set_params(n_jobs=2)

        for setting in VIEW_LIMITS:
            view = pd.read_csv(paths[f"view_{setting}"])

            if (
                not view["source_group_id"].is_unique
                or set(view["source_group_id"]) != set(validation_groups.index)
                or not view["variant_id"].eq(setting).all()
            ):
                raise ValueError(f"Invalid validation membership: {setting}")

            ids = view["source_group_id"].to_numpy(dtype=str)
            actual = view["canonical_label"].to_numpy(dtype=str)
            cohorts = view["evaluation_cohort"].to_numpy(dtype=str)

            expected = validation_groups.loc[ids]
            if not np.array_equal(
                actual, expected["canonical_label"].to_numpy(dtype=str)
            ):
                raise ValueError("Validation labels disagree with partitions")
            if not np.array_equal(
                cohorts, expected["partition"].to_numpy(dtype=str)
            ):
                raise ValueError("Validation cohorts disagree with partitions")

            if reference_ids is None:
                reference_ids = ids.copy()
                reference_actual = actual.copy()
                reference_cohorts = cohorts.copy()
            elif not (
                np.array_equal(ids, reference_ids)
                and np.array_equal(actual, reference_actual)
                and np.array_equal(cohorts, reference_cohorts)
            ):
                raise ValueError("Models or views received different groups")

            if name == "frozen_v2_harmonized":
                X = project_v2_features(view, v2_schema)
            else:
                X = view.loc[:, schema["feature_names"]]

            X = X.astype(np.float32)

            started = time.perf_counter()
            native_probabilities = np.vstack([
                model.predict_proba(X.iloc[start:start + 5000])
                for start in range(0, len(X), 5000)
            ])
            elapsed = time.perf_counter() - started

            probabilities = harmonize_probabilities(
                native_probabilities,
                model.classes_,
                output_labels,
            )
            predicted = output_array[probabilities.argmax(axis=1)]

            prediction_path = prediction_dir / f"{name}__{setting}.npz"
            np.savez_compressed(
                prediction_path,
                source_group_ids=ids,
                actual_labels=actual,
                evaluation_cohorts=cohorts,
                output_classes=output_array,
                probabilities=probabilities,
                native_classes=np.asarray(model.classes_, dtype=str),
                native_probabilities=native_probabilities,
            )
            prediction_hashes[prediction_path.name] = file_hash(prediction_path)

            inference_times.append({
                "model": name,
                "view": setting,
                "observations": len(view),
                "prediction_seconds": elapsed,
            })

            for cohort in (
                "selection_primary",
                "all_validation",
                "selection_legacy_overlap",
            ):
                selected = (
                    np.ones(len(view), dtype=bool)
                    if cohort == "all_validation"
                    else cohorts == cohort
                )

                result = score_predictions(
                    actual[selected],
                    predicted[selected],
                    ground_truth_labels=truth_labels,
                    output_labels=output_labels,
                )

                context = {"model": name, "view": setting, "cohort": cohort}
                summaries.append({**context, **result["summary"]})

                condition_rows.extend(
                    {**context, **row} for row in result["per_condition"]
                )

                for true_index, true_label in enumerate(output_labels):
                    for pred_index, pred_label in enumerate(output_labels):
                        confusion_rows.append({
                            **context,
                            "actual_condition": true_label,
                            "predicted_condition": pred_label,
                            "count": result["confusion_matrix"][
                                true_index
                            ][pred_index],
                        })

                if cohort == "selection_primary":
                    scores = result["summary"]
                    print(
                        f"  {setting}: "
                        f"accuracy={100 * scores['accuracy']:.2f}% | "
                        f"macro F1={100 * scores['macro_f1']:.2f}%",
                        flush=True,
                    )

            del view, X, native_probabilities, probabilities
            gc.collect()

        del model
        gc.collect()

    if before != fingerprints():
        raise RuntimeError("An input changed during comparison")

    summary = pd.DataFrame(summaries)
    primary = summary.loc[summary["cohort"].eq("selection_primary")]

    sparse_scores = {}
    full_scores = {}

    for name in MODEL_NAMES:
        model_rows = primary.loc[primary["model"].eq(name)]
        sparse = model_rows.loc[model_rows["view"].isin(SPARSE_SETTINGS)]
        full = model_rows.loc[model_rows["view"].eq("full")]

        if len(sparse) != 4 or len(full) != 1:
            raise ValueError("Incomplete candidate-selection results")

        sparse_scores[name] = float(sparse["macro_f1"].mean())
        full_scores[name] = float(full.iloc[0]["macro_f1"])

    selected_candidate = sorted(
        CANDIDATES,
        key=lambda name: (
            -sparse_scores[name],
            -full_scores[name],
            CANDIDATES.index(name),
        ),
    )[0]

    selection = {
        "criterion": "mean primary-cohort macro F1 across four sparse views",
        "mean_sparse_macro_f1": sparse_scores,
        "full_primary_macro_f1": full_scores,
        "selected_candidate": selected_candidate,
        "selected_candidate_exceeds_v2_selection_score": (
            sparse_scores[selected_candidate]
            > sparse_scores["frozen_v2_harmonized"]
        ),
        "deployment_approved": False,
    }

    summary.to_csv(OUTPUT / "summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(condition_rows).to_csv(
        OUTPUT / "per_condition.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT / "confusion_counts.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(inference_times).to_csv(
        OUTPUT / "inference_times.csv", index=False, lineterminator="\n"
    )

    (OUTPUT / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    write_documentation_table(summary, selection)

    report_names = [
        "summary.csv",
        "per_condition.csv",
        "confusion_counts.csv",
        "inference_times.csv",
        "selection.json",
        "comparison_table.md",
    ]

    manifest = {
        "issue": 48,
        "status": "validation_comparison_completed_not_deployed",
        "input_sha256": before,
        "ground_truth_labels": truth_labels,
        "harmonized_output_labels": output_labels,
        "selection": selection,
        "report_sha256": {
            filename: file_hash(OUTPUT / filename)
            for filename in report_names
        },
        "prediction_archive_sha256": prediction_hashes,
        "candidate_training_seconds": {
            name: training_manifests[name]["training_duration_seconds"]
            for name in CANDIDATES
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "notes": [
            "V2 allergy probabilities are combined before canonical ranking.",
            "V2 COVID-19 probability is retained without renormalization.",
            "Macro metrics use 21 fixed ground-truth labels and zero_division=0.",
            "Probability scores are not calibrated clinical probabilities.",
            "Views reuse source groups and are not independent patients.",
            "No calibration-reserved or final-test predictions were generated.",
            "Inference timing is descriptive, not a controlled benchmark.",
            "All input artifacts, including frozen V2, remained unchanged.",
        ],
    }

    (OUTPUT / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nCOMPARISON COMPLETED")
    for name in MODEL_NAMES:
        print(
            f"{name}: mean sparse macro F1 = "
            f"{100 * sparse_scores[name]:.2f}%"
        )
    print("Selected research candidate:", selected_candidate)
    print(
        "Exceeds V2 selection score:",
        selection["selected_candidate_exceeds_v2_selection_score"],
    )
    print("Reports and prediction archives:", OUTPUT)


if __name__ == "__main__":
    main()