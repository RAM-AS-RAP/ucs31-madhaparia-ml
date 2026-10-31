"""Fit temperature scaling using calibration_fit observations only."""

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_calibration_data import validate_calibration_partitions
from scripts.candidate_probability_calibration import (
    fit_temperature,
    validate_probabilities,
)


DATA = ROOT / "data/candidate_v1"
EXPERIMENT = ROOT / "data/candidate_experiment_v1"
CALIBRATION = EXPERIMENT / "calibration"
VIEWS = CALIBRATION / "views"
MODEL_FOLDER = EXPERIMENT / "models/rf_group_weighted"
OUTPUT = CALIBRATION / "temperature_fit"

SETTINGS = ("full", "up_to_1", "up_to_2", "up_to_3", "up_to_5")
PARTITION = "calibration_fit"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check inputs without predicting, fitting or writing files.",
    )
    args = parser.parse_args()

    if OUTPUT.exists() and not args.check_only:
        raise FileExistsError(f"Output directory already exists: {OUTPUT}")

    inputs = {
        "dataset_manifest": DATA / "manifest.json",
        "training_csv": DATA / "train.csv",
        "parent_manifest": EXPERIMENT / "partition_manifest.json",
        "parent_partitions": EXPERIMENT / "partitions.csv",
        "calibration_manifest": CALIBRATION / "partition_manifest.json",
        "calibration_partitions": CALIBRATION / "partitions.csv",
        "protocol": ROOT / "docs/CANDIDATE_CALIBRATION_PROTOCOL.md",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "feature_builder": ROOT / "scripts/candidate_feature_builder.py",
        "evaluation_helper": ROOT / "scripts/candidate_evaluation.py",
        "calibration_data_helper": ROOT / "scripts/candidate_calibration_data.py",
        "partition_script": (
            ROOT / "scripts/prepare_candidate_calibration_partitions.py"
        ),
        "previous_view_manifest": (
            EXPERIMENT / "validation_views/view_manifest.json"
        ),
        "view_preparation_script": (
            ROOT / "scripts/prepare_candidate_calibration_views.py"
        ),
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
        "view_manifest": VIEWS / "view_manifest.json",
        "model": MODEL_FOLDER / "model.joblib",
        "training_manifest": MODEL_FOLDER / "training_manifest.json",
        "archived_training_manifest": (
            ROOT
            / "reports/candidate_model_comparison"
            / "rf_group_weighted_training_manifest.json"
        ),
        "selection": ROOT / "reports/candidate_model_comparison/selection.json",
        "calibration_helper": (
            ROOT / "scripts/candidate_probability_calibration.py"
        ),
        "fitting_script": Path(__file__).resolve(),
    }

    for setting in SETTINGS:
        inputs[f"view_{setting}"] = VIEWS / PARTITION / f"{setting}.csv"

    def fingerprints():
        return {name: file_hash(path) for name, path in inputs.items()}

    print("Checking model, views and dependencies...", flush=True)
    before = fingerprints()

    view_manifest = read_json(inputs["view_manifest"])
    partition_manifest = read_json(inputs["calibration_manifest"])
    training_manifest = read_json(inputs["training_manifest"])
    schema = read_json(inputs["candidate_schema"])
    selection = read_json(inputs["selection"])

    if selection["selected_candidate"] != "rf_group_weighted":
        raise ValueError("Unexpected selected research candidate")

    if (
        before["training_manifest"]
        != before["archived_training_manifest"]
    ):
        raise ValueError("Training manifest differs from the issue #48 archive")

    if before["model"] != training_manifest["model_sha256"]:
        raise ValueError("Selected model fingerprint mismatch")

    if training_manifest["candidate_name"] != "rf_group_weighted":
        raise ValueError("Unexpected training manifest candidate")

    for name, expected in view_manifest["input_sha256"].items():
        if before[name] != expected:
            raise ValueError(f"View dependency changed: {name}")

    if (
        before["calibration_partitions"]
        != partition_manifest["partitions_sha256"]
    ):
        raise ValueError("Calibration partition fingerprint mismatch")

    training_dependencies = {
        "dataset_manifest": "dataset_manifest",
        "training_csv": "training_csv",
        "parent_manifest": "partition_manifest",
        "parent_partitions": "partitions_csv",
        "candidate_schema": "candidate_schema",
    }
    for current_name, original_name in training_dependencies.items():
        if (
            before[current_name]
            != training_manifest["input_sha256"][original_name]
        ):
            raise ValueError(f"Training dependency changed: {current_name}")

    if schema["feature_names"] != training_manifest["feature_names"]:
        raise ValueError("Model and candidate feature order disagree")

    partitions = pd.read_csv(inputs["calibration_partitions"], dtype=str)
    validate_calibration_partitions(partitions)

    fit_groups = partitions.loc[
        partitions["partition"].eq(PARTITION)
    ].set_index("source_group_id")

    expected_count = partition_manifest["group_counts"][PARTITION]
    if len(fit_groups) != expected_count or expected_count != 50174:
        raise ValueError("Unexpected calibration-fitting group count")

    expected_ids = set(fit_groups.index)
    expected_support = partition_manifest[
        "groups_by_partition_and_class"
    ][PARTITION]

    if fit_groups["canonical_label"].value_counts().to_dict() != expected_support:
        raise ValueError("Calibration-fitting condition support mismatch")

    classes = list(training_manifest["classes"])
    if len(classes) != 21 or len(classes) != len(set(classes)):
        raise ValueError("Expected 21 unique model classes")

    if set(classes) != set(expected_support):
        raise ValueError("Model classes differ from calibration-fitting labels")

    class_index = {label: index for index, label in enumerate(classes)}
    features = schema["feature_names"]
    frames = {}

    for setting in SETTINGS:
        details = view_manifest["views"][PARTITION][setting]

        if details["filename"] != f"{PARTITION}/{setting}.csv":
            raise ValueError(f"Unexpected view filename: {setting}")

        if before[f"view_{setting}"] != details["sha256"]:
            raise ValueError(f"View fingerprint mismatch: {setting}")

        frame = pd.read_csv(inputs[f"view_{setting}"])

        if len(frame) != expected_count or len(frame) != details["rows"]:
            raise ValueError(f"Unexpected view size: {setting}")

        if (
            not frame["source_group_id"].is_unique
            or set(frame["source_group_id"]) != expected_ids
        ):
            raise ValueError(f"View contains missing or foreign groups: {setting}")

        if not frame["calibration_partition"].eq(PARTITION).all():
            raise ValueError(f"View contains another partition: {setting}")

        if not frame["variant_id"].eq(setting).all():
            raise ValueError(f"View variant mismatch: {setting}")

        expected_labels = frame["source_group_id"].map(
            fit_groups["canonical_label"]
        )
        if not expected_labels.eq(frame["canonical_label"]).all():
            raise ValueError(f"View label mismatch: {setting}")

        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite features: {setting}")

        frames[setting] = frame

    model = joblib.load(inputs["model"])

    if list(model.classes_) != classes:
        raise ValueError("Loaded model class order differs from manifest")

    if model.n_features_in_ != len(features):
        raise ValueError("Loaded model feature count differs from schema")

    if (
        hasattr(model, "feature_names_in_")
        and list(model.feature_names_in_) != features
    ):
        raise ValueError("Loaded model feature order differs from schema")

    print("\nTEMPERATURE-FITTING INPUTS")
    print("Source groups:", expected_count)
    print("Views per group:", len(SETTINGS))
    print("Observations:", sum(len(frame) for frame in frames.values()))
    print("Weight per observation:", 1.0 / len(SETTINGS))
    print("Total fitting weight:", float(expected_count))
    print("Policy-development groups used: 0")
    print("Uncertainty-evaluation groups used: 0")

    if args.check_only:
        if before != fingerprints():
            raise RuntimeError("An input changed during preflight")
        print("\nPASS: preflight checks completed.")
        print("No predictions calculated, temperature fitted or files saved.")
        return

    archives = {}
    probability_blocks = []
    target_blocks = []

    for setting in SETTINGS:
        print(f"\nPredicting calibration_fit / {setting}...", flush=True)
        frame = frames[setting]
        blocks = []

        for start in range(0, len(frame), 5000):
            batch = frame.iloc[start:start + 5000]
            blocks.append(model.predict_proba(batch[features]))

        probabilities = np.vstack(blocks)
        validate_probabilities(probabilities)

        targets = frame["canonical_label"].map(
            class_index
        ).to_numpy(dtype=np.int64)

        archives[setting] = {
            "source_group_ids": frame["source_group_id"].to_numpy(dtype=str),
            "actual": frame["canonical_label"].to_numpy(dtype=str),
            "classes": np.asarray(classes, dtype=str),
            "probabilities": probabilities,
            "target_indices": targets,
            "reported_unit_count": frame[
                "reported_unit_count"
            ].to_numpy(dtype=np.int64),
        }
        probability_blocks.append(probabilities)
        target_blocks.append(targets)

    probabilities = np.vstack(probability_blocks)
    targets = np.concatenate(target_blocks)
    weights = np.full(
        len(targets), 1.0 / len(SETTINGS), dtype=np.float64
    )

    if not np.isclose(weights.sum(), expected_count):
        raise ValueError("Unexpected total calibration-fitting weight")

    print("\nFitting scalar temperature...", flush=True)
    fitted = fit_temperature(
        probabilities,
        targets,
        sample_weight=weights,
    )

    if before != fingerprints():
        raise RuntimeError("An input changed during temperature fitting")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    prediction_folder = OUTPUT / "predictions"
    prediction_folder.mkdir()

    output_hashes = {}

    for setting, arrays in archives.items():
        path = prediction_folder / f"{setting}.npz"
        np.savez_compressed(path, **arrays)
        output_hashes[f"predictions/{setting}.npz"] = file_hash(path)

    report = {
        "issue": 49,
        "status": "temperature_fitted_not_selected_or_deployed",
        "candidate_name": "rf_group_weighted",
        "partition": PARTITION,
        "source_groups": expected_count,
        "settings": list(SETTINGS),
        "weight_per_observation": 1.0 / len(SETTINGS),
        "classes": classes,
        "feature_names": features,
        "input_sha256": before,
        "output_sha256": output_hashes,
        "fit": fitted,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "notes": [
            "Only calibration_fit observations supplied to the optimizer.",
            "The original Random Forest was not retrained or overwritten.",
            "Fitting loss is not held-out calibration performance.",
            "Policy-development predictions were not calculated.",
            "Uncertainty-evaluation predictions were not calculated.",
            "The final test split was not inspected.",
        ],
    }

    (OUTPUT / "temperature_fit.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nTEMPERATURE FITTED")
    print(f"Temperature: {fitted['temperature']:.6f}")
    print(
        "Fitting log loss before:",
        f"{fitted['weighted_log_loss_before']:.6f}",
    )
    print(
        "Fitting log loss after:",
        f"{fitted['weighted_log_loss_after']:.6f}",
    )
    print("Boundary solution:", fitted["boundary_solution"])
    print("Leading-class indices unchanged.")
    print("Input/model fingerprints unchanged.")
    print("Status: fitted, not selected or deployed.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()