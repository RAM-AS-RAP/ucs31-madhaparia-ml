"""Train one candidate using the fixed issue #48 experiment protocol."""

import argparse
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
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_feature_builder import load_candidate_schema
from scripts.candidate_training_data import (
    CANDIDATE_NAMES,
    load_fitting_rows,
    sample_weights_for_candidate,
)


DATA = ROOT / "data/candidate_v1"
EXPERIMENT = ROOT / "data/candidate_experiment_v1"

RF_PARAMETERS = {
    "n_estimators": 180,
    "criterion": "gini",
    "max_depth": 18,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "max_leaf_nodes": 4096,
    "bootstrap": True,
    "class_weight": None,
    "random_state": 42,
    "n_jobs": 2,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(candidate_name, check_only=False):
    if candidate_name not in CANDIDATE_NAMES:
        raise ValueError("Unknown candidate")

    output = EXPERIMENT / "models" / candidate_name
    if output.exists() and not check_only:
        raise FileExistsError(
            f"Candidate output already exists: {output}"
        )

    input_paths = {
        "dataset_manifest": DATA / "manifest.json",
        "training_csv": DATA / "train.csv",
        "partition_manifest": EXPERIMENT / "partition_manifest.json",
        "partitions_csv": EXPERIMENT / "partitions.csv",
        "protocol": ROOT / "docs/CANDIDATE_MODEL_EXPERIMENT_PROTOCOL.md",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "partition_script": ROOT / "scripts/prepare_candidate_partitions.py",
        "training_data_helper": ROOT / "scripts/candidate_training_data.py",
        "feature_builder": ROOT / "scripts/candidate_feature_builder.py",
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
        "training_script": Path(__file__).resolve(),
        "frozen_v2_model": ROOT / "models/rf_v2_23class_prototype.joblib",
        "frozen_v2_schema": ROOT / "models/v2_feature_schema.json",
    }

    def fingerprints():
        return {
            name: file_hash(path)
            for name, path in input_paths.items()
        }

    print("Checking input fingerprints...", flush=True)
    before = fingerprints()

    dataset_manifest = read_json(input_paths["dataset_manifest"])
    partition_manifest = read_json(input_paths["partition_manifest"])
    schema = load_candidate_schema()

    if schema != dataset_manifest["schema"]:
        raise ValueError("Current schema differs from the built dataset")

    if before["training_csv"] != dataset_manifest[
        "output_sha256"
    ]["train.csv"]:
        raise ValueError("Training CSV fingerprint mismatch")

    if before["partitions_csv"] != partition_manifest["partitions_sha256"]:
        raise ValueError("Partition CSV fingerprint mismatch")

    partition_inputs = partition_manifest["input_sha256"]

    for name in (
        "dataset_manifest",
        "protocol",
        "partition_script",
        "hash_helper",
    ):
        if before[name] != partition_inputs[name]:
            raise ValueError(f"Partition dependency changed: {name}")

    if before["feature_builder"] != dataset_manifest[
        "input_sha256"
    ]["candidate_feature_builder"]:
        raise ValueError("Feature builder changed since dataset generation")

    partitions = pd.read_csv(input_paths["partitions_csv"])

    required = {
        "source_group_id",
        "source_split",
        "canonical_label",
        "partition",
    }
    if not required.issubset(partitions.columns):
        raise ValueError("Missing partition columns")

    if partitions[list(required)].isna().any().any():
        raise ValueError("Missing partition values")

    if partitions["source_group_id"].duplicated().any():
        raise ValueError("Duplicate partition group IDs")

    expected_source_split = {
        "fit": "train",
        "calibration_reserved": "train",
        "selection_primary": "validation",
        "selection_legacy_overlap": "validation",
    }

    mapped_splits = partitions["partition"].map(expected_source_split)
    if not mapped_splits.eq(partitions["source_split"]).all():
        raise ValueError("Partition roles disagree with original splits")

    actual_counts = partitions["partition"].value_counts().to_dict()
    if actual_counts != partition_manifest["group_counts"]:
        raise ValueError("Partition counts differ from the manifest")

    fit_groups = partitions.loc[
        partitions["partition"].eq("fit")
    ].copy()

    actual_class_groups = (
        fit_groups["canonical_label"].value_counts().to_dict()
    )
    expected_class_groups = partition_manifest[
        "groups_by_partition_and_class"
    ]["fit"]

    if actual_class_groups != expected_class_groups:
        raise ValueError("Fitting class counts differ from the manifest")

    print("Loading fitting observations only...", flush=True)
    rows = load_fitting_rows(
        input_paths["training_csv"], partitions, schema
    )

    weights, class_factors = sample_weights_for_candidate(
        rows, fit_groups, candidate_name
    )

    feature_names = schema["feature_names"]
    X = rows.loc[:, feature_names]
    y = rows["canonical_label"].to_numpy()

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite model inputs")

    expected_classes = sorted(set(dataset_manifest["label_mapping"].values()))
    if sorted(set(y)) != expected_classes or len(expected_classes) != 21:
        raise ValueError("Unexpected fitting condition coverage")

    rows_by_class = rows["canonical_label"].value_counts().to_dict()
    weight_by_class = (
        pd.DataFrame({"label": y, "weight": weights})
        .groupby("label")["weight"]
        .sum()
        .to_dict()
    )

    fitting_summary = {
        "source_groups": len(fit_groups),
        "observations": len(rows),
        "feature_count": len(feature_names),
        "source_groups_by_class": dict(sorted(actual_class_groups.items())),
        "observations_by_class": dict(sorted(rows_by_class.items())),
        "class_weight_factors": class_factors,
        "sample_weight_by_class": dict(sorted(weight_by_class.items())),
        "total_sample_weight": float(weights.sum()),
        "minimum_sample_weight": float(weights.min()),
        "maximum_sample_weight": float(weights.max()),
        "approx_feature_matrix_mib": float(
            X.memory_usage(index=False, deep=True).sum() / (1024 ** 2)
        ),
    }

    print("\nCANDIDATE:", candidate_name)
    print("Fitting source groups:", fitting_summary["source_groups"])
    print("Fitting observations:", fitting_summary["observations"])
    print("Features:", fitting_summary["feature_count"])
    print("Conditions:", len(expected_classes))
    print("Total sample weight:", fitting_summary["total_sample_weight"])
    print(
        "Sample weight range:",
        fitting_summary["minimum_sample_weight"],
        "to",
        fitting_summary["maximum_sample_weight"],
    )
    print(
        "Feature matrix MiB:",
        round(fitting_summary["approx_feature_matrix_mib"], 1),
    )
    print("Calibration-reserved groups used for fitting: 0")
    print("Validation groups used for fitting: 0")

    if before != fingerprints():
        raise RuntimeError("An input changed during fitting-data preparation")

    if check_only:
        print("\nPASS: preflight checks completed. No model trained or saved.")
        return

    # Keep the model input matrix, targets and weights; release metadata.
    del rows, partitions, fit_groups
    gc.collect()

    print("\nTraining Random Forest...", flush=True)
    model = RandomForestClassifier(**RF_PARAMETERS)

    started = time.perf_counter()
    model.fit(X, y, sample_weight=weights)
    training_seconds = time.perf_counter() - started

    if model.classes_.tolist() != expected_classes:
        raise ValueError("Trained model classes differ from expected classes")

    if model.feature_names_in_.tolist() != feature_names:
        raise ValueError("Trained model feature order differs from schema")

    if before != fingerprints():
        raise RuntimeError("An input changed during training; model not saved")

    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "model.joblib"
    joblib.dump(model, model_path, compress=3)

    report = {
        "issue": 48,
        "status": "trained_not_evaluated",
        "candidate_name": candidate_name,
        "schema_id": schema["schema_id"],
        "feature_names": feature_names,
        "classes": model.classes_.tolist(),
        "random_forest_parameters": RF_PARAMETERS,
        "fitting_summary": fitting_summary,
        "training_duration_seconds": training_seconds,
        "input_sha256": before,
        "model_sha256": file_hash(model_path),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "notes": [
            "Only fit-partition observations were used for model fitting.",
            "Calibration-reserved and validation groups were excluded.",
            "Class balancing, when used, is based on fitting source groups.",
            "No final-test records were inspected.",
            "The frozen V2 model and schema fingerprints are unchanged.",
            "This artifact is not calibrated or approved for deployment.",
        ],
    }

    (output / "training_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nTRAINING COMPLETED")
    print("Candidate:", candidate_name)
    print("Duration in seconds:", round(training_seconds, 1))
    print("Model:", model_path)
    print("Status: trained, not evaluated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        required=True,
        choices=CANDIDATE_NAMES,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate fitting inputs without training or writing artifacts",
    )
    args = parser.parse_args()

    run(args.candidate, check_only=args.check_only)


if __name__ == "__main__":
    main()