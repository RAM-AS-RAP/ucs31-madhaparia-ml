"""Prepare reserved-data observation views without inspecting predictions."""

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_calibration_data import (
    CALIBRATION_PARTITIONS,
    load_calibration_full_rows,
    validate_calibration_partitions,
)
from scripts.candidate_evaluation import VIEW_LIMITS, make_validation_view
from scripts.candidate_feature_builder import load_candidate_schema
from scripts.prepare_candidate_calibration_partitions import (
    assign_calibration_partitions,
)


DATA = ROOT / "data/candidate_v1"
EXPERIMENT = ROOT / "data/candidate_experiment_v1"
CALIBRATION = EXPERIMENT / "calibration"
OUTPUT = CALIBRATION / "views"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if OUTPUT.exists():
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
        "view_preparation_script": Path(__file__).resolve(),
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
    }

    def fingerprints():
        return {name: file_hash(path) for name, path in inputs.items()}

    print("Checking input fingerprints...", flush=True)
    before = fingerprints()

    dataset_manifest = read_json(inputs["dataset_manifest"])
    parent_manifest = read_json(inputs["parent_manifest"])
    calibration_manifest = read_json(inputs["calibration_manifest"])
    previous_views = read_json(inputs["previous_view_manifest"])
    schema = load_candidate_schema()

    if schema != dataset_manifest["schema"]:
        raise ValueError("Candidate schema differs from the built dataset")

    if (
        before["training_csv"]
        != dataset_manifest["output_sha256"]["train.csv"]
    ):
        raise ValueError("Training CSV fingerprint mismatch")

    if (
        before["dataset_manifest"]
        != parent_manifest["input_sha256"]["dataset_manifest"]
    ):
        raise ValueError("Dataset manifest changed since original partitioning")

    if before["parent_partitions"] != parent_manifest["partitions_sha256"]:
        raise ValueError("Original partition CSV fingerprint mismatch")

    if (
        before["calibration_partitions"]
        != calibration_manifest["partitions_sha256"]
    ):
        raise ValueError("Calibration partition CSV fingerprint mismatch")

    for name in (
        "protocol",
        "parent_manifest",
        "parent_partitions",
        "partition_script",
        "hash_helper",
    ):
        if before[name] != calibration_manifest["input_sha256"][name]:
            raise ValueError(f"Calibration partition dependency changed: {name}")

    if (
        before["feature_builder"]
        != dataset_manifest["input_sha256"]["candidate_feature_builder"]
    ):
        raise ValueError("Feature builder changed since dataset generation")

    if (
        before["evaluation_helper"]
        != previous_views["input_sha256"]["evaluation_helper"]
    ):
        raise ValueError("View-generation helper changed since issue #48")

    parent = pd.read_csv(inputs["parent_partitions"], dtype=str)
    partitions = pd.read_csv(inputs["calibration_partitions"], dtype=str)

    validate_calibration_partitions(partitions)

    if not parent["source_group_id"].is_unique:
        raise ValueError("Duplicate source groups in original partitions")

    reserved = parent.loc[
        parent["partition"].eq("calibration_reserved")
    ]

    # Recompute the deterministic assignments to check membership,
    # labels and roles, rather than checking only total counts.
    expected = assign_calibration_partitions(
        reserved.to_dict(orient="records")
    )
    actual = partitions.sort_values(
        "source_group_id"
    ).to_dict(orient="records")

    if actual != expected:
        raise ValueError("Calibration assignments differ from the protocol")

    counts = {
        str(name): int(count)
        for name, count in partitions["partition"].value_counts().items()
    }

    if counts != calibration_manifest["group_counts"]:
        raise ValueError("Calibration partition counts differ from manifest")

    other_ids = set(
        parent.loc[
            parent["partition"].ne("calibration_reserved"),
            "source_group_id",
        ]
    )
    if set(partitions["source_group_id"]) & other_ids:
        raise ValueError("Reserved groups overlap fitting or validation groups")

    expected_settings = {
        "full",
        "up_to_1",
        "up_to_2",
        "up_to_3",
        "up_to_5",
    }
    if set(VIEW_LIMITS) != expected_settings:
        raise ValueError("Unexpected observation settings")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    view_reports = {}
    condition_support = {}

    for partition in CALIBRATION_PARTITIONS:
        print(f"\nLoading {partition} full observations...", flush=True)

        full = load_calibration_full_rows(
            inputs["training_csv"],
            partitions,
            schema,
            partition=partition,
        )

        if len(full) != counts[partition]:
            raise ValueError(f"Unexpected observation count: {partition}")

        support = {
            str(label): int(count)
            for label, count in (
                full["canonical_label"].value_counts().sort_index().items()
            )
        }

        if support != calibration_manifest[
            "groups_by_partition_and_class"
        ][partition]:
            raise ValueError(f"Condition support mismatch: {partition}")

        condition_support[partition] = support
        partition_output = OUTPUT / partition
        partition_output.mkdir(exist_ok=False)
        view_reports[partition] = {}

        for setting, limit in VIEW_LIMITS.items():
            print(f"  Preparing {setting}...", flush=True)

            if setting == "full":
                view = full.copy()
            else:
                view = make_validation_view(
                    full, schema, setting=setting, seed=42
                )

            for name in (
                "source_group_id",
                "canonical_label",
                "calibration_partition",
            ):
                if not view[name].equals(full[name]):
                    raise ValueError(f"View changed observation identity: {name}")

            if not view["source_group_id"].is_unique:
                raise ValueError("Duplicate source group within a view")

            filename = f"{partition}/{setting}.csv"
            path = OUTPUT / filename
            view.to_csv(path, index=False, lineterminator="\n")

            view_reports[partition][setting] = {
                "filename": filename,
                "sha256": file_hash(path),
                "rows": len(view),
                "requested_max_reporting_units": limit,
                "actual_reporting_unit_counts": {
                    str(int(count)): int(frequency)
                    for count, frequency in (
                        view["reported_unit_count"]
                        .value_counts()
                        .sort_index()
                        .items()
                    )
                },
            }

            print(f"    Saved {len(view)} observations.", flush=True)

        del full, view

    if before != fingerprints():
        raise RuntimeError("An input changed while preparing views")

    report = {
        "issue": 49,
        "status": "reserved_views_prepared_no_predictions",
        "seed": 42,
        "input_sha256": before,
        "partition_group_counts": {
            name: counts[name] for name in CALIBRATION_PARTITIONS
        },
        "condition_support": condition_support,
        "calibration_fit_weight_per_group_per_view": 1.0 / len(VIEW_LIMITS),
        "views": view_reports,
        "notes": [
            "Each view contains one observation per source group.",
            "Each group remains in its assigned partition across all views.",
            "Training sample weights were not carried into these views.",
            "Calibration fitting must assign weight 1/5 per group per view.",
            "Report each evaluation view separately.",
            "Repeated views are not independent patients.",
            "No model or prediction archive was loaded.",
            "Evaluation predictions must wait until the policy is frozen.",
            "The final test split was not inspected.",
        ],
    }

    (OUTPUT / "view_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nRESERVED OBSERVATION VIEWS READY")
    for partition in CALIBRATION_PARTITIONS:
        print(f"  {partition}: {counts[partition]} groups per view")

    print("Views:", ", ".join(VIEW_LIMITS))
    print("Input fingerprints unchanged.")
    print("No predictions inspected and no model trained.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()