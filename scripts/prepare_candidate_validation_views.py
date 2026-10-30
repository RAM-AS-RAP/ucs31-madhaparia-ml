"""Prepare shared full and sparse validation views without running models."""

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_evaluation import VIEW_LIMITS, make_validation_view
from scripts.candidate_feature_builder import load_candidate_schema


DATA = ROOT / "data/candidate_v1"
EXPERIMENT = ROOT / "data/candidate_experiment_v1"
OUTPUT = EXPERIMENT / "validation_views"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT}")

    inputs = {
        "dataset_manifest": DATA / "manifest.json",
        "validation_csv": DATA / "validation.csv",
        "partition_manifest": EXPERIMENT / "partition_manifest.json",
        "partitions_csv": EXPERIMENT / "partitions.csv",
        "protocol": ROOT / "docs/CANDIDATE_MODEL_EXPERIMENT_PROTOCOL.md",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "feature_builder": ROOT / "scripts/candidate_feature_builder.py",
        "evaluation_helper": ROOT / "scripts/candidate_evaluation.py",
        "view_preparation_script": Path(__file__).resolve(),
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
    }

    def fingerprints():
        return {
            name: file_hash(path) for name, path in inputs.items()
        }

    before = fingerprints()

    dataset_manifest = read_json(inputs["dataset_manifest"])
    partition_manifest = read_json(inputs["partition_manifest"])
    schema = load_candidate_schema()

    if schema != dataset_manifest["schema"]:
        raise ValueError("Candidate schema differs from built dataset")

    if before["validation_csv"] != dataset_manifest[
        "output_sha256"
    ]["validation.csv"]:
        raise ValueError("Validation CSV fingerprint mismatch")

    if before["partitions_csv"] != partition_manifest["partitions_sha256"]:
        raise ValueError("Partition CSV fingerprint mismatch")

    for name in ("dataset_manifest", "protocol"):
        if before[name] != partition_manifest["input_sha256"][name]:
            raise ValueError(f"Partition dependency changed: {name}")

    if before["feature_builder"] != dataset_manifest[
        "input_sha256"
    ]["candidate_feature_builder"]:
        raise ValueError("Feature builder changed since dataset generation")

    partitions = pd.read_csv(inputs["partitions_csv"])
    if partitions["source_group_id"].duplicated().any():
        raise ValueError("Duplicate source-group IDs in partition table")

    validation_partitions = partitions.loc[
        partitions["source_split"].eq("validation")
    ].set_index("source_group_id")

    allowed_cohorts = {
        "selection_primary",
        "selection_legacy_overlap",
    }
    if not validation_partitions["partition"].isin(allowed_cohorts).all():
        raise ValueError("Unexpected validation partition role")

    full = pd.read_csv(inputs["validation_csv"])

    if not full["source_group_id"].is_unique:
        raise ValueError("Duplicate source groups in validation CSV")

    if set(full["source_group_id"]) != set(validation_partitions.index):
        raise ValueError("Validation observations and partition groups differ")

    expected_labels = full["source_group_id"].map(
        validation_partitions["canonical_label"]
    )
    if not expected_labels.eq(full["canonical_label"]).all():
        raise ValueError("Validation labels disagree with partition labels")

    full["evaluation_cohort"] = full["source_group_id"].map(
        validation_partitions["partition"]
    )

    cohort_counts = full["evaluation_cohort"].value_counts().to_dict()
    for cohort in allowed_cohorts:
        if cohort_counts.get(cohort, 0) != partition_manifest[
            "group_counts"
        ][cohort]:
            raise ValueError(f"Unexpected group count for {cohort}")

    if len(full) != dataset_manifest["splits"]["validation"][
        "totals"
    ]["scenario_rows"]:
        raise ValueError("Validation row count differs from dataset manifest")

    cohort_support = {
        cohort: {
            label: int(count)
            for label, count in (
                full.loc[full["evaluation_cohort"].eq(cohort), "canonical_label"]
                .value_counts()
                .sort_index()
                .items()
            )
        }
        for cohort in sorted(allowed_cohorts)
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    views = {}

    for setting, limit in VIEW_LIMITS.items():
        print(f"Preparing {setting}...", flush=True)
        view = make_validation_view(full, schema, setting, seed=42)

        if not view["source_group_id"].equals(full["source_group_id"]):
            raise ValueError("Source-group order changed")

        if not view["canonical_label"].equals(full["canonical_label"]):
            raise ValueError("Ground-truth labels changed")

        filename = f"{setting}.csv"
        path = OUTPUT / filename
        view.to_csv(path, index=False, lineterminator="\n")

        views[setting] = {
            "filename": filename,
            "sha256": file_hash(path),
            "rows": len(view),
            "requested_max_reporting_units": limit,
            "actual_unit_counts_by_cohort": {
                cohort: {
                    str(count): int(frequency)
                    for count, frequency in (
                        view.loc[
                            view["evaluation_cohort"].eq(cohort),
                            "reported_unit_count",
                        ]
                        .value_counts()
                        .sort_index()
                        .items()
                    )
                }
                for cohort in sorted(allowed_cohorts)
            },
        }

        print(f"  Saved {len(view)} observations.")

    if before != fingerprints():
        raise RuntimeError("An input changed while preparing validation views")

    report = {
        "issue": 48,
        "status": "shared_validation_views_prepared_no_predictions",
        "seed": 42,
        "input_sha256": before,
        "cohort_group_counts": cohort_counts,
        "cohort_condition_support": cohort_support,
        "views": views,
        "notes": [
            "Every model must use these same saved observation views.",
            "Each view contains one observation per validation source group.",
            "Views are separate evaluation settings, not independent patients.",
            "V2 receives its supported feature projection of each same view.",
            "Calibration-reserved and fitting groups are not included.",
            "No final-test records or model predictions were inspected.",
        ],
    }

    (OUTPUT / "view_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSHARED VALIDATION VIEWS READY")
    for cohort, count in sorted(cohort_counts.items()):
        print(f"  {cohort}: {count}")
    print("Views:", ", ".join(VIEW_LIMITS))
    print("Input fingerprints unchanged.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()