"""Prepare final-evaluation observations without calculating predictions."""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_final_source import (
    define_test_cohorts,
    scan_source,
    summarize_cohort,
)
from scripts.audit_candidate_source_records import file_hash
from scripts.build_candidate_dataset import numerical_values, read_source
from scripts.candidate_evaluation import VIEW_LIMITS, make_validation_view
from scripts.candidate_feature_builder import get_reporting_units, load_candidate_schema


EXPERIMENT = ROOT / "data/candidate_experiment_v1"
OUTPUT = EXPERIMENT / "final_test/views"

METADATA = [
    "source_group_id",
    "variant_id",
    "canonical_label",
    "evaluation_cohort",
    "source_file",
    "source_row",
    "source_label",
    "source_group_size",
    "available_reporting_units",
    "reported_unit_count",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_final_full_frames(test_path, schema, label_mapping, groups, cohorts):
    """Return eligible and zero-supported full observations with provenance."""
    expected_names = {"primary", "complete_eligible", "overlap"}
    if set(cohorts) != expected_names:
        raise ValueError("Unexpected cohort definitions")

    eligible = {
        key for key, group in groups.items() if group.reporting_units > 0
    }

    if cohorts["complete_eligible"] != eligible:
        raise ValueError("Complete cohort differs from eligible source groups")
    if cohorts["primary"] & cohorts["overlap"]:
        raise ValueError("Primary and overlap cohorts intersect")
    if cohorts["primary"] | cohorts["overlap"] != eligible:
        raise ValueError("Primary and overlap do not partition eligible groups")

    observations = []
    zero_supported = []
    observed_counts = Counter()
    retained = set()

    for record in read_source(test_path, schema, label_mapping):
        key = record["signature"]
        if key not in groups:
            raise ValueError("Source contains an unaudited group")

        group = groups[key]
        observed_counts[key] += 1

        if (
            record["source_label"] != group.source_label
            or label_mapping[record["source_label"]] != group.canonical_label
        ):
            raise ValueError("Source label differs from audited group")

        if record["source_row"] != group.first_source_row:
            continue

        if key in retained:
            raise ValueError("Duplicate retained source group")
        retained.add(key)

        units = len(get_reporting_units(record["state"], schema))
        if units != group.reporting_units:
            raise ValueError("Reporting-unit count differs from source audit")

        values = numerical_values(
            record["age"], record["sex"], record["state"], schema
        )

        cohort = (
            "zero_supported"
            if units == 0
            else "primary" if key in cohorts["primary"] else "overlap"
        )

        row = {
            "source_group_id": key.hex(),
            "variant_id": "full",
            "canonical_label": group.canonical_label,
            "evaluation_cohort": cohort,
            "source_file": Path(test_path).name,
            "source_row": group.first_source_row,
            "source_label": group.source_label,
            "source_group_size": group.source_rows,
            "available_reporting_units": units,
            "reported_unit_count": units,
            **dict(zip(schema["feature_names"], values)),
        }

        if units:
            observations.append(row)
        else:
            zero_supported.append(row)

    if retained != set(groups):
        raise ValueError("Missing audited source-group representatives")

    for key, group in groups.items():
        if observed_counts[key] != group.source_rows:
            raise ValueError("Source-group size differs from audit")

    columns = METADATA + list(schema["feature_names"])
    full = pd.DataFrame(observations, columns=columns)
    zero = pd.DataFrame(zero_supported, columns=columns)

    for frame in (full, zero):
        if not frame["source_group_id"].is_unique:
            raise ValueError("Duplicate observation IDs")

    if len(full) != len(eligible):
        raise ValueError("Incorrect number of eligible observations")

    return full, zero


def main():
    if OUTPUT.exists():
        raise FileExistsError("Final observation-view directory already exists.")

    sources = {
        "train": ROOT / "data/raw/release_train_patients",
        "validation": ROOT / "data/raw/release_validate_patients",
        "test": ROOT / "data/raw/release_test_patients",
    }

    inputs = {
        **{f"{name}_source": path for name, path in sources.items()},
        "dataset_manifest": ROOT / "data/candidate_v1/manifest.json",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "protocol": ROOT / "docs/CANDIDATE_FINAL_EVALUATION_PROTOCOL.md",
        "candidate_feature_builder": ROOT / "scripts/candidate_feature_builder.py",
        "source_audit_helpers": ROOT / "scripts/audit_candidate_source_records.py",
        "overlap_audit_helpers": ROOT / "scripts/audit_candidate_overlap.py",
        "policy": EXPERIMENT / "calibration/uncertainty_policy/policy.json",
        "model": EXPERIMENT / "models/rf_group_weighted/model.joblib",
        "audit_script": ROOT / "scripts/audit_candidate_final_source.py",
        "source_audit": ROOT / "reports/candidate_final_evaluation/source_audit.json",
        "dataset_builder": ROOT / "scripts/build_candidate_dataset.py",
        "evaluation_helper": ROOT / "scripts/candidate_evaluation.py",
        "previous_view_manifest": (
            ROOT / "reports/candidate_model_comparison/view_manifest.json"
        ),
        "view_preparation_script": Path(__file__).resolve(),
    }

    def fingerprints():
        return {name: file_hash(path) for name, path in inputs.items()}

    print("Checking source-audit and preparation fingerprints...", flush=True)
    before = fingerprints()

    audit = read_json(inputs["source_audit"])
    dataset = read_json(inputs["dataset_manifest"])
    previous_views = read_json(inputs["previous_view_manifest"])
    schema = load_candidate_schema()
    label_mapping = dataset["label_mapping"]

    if audit["status"] != "source_audit_completed_no_predictions":
        raise ValueError("Unexpected source-audit status")

    for name, expected in audit["source_sha256"].items():
        if name not in before or before[name] != expected:
            raise ValueError(f"Source-audit dependency changed: {name}")

    if before["dataset_builder"] != dataset["input_sha256"]["dataset_builder"]:
        raise ValueError("Dataset encoding helper changed")

    if (
        before["evaluation_helper"]
        != previous_views["input_sha256"]["evaluation_helper"]
    ):
        raise ValueError("Observation masking helper changed")

    if schema != dataset["schema"] or label_mapping != audit["label_mapping"]:
        raise ValueError("Schema or condition mapping changed")

    indexes = {}
    for split, path in sources.items():
        print(f"Reconstructing {split} source groups...", flush=True)
        groups, summary = scan_source(path, schema, label_mapping)
        indexes[split] = groups

        for section in (
            "counts",
            "outside_scope_rows_by_source_label",
            "by_canonical_condition",
        ):
            if summary[section] != audit["splits"][split][section]:
                raise ValueError(f"Source-audit counts differ: {split}/{section}")

    cohorts, overlap = define_test_cohorts(
        indexes["train"], indexes["validation"], indexes["test"]
    )
    if overlap != audit["eligible_test_overlap"]:
        raise ValueError("Source overlap differs from audit")

    earlier_candidate_keys = {
        group.candidate_key
        for split in ("train", "validation")
        for group in indexes[split].values()
        if group.reporting_units > 0
    }

    for name, selected in cohorts.items():
        summary = summarize_cohort(
            indexes["test"], selected,
            list(label_mapping.values()), earlier_candidate_keys,
        )
        if summary != audit["test_cohorts"][name]:
            raise ValueError(f"Test cohort differs from audit: {name}")

    print("Encoding full test observations...", flush=True)
    full, zero = build_final_full_frames(
        sources["test"], schema, label_mapping, indexes["test"], cohorts
    )
    if full.empty:
        raise ValueError("No eligible test groups; evaluation is not assessable")

    if len(zero) != audit["splits"]["test"]["counts"]["unique_zero_supported_groups"]:
        raise ValueError("Zero-supported group count differs from audit")

    cohort_counts = {
        name: len(selected) for name, selected in cohorts.items()
    }

    del indexes, earlier_candidate_keys, cohorts

    if before != fingerprints():
        raise RuntimeError("An input changed during observation preparation")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    view_details = {}

    def save_csv(filename, frame):
        path = OUTPUT / filename
        frame.to_csv(path, index=False, lineterminator="\n")
        output_hashes[filename] = file_hash(path)

    save_csv("groups.csv", full[METADATA])
    save_csv("zero_supported.csv", zero)

    for setting in VIEW_LIMITS:
        print(f"Preparing {setting}...", flush=True)
        frame = make_validation_view(full, schema, setting, seed=42)

        if not frame["source_group_id"].equals(full["source_group_id"]):
            raise ValueError("Masking changed source-group order")
        if not frame["canonical_label"].equals(full["canonical_label"]):
            raise ValueError("Masking changed labels")
        if not frame["evaluation_cohort"].equals(full["evaluation_cohort"]):
            raise ValueError("Masking changed cohorts")

        filename = f"{setting}.csv"
        save_csv(filename, frame)

        view_details[setting] = {
            "filename": filename,
            "rows": len(frame),
            "sha256": output_hashes[filename],
            "requested_maximum_reporting_units": VIEW_LIMITS[setting],
            "actual_reporting_unit_counts": {
                str(int(count)): int(frequency)
                for count, frequency in sorted(
                    frame["reported_unit_count"].value_counts().items()
                )
            },
        }
        print(f"  Saved {len(frame):,} observations.", flush=True)

    if before != fingerprints():
        raise RuntimeError(
            "An input changed; outputs are incomplete and have no valid manifest"
        )

    manifest = {
        "issue": 50,
        "status": "final_observation_views_prepared_no_predictions",
        "seed": 42,
        "schema_id": schema["schema_id"],
        "feature_names": schema["feature_names"],
        "classes": sorted(label_mapping.values()),
        "source": sources["test"].name,
        "cohort_counts": cohort_counts,
        "cohort_support": {
            name: audit["test_cohorts"][name]["per_condition_support"]
            for name in cohort_counts
        },
        "zero_supported_groups": len(zero),
        "input_sha256": before,
        "output_sha256": output_hashes,
        "views": view_details,
        "pandas_version": pd.__version__,
        "notes": [
            "Primary and overlap are mutually exclusive cohort assignments.",
            "Complete eligible results use both assignments together.",
            "Zero-supported observations are saved separately.",
            "Each view has one observation per eligible source group.",
            "Source row numbers count data rows after the CSV header.",
            "Partial views retain the full source provenance metadata.",
            "Available reporting units describe the full source observation.",
            "Reported unit counts describe each prepared view.",
            "Hidden features remain unknown; absent columns remain zero.",
            "Repeated views are not independent patients.",
            "No model was loaded or used to calculate predictions.",
            "Existing model, temperature and uncertainty policy are unchanged.",
        ],
    }

    (OUTPUT / "view_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL OBSERVATION VIEWS READY")
    for name, count in cohort_counts.items():
        print(f"  {name}: {count:,} groups")
    print("Zero-supported groups saved separately:", len(zero))
    print("Views:", ", ".join(VIEW_LIMITS))
    print("Input fingerprints unchanged.")
    print("No predictions calculated.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()