"""Assign candidate source groups to fitting, reserve and validation cohorts."""

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash


DATA = ROOT / "data/candidate_v1"
OUTPUT = ROOT / "data/candidate_experiment_v1"
PROTOCOL = ROOT / "docs/CANDIDATE_MODEL_EXPERIMENT_PROTOCOL.md"

COLUMNS = [
    "source_group_id",
    "source_split",
    "canonical_label",
    "partition",
]


def assign_partitions(groups, legacy_validation_ids, seed=42):
    """Assign whole source-input groups without inspecting model predictions."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    by_id = {}
    training_by_label = defaultdict(list)
    validation_ids = set()

    for group in groups:
        group_id = group["source_group_id"]
        split = group["split"]
        label = group["canonical_label"]

        if (
            not isinstance(group_id, str)
            or len(group_id) != 64
            or any(character not in "0123456789abcdef" for character in group_id)
        ):
            raise ValueError("Invalid source-group signature")

        if group_id in by_id:
            raise ValueError("Duplicate source-group ID")

        if split not in {"train", "validation"}:
            raise ValueError("Unexpected source split")

        if not isinstance(label, str) or not label.strip():
            raise ValueError("Invalid canonical label")

        by_id[group_id] = group

        if split == "train":
            training_by_label[label].append(group_id)
        else:
            validation_ids.add(group_id)

    legacy_validation_ids = set(legacy_validation_ids)

    if not legacy_validation_ids.issubset(validation_ids):
        raise ValueError("Legacy-overlap IDs must belong to validation")

    if not training_by_label or not validation_ids:
        raise ValueError("Both training and validation groups are required")

    def ordering_key(group_id):
        payload = f"{seed}:calibration_holdout:{group_id}"
        return (
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            group_id,
        )

    reserved = set()

    for label in sorted(training_by_label):
        ids = training_by_label[label]
        if len(ids) < 2:
            raise ValueError(
                f"At least two training groups required for {label}"
            )

        # Integer division implements floor(20% of this class).
        reserve_count = max(1, len(ids) // 5)
        reserve_count = min(reserve_count, len(ids) - 1)

        reserved.update(
            sorted(ids, key=ordering_key)[:reserve_count]
        )

    assignments = []

    for group_id, group in by_id.items():
        split = group["split"]

        if split == "train":
            partition = (
                "calibration_reserved" if group_id in reserved else "fit"
            )
        else:
            partition = (
                "selection_legacy_overlap"
                if group_id in legacy_validation_ids
                else "selection_primary"
            )

        assignments.append({
            "source_group_id": group_id,
            "source_split": split,
            "canonical_label": group["canonical_label"],
            "partition": partition,
        })

    return sorted(
        assignments,
        key=lambda row: (row["source_split"], row["source_group_id"]),
    )


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT}")

    source_paths = {
        "dataset_manifest": DATA / "manifest.json",
        "source_groups": DATA / "source_groups.csv",
        "excluded_rows": DATA / "excluded_rows.csv",
        "protocol": PROTOCOL,
        "partition_script": Path(__file__).resolve(),
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
    }

    def fingerprints():
        return {
            name: file_hash(path)
            for name, path in source_paths.items()
        }

    before = fingerprints()

    dataset_manifest = json.loads(
        source_paths["dataset_manifest"].read_text(encoding="utf-8")
    )

    for key, filename in (
        ("source_groups", "source_groups.csv"),
        ("excluded_rows", "excluded_rows.csv"),
    ):
        if before[key] != dataset_manifest["output_sha256"][filename]:
            raise ValueError(f"Dataset fingerprint mismatch: {filename}")

    with source_paths["source_groups"].open(
        encoding="utf-8", newline=""
    ) as source:
        groups = list(csv.DictReader(source))

    for split in ("train", "validation"):
        actual = Counter(
            row["canonical_label"]
            for row in groups if row["split"] == split
        )
        expected = dataset_manifest["splits"][split][
            "groups_by_canonical_label"
        ]
        if dict(actual) != expected:
            raise ValueError(f"Source-group counts differ for {split}")

    legacy_ids = set()
    with source_paths["excluded_rows"].open(
        encoding="utf-8", newline=""
    ) as source:
        for row in csv.DictReader(source):
            if (
                row["split"] == "train"
                and row["reason"] == "source_group_reserved_for_validation"
            ):
                legacy_ids.add(row["source_group_id"])

    if len(legacy_ids) != dataset_manifest[
        "shared_source_groups_removed_from_training"
    ]:
        raise ValueError("Legacy-overlap group count differs from manifest")

    assignments = assign_partitions(groups, legacy_ids, seed=42)

    counts = Counter(row["partition"] for row in assignments)
    per_class = defaultdict(Counter)
    for row in assignments:
        per_class[row["partition"]][row["canonical_label"]] += 1

    if before != fingerprints():
        raise RuntimeError("An input changed during partition preparation")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    partition_path = OUTPUT / "partitions.csv"

    with partition_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(assignments)

    report = {
        "issue": 48,
        "status": "partitions_prepared_no_model_trained",
        "seed": 42,
        "reserve_fraction": "1/5 per class, rounded down; minimum 1",
        "ordering": (
            "SHA256 of seed:calibration_holdout:source_group_id; "
            "source_group_id breaks any hash tie"
        ),
        "input_sha256": before,
        "partitions_sha256": file_hash(partition_path),
        "group_counts": dict(sorted(counts.items())),
        "groups_by_partition_and_class": {
            partition: dict(sorted(values.items()))
            for partition, values in sorted(per_class.items())
        },
        "notes": [
            "All variants inherit their source group's partition.",
            "Calibration-reserved groups must not enter model fitting.",
            "Legacy overlap refers to the original source training pool.",
            "No prediction performance was used for partition assignment.",
            "The final test split was not inspected.",
            "The prepared candidate dataset was not modified.",
        ],
    }

    (OUTPUT / "partition_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("CANDIDATE EXPERIMENT PARTITIONS")
    for partition, count in sorted(counts.items()):
        print(f"  {partition}: {count}")
    print("Input fingerprints unchanged.")
    print("No model trained.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()