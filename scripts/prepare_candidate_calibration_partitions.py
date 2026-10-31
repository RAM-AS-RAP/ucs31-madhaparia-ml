"""Partition reserved source groups for calibration and uncertainty work."""

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


EXPERIMENT = ROOT / "data/candidate_experiment_v1"
OUTPUT = EXPERIMENT / "calibration"
PROTOCOL = ROOT / "docs/CANDIDATE_CALIBRATION_PROTOCOL.md"

PARTITIONS = (
    "calibration_fit",
    "policy_development",
    "uncertainty_evaluation",
)

COLUMNS = [
    "source_group_id",
    "source_split",
    "canonical_label",
    "parent_partition",
    "partition",
]


def assign_calibration_partitions(reserved_groups):
    """Assign only reserved groups, keeping each group in one partition."""

    by_label = defaultdict(list)
    seen = set()

    for row in reserved_groups:
        group_id = row.get("source_group_id")
        label = row.get("canonical_label")

        if (
            not isinstance(group_id, str)
            or len(group_id) != 64
            or any(char not in "0123456789abcdef" for char in group_id)
        ):
            raise ValueError("Invalid source-group signature")

        if group_id in seen:
            raise ValueError("Duplicate source-group ID")

        if (
            row.get("source_split") != "train"
            or row.get("partition") != "calibration_reserved"
        ):
            raise ValueError("Only calibration_reserved training groups allowed")

        if not isinstance(label, str) or not label.strip():
            raise ValueError("Invalid canonical label")

        seen.add(group_id)
        by_label[label].append(group_id)

    if not seen:
        raise ValueError("No reserved groups supplied")

    def ordering_key(group_id):
        payload = f"49:calibration_partition:{group_id}"
        return (
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            group_id,
        )

    assignments = []

    for label, group_ids in sorted(by_label.items()):
        ordered = sorted(group_ids, key=ordering_key)
        count = len(ordered)

        # Exact integer arithmetic implements the protocol's floor rules.
        fit_count = 3 * count // 5
        development_count = count // 5
        evaluation_count = count - fit_count - development_count

        if min(fit_count, development_count, evaluation_count) < 1:
            raise ValueError(
                f"Insufficient reserved groups for all three partitions: {label}"
            )

        development_end = fit_count + development_count

        for index, group_id in enumerate(ordered):
            if index < fit_count:
                partition = "calibration_fit"
            elif index < development_end:
                partition = "policy_development"
            else:
                partition = "uncertainty_evaluation"

            assignments.append({
                "source_group_id": group_id,
                "source_split": "train",
                "canonical_label": label,
                "parent_partition": "calibration_reserved",
                "partition": partition,
            })

    return sorted(assignments, key=lambda row: row["source_group_id"])


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT}")

    paths = {
        "protocol": PROTOCOL,
        "parent_partitions": EXPERIMENT / "partitions.csv",
        "parent_manifest": EXPERIMENT / "partition_manifest.json",
        "archived_parent_manifest": (
            ROOT / "reports/candidate_model_comparison/partition_manifest.json"
        ),
        "partition_script": Path(__file__).resolve(),
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
    }

    def fingerprints():
        return {name: file_hash(path) for name, path in paths.items()}

    before = fingerprints()

    if before["parent_manifest"] != before["archived_parent_manifest"]:
        raise ValueError("Parent manifest differs from the issue #48 archive")

    parent_manifest = json.loads(
        paths["parent_manifest"].read_text(encoding="utf-8")
    )

    if (
        before["parent_partitions"]
        != parent_manifest["partitions_sha256"]
    ):
        raise ValueError("Parent partition fingerprint mismatch")

    with paths["parent_partitions"].open(
        encoding="utf-8", newline=""
    ) as source:
        reader = csv.DictReader(source)
        expected_columns = {
            "source_group_id",
            "source_split",
            "canonical_label",
            "partition",
        }
        if (
            len(reader.fieldnames or []) != len(expected_columns)
            or set(reader.fieldnames or []) != expected_columns
        ):
            raise ValueError("Unexpected parent partition columns")

        parent_rows = list(reader)

    parent_ids = [row["source_group_id"] for row in parent_rows]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Duplicate IDs in parent partitions")

    allowed_splits = {
        "fit": "train",
        "calibration_reserved": "train",
        "selection_primary": "validation",
        "selection_legacy_overlap": "validation",
    }

    parent_counts = Counter()
    parent_per_class = defaultdict(Counter)

    for row in parent_rows:
        partition = row["partition"]

        if partition not in allowed_splits:
            raise ValueError("Unknown parent partition")

        if row["source_split"] != allowed_splits[partition]:
            raise ValueError("Parent partition has an unexpected source split")

        parent_counts[partition] += 1
        parent_per_class[partition][row["canonical_label"]] += 1

    if dict(parent_counts) != parent_manifest["group_counts"]:
        raise ValueError("Parent group counts differ from their manifest")

    actual_per_class = {
        partition: dict(counts)
        for partition, counts in parent_per_class.items()
    }
    if (
        actual_per_class
        != parent_manifest["groups_by_partition_and_class"]
    ):
        raise ValueError("Parent per-condition counts differ from their manifest")

    reserved = [
        row for row in parent_rows
        if row["partition"] == "calibration_reserved"
    ]

    if len(reserved) != 83635:
        raise ValueError("Expected 83,635 reserved source groups")

    if len({row["canonical_label"] for row in reserved}) != 21:
        raise ValueError("Expected 21 reserved conditions")

    assignments = assign_calibration_partitions(reserved)

    reserved_ids = {row["source_group_id"] for row in reserved}
    assigned_ids = {row["source_group_id"] for row in assignments}

    if len(assignments) != len(reserved) or assigned_ids != reserved_ids:
        raise ValueError("Assignments do not exactly cover the reserved groups")

    other_ids = {
        row["source_group_id"]
        for row in parent_rows
        if row["partition"] != "calibration_reserved"
    }
    if assigned_ids & other_ids:
        raise ValueError("Assignment overlaps fitting or validation groups")

    counts = Counter(row["partition"] for row in assignments)
    per_class = defaultdict(Counter)

    for row in assignments:
        per_class[row["partition"]][row["canonical_label"]] += 1

    if before != fingerprints():
        raise RuntimeError("An input changed during partition preparation")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    partition_path = OUTPUT / "partitions.csv"

    with partition_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(assignments)

    report = {
        "issue": 49,
        "status": "partitions_prepared_no_predictions_inspected",
        "python_version": sys.version,
        "ordering": (
            "SHA256 of 49:calibration_partition:source_group_id; "
            "source_group_id breaks any hash tie"
        ),
        "allocation": {
            "calibration_fit": "floor(3 * class_group_count / 5)",
            "policy_development": "floor(class_group_count / 5)",
            "uncertainty_evaluation": "remaining groups",
        },
        "input_sha256": before,
        "partitions_sha256": file_hash(partition_path),
        "group_counts": {
            name: counts[name] for name in PARTITIONS
        },
        "groups_by_partition_and_class": {
            name: dict(sorted(per_class[name].items()))
            for name in PARTITIONS
        },
        "overlap_with_fit_or_validation_groups": 0,
        "notes": [
            "Every reserved source group appears exactly once.",
            "All future views must inherit their source group's partition.",
            "No model or prediction archive was loaded.",
            "No calibration parameters or thresholds were fitted.",
            "The final test split was not inspected.",
        ],
    }

    (OUTPUT / "partition_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("CALIBRATION AND UNCERTAINTY PARTITIONS")
    for name in PARTITIONS:
        print(f"  {name}: {counts[name]}")

    print("Conditions in each partition: 21")
    print("Overlap with fitting or validation groups: 0")
    print("Input fingerprints unchanged.")
    print("No predictions inspected and no model trained.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()