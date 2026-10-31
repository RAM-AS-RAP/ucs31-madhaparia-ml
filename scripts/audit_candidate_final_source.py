"""Audit final-evaluation sources without loading models or predicting."""

import ast
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_overlap import (
    add_record,
    input_signature,
    new_index,
    summarize_index,
)
from scripts.audit_candidate_source_records import file_hash, valid_demographics
from scripts.candidate_feature_builder import (
    derive_candidate_state,
    get_reporting_units,
    load_candidate_schema,
)


OUTPUT = ROOT / "reports/candidate_final_evaluation/source_audit.json"


@dataclass(slots=True)
class SourceGroup:
    first_source_row: int
    source_label: str
    canonical_label: str
    candidate_key: bytes
    reporting_units: int
    source_rows: int = 1


def scan_source(path, schema, label_mapping):
    """Index all valid in-scope inputs, including zero-supported inputs."""
    path = Path(path)
    csv.field_size_limit(2_147_483_647)

    if not label_mapping or len(set(label_mapping.values())) != len(label_mapping):
        raise ValueError("Expected a non-empty one-to-one label mapping")

    groups = {}
    totals = Counter({
        "all_source_rows": 0,
        "outside_scope_rows": 0,
        "in_scope_rows": 0,
        "no_supported_positive_rows": 0,
        "eligible_rows": 0,
    })
    outside_labels = Counter()
    per_condition = {
        label: Counter({
            "in_scope_rows": 0,
            "no_supported_positive_rows": 0,
            "eligible_rows": 0,
        })
        for label in sorted(label_mapping.values())
    }

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"AGE", "SEX", "PATHOLOGY", "EVIDENCES"}

        if (
            not required.issubset(reader.fieldnames or [])
            or len(reader.fieldnames) != len(set(reader.fieldnames))
        ):
            raise ValueError(f"Missing or duplicate columns: {path.name}")

        for row_number, row in enumerate(reader, start=1):
            totals["all_source_rows"] += 1
            source_label = row["PATHOLOGY"]

            if source_label not in label_mapping:
                totals["outside_scope_rows"] += 1
                outside_labels[source_label] += 1
                continue

            canonical_label = label_mapping[source_label]
            totals["in_scope_rows"] += 1
            per_condition[canonical_label]["in_scope_rows"] += 1

            if not valid_demographics(row):
                raise ValueError(
                    f"Invalid demographics: {path.name}, source row {row_number}"
                )

            try:
                evidence = ast.literal_eval(row["EVIDENCES"])
                state = derive_candidate_state(evidence, schema)
                units = len(get_reporting_units(state, schema))
            except (ValueError, TypeError, SyntaxError) as error:
                raise ValueError(
                    f"Invalid evidence: {path.name}, source row {row_number}: {error}"
                ) from error

            if state["absent"]:
                raise ValueError("Source derivation unexpectedly created absences")

            age = float(row["AGE"])
            sex = row["SEX"]
            source_key = input_signature(age, sex, evidence)
            candidate_key = input_signature(age, sex, state["present"])

            category = (
                "eligible_rows" if units else "no_supported_positive_rows"
            )
            totals[category] += 1
            per_condition[canonical_label][category] += 1

            if source_key in groups:
                group = groups[source_key]

                if group.canonical_label != canonical_label:
                    raise ValueError(
                        "Conflicting labels within normalized source group: "
                        f"{path.name}, rows {group.first_source_row} and {row_number}"
                    )

                if (
                    group.candidate_key != candidate_key
                    or group.reporting_units != units
                ):
                    raise ValueError("Equivalent source inputs derived differently")

                group.source_rows += 1
            else:
                groups[source_key] = SourceGroup(
                    first_source_row=row_number,
                    source_label=source_label,
                    canonical_label=canonical_label,
                    candidate_key=candidate_key,
                    reporting_units=units,
                )

    eligible = {
        key for key, group in groups.items() if group.reporting_units > 0
    }

    for label, counts in per_condition.items():
        counts["unique_eligible_groups"] = sum(
            groups[key].canonical_label == label for key in eligible
        )
        counts["unique_zero_supported_groups"] = sum(
            group.canonical_label == label and group.reporting_units == 0
            for group in groups.values()
        )

    summary = {
        "counts": {
            **dict(totals),
            "unique_in_scope_groups": len(groups),
            "unique_eligible_groups": len(eligible),
            "unique_zero_supported_groups": len(groups) - len(eligible),
            "repeated_in_scope_groups": sum(
                group.source_rows > 1 for group in groups.values()
            ),
            "extra_in_scope_rows_beyond_one_per_group": (
                totals["in_scope_rows"] - len(groups)
            ),
        },
        "outside_scope_rows_by_source_label": dict(sorted(outside_labels.items())),
        "by_canonical_condition": {
            label: dict(counts) for label, counts in per_condition.items()
        },
    }
    return groups, summary


def require_consistent_labels(first, second, description):
    shared = first.keys() & second.keys()
    conflicts = [
        key for key in shared
        if first[key].canonical_label != second[key].canonical_label
    ]
    if conflicts:
        raise ValueError(
            f"Conflicting labels across {description}: {len(conflicts)} groups"
        )


def define_test_cohorts(train, validation, test):
    """Use original source-pool matches, not reduced-feature matches."""
    require_consistent_labels(train, validation, "training and validation")
    require_consistent_labels(train, test, "training and test")
    require_consistent_labels(validation, test, "validation and test")

    eligible = {
        key for key, group in test.items() if group.reporting_units > 0
    }
    training_matches = eligible & train.keys()
    validation_matches = eligible & validation.keys()
    overlap = training_matches | validation_matches

    cohorts = {
        "primary": eligible - overlap,
        "complete_eligible": eligible,
        "overlap": overlap,
    }
    details = {
        "training_matches": len(training_matches),
        "validation_matches": len(validation_matches),
        "matches_both": len(training_matches & validation_matches),
        "training_only_matches": len(training_matches - validation_matches),
        "validation_only_matches": len(validation_matches - training_matches),
    }

    if cohorts["primary"] & overlap:
        raise AssertionError("Primary and overlap cohorts intersect")
    if cohorts["primary"] | overlap != eligible:
        raise AssertionError("Cohorts do not partition eligible test groups")

    return cohorts, details


def summarize_cohort(groups, selected, labels, earlier_candidate_keys=None):
    index = new_index()
    support = Counter()
    units = Counter()
    source_rows = 0
    candidate_matches = 0

    for key in selected:
        group = groups[key]
        if group.reporting_units <= 0:
            raise ValueError("Zero-supported group entered an eligible cohort")

        add_record(index, group.candidate_key, group.canonical_label)
        support[group.canonical_label] += 1
        units[group.reporting_units] += 1
        source_rows += group.source_rows

        if (
            earlier_candidate_keys is not None
            and group.candidate_key in earlier_candidate_keys
        ):
            candidate_matches += 1

    result = {
        "source_groups": len(selected),
        "original_source_rows_represented": source_rows,
        "per_condition_support": {label: support[label] for label in sorted(labels)},
        "actual_reporting_unit_counts": {
            str(count): frequency for count, frequency in sorted(units.items())
        },
        "candidate_inputs_after_source_deduplication": summarize_index(index),
    }

    if earlier_candidate_keys is not None:
        result["source_groups_matching_earlier_candidate_inputs"] = candidate_matches

    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError("Final source audit already exists.")

    sources = {
        "train": ROOT / "data/raw/release_train_patients",
        "validation": ROOT / "data/raw/release_validate_patients",
        "test": ROOT / "data/raw/release_test_patients",
    }
    paths = {
        **{f"{name}_source": path for name, path in sources.items()},
        "dataset_manifest": ROOT / "data/candidate_v1/manifest.json",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "protocol": ROOT / "docs/CANDIDATE_FINAL_EVALUATION_PROTOCOL.md",
        "candidate_feature_builder": ROOT / "scripts/candidate_feature_builder.py",
        "source_audit_helpers": ROOT / "scripts/audit_candidate_source_records.py",
        "overlap_audit_helpers": ROOT / "scripts/audit_candidate_overlap.py",
        "policy": (
            ROOT / "data/candidate_experiment_v1/calibration"
            / "uncertainty_policy/policy.json"
        ),
        "model": (
            ROOT / "data/candidate_experiment_v1/models"
            / "rf_group_weighted/model.joblib"
        ),
        "audit_script": Path(__file__).resolve(),
    }

    def fingerprints():
        return {name: file_hash(path) for name, path in paths.items()}

    print("Checking source and frozen-input fingerprints...", flush=True)
    before = fingerprints()

    dataset = json.loads(paths["dataset_manifest"].read_text(encoding="utf-8"))
    policy = json.loads(paths["policy"].read_text(encoding="utf-8"))
    schema = load_candidate_schema()
    label_mapping = dataset["label_mapping"]

    if schema != dataset["schema"]:
        raise ValueError("Candidate schema differs from prepared dataset")
    if (
        len(label_mapping) != 21
        or len(set(label_mapping.values())) != 21
        or set(label_mapping.values()) != set(policy["classes"])
    ):
        raise ValueError("Candidate condition scope differs")
    if label_mapping.get("Allergic sinusitis") != "Allergic rhinitis":
        raise ValueError("Expected allergy label mapping is missing")

    for name in (
        "train_source",
        "validation_source",
        "candidate_feature_builder",
        "source_audit_helpers",
        "overlap_audit_helpers",
    ):
        if before[name] != dataset["input_sha256"][name]:
            raise ValueError(f"Dataset dependency changed: {name}")

    if before["model"] != policy["model_sha256"]:
        raise ValueError("Frozen candidate model changed")
    if before["candidate_schema"] != policy["candidate_schema_sha256"]:
        raise ValueError("Frozen candidate schema changed")

    indexes = {}
    summaries = {}

    for split, path in sources.items():
        print(f"Auditing {split} source inputs...", flush=True)
        indexes[split], summaries[split] = scan_source(
            path, schema, label_mapping
        )
        print(" ", summaries[split]["counts"], flush=True)

    cohorts, overlap = define_test_cohorts(
        indexes["train"], indexes["validation"], indexes["test"]
    )

    earlier_candidate_keys = {
        group.candidate_key
        for split in ("train", "validation")
        for group in indexes[split].values()
        if group.reporting_units > 0
    }

    labels = list(label_mapping.values())
    cohort_summaries = {
        name: summarize_cohort(
            indexes["test"], selected, labels, earlier_candidate_keys
        )
        for name, selected in cohorts.items()
    }

    for split in ("train", "validation"):
        selected = {
            key for key, group in indexes[split].items()
            if group.reporting_units > 0
        }
        summaries[split]["eligible_groups_after_source_deduplication"] = (
            summarize_cohort(indexes[split], selected, labels)
        )

    if before != fingerprints():
        raise RuntimeError("An input changed during the source audit")

    report = {
        "issue": 50,
        "status": "source_audit_completed_no_predictions",
        "source_sha256": before,
        "label_mapping": label_mapping,
        "splits": summaries,
        "eligible_test_overlap": overlap,
        "test_cohorts": cohort_summaries,
        "validation": {
            "in_scope_demographics_valid": True,
            "in_scope_evidence_derivation_succeeded": True,
            "source_group_labels_consistent_within_and_across_splits": True,
        },
        "notes": [
            "The official test source was inspected; no predictions were calculated.",
            "Original in-scope training and validation pools define overlap.",
            "Earlier source indexes include zero-supported inputs.",
            "Eligible test cohorts require supported positive concepts.",
            "One representative per source group: first eligible source row.",
            "Candidate-input collisions are retained, including conflicting labels.",
            "Signature grouping does not establish patient identity.",
            "Source-pool overlap does not prove use in model fitting.",
            "Prior project-level test exposure is documented in the protocol.",
            "No source datasets, models, calibration parameters or policies changed.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL SOURCE AUDIT COMPLETE")
    for name, summary in cohort_summaries.items():
        print(f"  {name}: {summary['source_groups']} eligible test groups")
    print("Eligible test overlap:", overlap)
    print("Input fingerprints unchanged.")
    print("No model loaded, trained or used for predictions.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()