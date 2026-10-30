"""Audit normalized source-input and candidate-input overlap.

Does not train models, change data, or inspect the final test split.
Matching input signatures do not establish patient identity.
"""

import ast
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import (
    file_hash,
    valid_demographics,
)
from scripts.candidate_feature_builder import (
    derive_candidate_state,
    load_candidate_schema,
)


SOURCES = {
    "train": ROOT / "data/raw/release_train_patients",
    "validation": ROOT / "data/raw/release_validate_patients",
}
OUTPUT = ROOT / "reports/candidate_source_audit/overlap_audit.json"


def input_signature(age, sex, codes):
    payload = json.dumps(
        [age, sex, sorted(set(codes))],
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def new_index():
    return {
        "counts": Counter(),
        "first_label": {},
        "conflicting_labels": set(),
    }


def add_record(index, signature, label):
    index["counts"][signature] += 1
    if signature not in index["first_label"]:
        index["first_label"][signature] = label
    elif index["first_label"][signature] != label:
        index["conflicting_labels"].add(signature)


def summarize_index(index):
    counts = index["counts"]
    conflicts = index["conflicting_labels"]

    return {
        "records": sum(counts.values()),
        "unique_input_groups": len(counts),
        "repeated_input_groups": sum(
            count > 1 for count in counts.values()
        ),
        "extra_rows_beyond_one_per_input_group": (
            sum(counts.values()) - len(counts)
        ),
        "groups_with_multiple_condition_labels": len(conflicts),
        "rows_in_multiple_label_groups": sum(
            counts[key] for key in conflicts
        ),
    }


def summarize_overlap(train, validation):
    shared = train["counts"].keys() & validation["counts"].keys()

    multiple_labels = {
        key
        for key in shared
        if (
            key in train["conflicting_labels"]
            or key in validation["conflicting_labels"]
            or train["first_label"][key]
            != validation["first_label"][key]
        )
    }

    return {
        "shared_input_groups": len(shared),
        "training_rows_in_shared_groups": sum(
            train["counts"][key] for key in shared
        ),
        "validation_rows_matching_training_inputs": sum(
            validation["counts"][key] for key in shared
        ),
        "shared_groups_with_multiple_condition_labels": len(
            multiple_labels
        ),
    }


def main():
    csv.field_size_limit(2_147_483_647)

    labels_path = ROOT / "data/processed/diseases.json"
    input_paths = [
        labels_path,
        ROOT / "docs/candidate_feature_schema_v1.json",
        ROOT / "scripts/candidate_feature_builder.py",
        ROOT / "scripts/audit_candidate_source_records.py",
        Path(__file__).resolve(),
        *SOURCES.values(),
    ]

    def fingerprints():
        return {
            path.relative_to(ROOT).as_posix(): file_hash(path)
            for path in input_paths
        }

    before = fingerprints()
    schema = load_candidate_schema()
    labels = {
        item["name"]
        for item in json.loads(
            labels_path.read_text(encoding="utf-8")
        )
    }
    assert len(labels) == 21

    indexes = {}
    split_summaries = {}

    for split, path in SOURCES.items():
        print(f"Reading {split} input signatures...", flush=True)

        raw_index = new_index()
        candidate_index = new_index()
        exclusions = Counter()
        eligible = Counter()
        raw_overlap_by_condition = Counter()
        candidate_overlap_by_condition = Counter()

        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"AGE", "SEX", "PATHOLOGY", "EVIDENCES"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing required columns: {path.name}")

            for row_number, row in enumerate(reader, start=1):
                label = row["PATHOLOGY"]
                if label not in labels:
                    exclusions["outside_scope"] += 1
                    continue

                if not valid_demographics(row):
                    raise ValueError(
                        f"Invalid demographics: {split} row {row_number}"
                    )

                evidence = ast.literal_eval(row["EVIDENCES"])
                state = derive_candidate_state(evidence, schema)

                if not state["present"]:
                    exclusions["no_supported_positive_concepts"] += 1
                    continue

                age = float(row["AGE"])
                sex = row["SEX"]

                raw_key = input_signature(age, sex, evidence)
                candidate_key = input_signature(
                    age, sex, state["present"]
                )

                add_record(raw_index, raw_key, label)
                add_record(candidate_index, candidate_key, label)
                eligible[label] += 1

                if split == "validation":
                    if raw_key in indexes["train"]["raw"]["counts"]:
                        raw_overlap_by_condition[label] += 1

                    if (
                        candidate_key
                        in indexes["train"]["candidate"]["counts"]
                    ):
                        candidate_overlap_by_condition[label] += 1

        indexes[split] = {
            "raw": raw_index,
            "candidate": candidate_index,
        }

        split_summaries[split] = {
            "exclusions": dict(exclusions),
            "eligible_rows_by_source_condition": {
                label: eligible[label] for label in sorted(labels)
            },
            "normalized_source_inputs": summarize_index(raw_index),
            "candidate_inputs_before_masking": summarize_index(
                candidate_index
            ),
        }

        if split == "validation":
            split_summaries[split]["overlap_by_source_condition"] = {
                label: {
                    "normalized_source_input_matches": (
                        raw_overlap_by_condition[label]
                    ),
                    "candidate_input_matches": (
                        candidate_overlap_by_condition[label]
                    ),
                }
                for label in sorted(labels)
            }

    overlap = {
        "normalized_source_inputs": summarize_overlap(
            indexes["train"]["raw"],
            indexes["validation"]["raw"],
        ),
        "candidate_inputs_before_masking": summarize_overlap(
            indexes["train"]["candidate"],
            indexes["validation"]["candidate"],
        ),
    }

    if before != fingerprints():
        raise RuntimeError("An input changed during the audit; rerun it.")

    report = {
        "issue": 47,
        "source_sha256": before,
        "splits": split_summaries,
        "train_validation_overlap": overlap,
        "signature_definition": {
            "normalized_source_inputs": (
                "Numeric age, recorded sex, sorted unique original "
                "evidence codes; excludes diagnosis and differential."
            ),
            "candidate_inputs_before_masking": (
                "Numeric age, recorded sex, sorted positive candidate "
                "concepts. Other concepts are unknown; none are absent."
            ),
        },
        "limitations": [
            "Signatures are input groups, not verified patient identifiers.",
            "Different patients may have identical inputs.",
            "Candidate collisions do not prove source-record duplication.",
            "Multiple labels may reflect ambiguity, not annotation errors.",
            "Related records with different inputs are not detected.",
            "Counts retain original source labels.",
            "Rows with no supported positive concepts are excluded.",
            "No partial-symptom examples have been generated.",
            "The final test split was not inspected.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for split, summary in split_summaries.items():
        print(f"\n{split.upper()}")
        for kind in (
            "normalized_source_inputs",
            "candidate_inputs_before_masking",
        ):
            print(f"  {kind}:")
            for name, value in summary[kind].items():
                print(f"    {name}: {value}")

    print("\nTRAIN-VALIDATION OVERLAP")
    for kind, values in overlap.items():
        print(f"  {kind}:")
        for name, value in values.items():
            print(f"    {name}: {value}")

    print("\nInput fingerprints unchanged.")
    print("No datasets or models changed.")
    print("Report saved to:", OUTPUT)


if __name__ == "__main__":
    main()