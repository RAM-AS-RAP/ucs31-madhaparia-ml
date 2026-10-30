"""Audit source records before candidate scenario generation."""

import ast
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_feature_builder import (
    derive_candidate_state,
    load_candidate_schema,
)


OUTPUT = ROOT / "reports/candidate_source_audit/source_audit.json"

SOURCES = {
    "train": ROOT / "data/raw/release_train_patients",
    "validation": ROOT / "data/raw/release_validate_patients",
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_demographics(row):
    try:
        age = float(row["AGE"])
    except (ValueError, TypeError):
        return False

    return (
        math.isfinite(age)
        and 0 <= age <= 120
        and row["SEX"] in {"M", "F"}
    )


def audit_split(path, labels, schema):
    totals = Counter()
    by_condition = {label: Counter() for label in sorted(labels)}
    feature_counts = Counter()
    positive_feature_count_histogram = Counter()
    issue_examples = []

    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"AGE", "SEX", "PATHOLOGY", "EVIDENCES"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns: {path.name}")

        for row_number, row in enumerate(reader, start=1):
            totals["all_source_rows"] += 1
            label = row["PATHOLOGY"]

            if label not in labels:
                totals["outside_scope_rows"] += 1
                continue

            per_class = by_condition[label]

            def record(name):
                totals[name] += 1
                per_class[name] += 1

            record("in_scope_rows")

            demographics_ok = valid_demographics(row)
            if not demographics_ok:
                record("invalid_demographics")

            try:
                evidence = ast.literal_eval(row["EVIDENCES"])
                state = derive_candidate_state(evidence, schema)
            except (ValueError, SyntaxError, TypeError) as error:
                record("evidence_derivation_errors")

                if len(issue_examples) < 25:
                    issue_examples.append({
                        "source_row": row_number,
                        "source_label": label,
                        "error": str(error),
                    })
                continue

            record("successfully_derived_rows")
            present = set(state["present"])

            assert not state["absent"]
            assert present.isdisjoint(state["unknown"])
            assert present | set(state["unknown"]) == set(
                schema["symptom_features"]
            )

            feature_counts.update(present)
            positive_feature_count_histogram[len(present)] += 1

            has_head = "head_region_pain" in present
            has_chest = "chest_region_pain" in present

            if has_head:
                record("head_region_present")
            if has_chest:
                record("chest_region_present")
            if has_head and has_chest:
                record("both_regions_present")

            if "E_53" in present and not (has_head or has_chest):
                record("general_pain_without_selected_region")

            if not present:
                record("no_supported_positive_concepts")
            elif demographics_ok:
                record("eligible_before_duplicate_review")

    return {
        "totals": dict(totals),
        "by_source_condition": {
            label: dict(values)
            for label, values in by_condition.items()
        },
        "positive_feature_counts": {
            code: feature_counts[code]
            for code in schema["symptom_features"]
        },
        "positive_feature_count_histogram": {
            str(count): frequency
            for count, frequency in sorted(
                positive_feature_count_histogram.items()
            )
        },
        "first_25_evidence_issue_examples": issue_examples,
    }


def main():
    csv.field_size_limit(2_147_483_647)

    labels_path = ROOT / "data/processed/diseases.json"
    input_paths = [
        labels_path,
        ROOT / "docs/candidate_feature_schema_v1.json",
        ROOT / "scripts/candidate_feature_builder.py",
        Path(__file__).resolve(),
        *SOURCES.values(),
    ]

    before = {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path)
        for path in input_paths
    }

    schema = load_candidate_schema()
    labels = {
        item["name"]
        for item in json.loads(labels_path.read_text(encoding="utf-8"))
    }
    assert len(labels) == 21

    results = {}
    for split, path in SOURCES.items():
        print(f"Auditing {split} source records...", flush=True)
        results[split] = audit_split(path, labels, schema)

    after = {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path)
        for path in input_paths
    }
    if before != after:
        raise RuntimeError("An input changed during the audit; rerun it.")

    report = {
        "issue": 47,
        "schema_id": schema["schema_id"],
        "source_sha256": before,
        "splits": results,
        "notes": [
            "Counts retain original source condition labels.",
            "Demographic and evidence issue counts may overlap.",
            "Feature coverage counts use successfully derived records.",
            "Feature counts are not counts of independent reported symptoms.",
            "General pain may occur outside the selected head/chest regions.",
            "Missing region evidence is unknown, not confirmed absence.",
            "Eligibility is provisional until duplicate and leakage review.",
            "This audit does not validate all source evidence definitions.",
            "The final test split was not inspected.",
            "No training dataset or model was changed.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for split, result in results.items():
        totals = result["totals"]
        print(f"\n{split.upper()}")
        for name in (
            "in_scope_rows",
            "successfully_derived_rows",
            "evidence_derivation_errors",
            "invalid_demographics",
            "no_supported_positive_concepts",
            "head_region_present",
            "chest_region_present",
            "both_regions_present",
            "general_pain_without_selected_region",
            "eligible_before_duplicate_review",
        ):
            print(f"  {name}: {totals.get(name, 0)}")

    print("\nInput fingerprints unchanged.")
    print("Report saved to:", OUTPUT)


if __name__ == "__main__":
    main()