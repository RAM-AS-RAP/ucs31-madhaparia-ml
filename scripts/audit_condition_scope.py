"""Audit condition coverage and a proposed candidate label mapping.

Reads training and validation sources. Does not inspect the test split,
change source labels, or train a model.
"""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/condition_scope_audit"

INPUTS = {
    "conditions": "data/raw/release_conditions.json",
    "original_labels": "data/processed/diseases.json",
    "manifest": "models/rf_v2_23class_prototype_manifest.json",
    "profiles": "docs/v2_candidate_condition_profiles.json",
    "train": "data/raw/release_train_patients",
    "validation": "data/raw/release_validate_patients",
}

PROPOSED_ALIASES = {
    "Allergic sinusitis": "Allergic rhinitis",
}


def load_json(name):
    return json.loads(
        (ROOT / INPUTS[name]).read_text(encoding="utf-8-sig")
    )


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints():
    return {
        relative: file_hash(ROOT / relative)
        for relative in [*INPUTS.values(), "scripts/audit_condition_scope.py"]
    }


def read_counts(name):
    counts = Counter()
    with (ROOT / INPUTS[name]).open(
        encoding="utf-8-sig", newline=""
    ) as source:
        reader = csv.DictReader(source)
        if "PATHOLOGY" not in (reader.fieldnames or []):
            raise ValueError(f"Missing PATHOLOGY column: {name}")

        for row_number, row in enumerate(reader, start=1):
            label = row["PATHOLOGY"]
            if not label or not label.strip():
                raise ValueError(
                    f"Empty pathology in {name}, row {row_number}"
                )
            counts[label] += 1

    return counts


def main():
    csv.field_size_limit(2_147_483_647)
    before = fingerprints()

    metadata = load_json("conditions")
    manifest = load_json("manifest")
    original = {
        item["name"] for item in load_json("original_labels")
    }
    current = set(manifest["classes"])
    extensions = set(manifest["synthetic_conditions"])
    profiles = {
        item["name"]: item
        for item in load_json("profiles")["conditions"]
    }

    assert len(original) == 21
    assert len(current) == 23
    assert original == current - extensions
    assert extensions == {"COVID-19", "Allergic rhinitis"}
    assert set(manifest["training_counts_by_class"]) == current
    assert (
        sum(manifest["training_counts_by_class"].values())
        == manifest["training_count"]
    )

    counts = {}
    for split in ("train", "validation"):
        print(f"Reading {split} labels...", flush=True)
        counts[split] = read_counts(split)

    raw_labels = set(counts["train"]) | set(counts["validation"])
    assert raw_labels.issubset(metadata)
    assert original.issubset(counts["train"])
    assert original.issubset(counts["validation"])

    rows = []
    for label in sorted(current):
        definition = metadata.get(label, {})
        rows.append({
            "source_label": label,
            "proposed_canonical_label": PROPOSED_ALIASES.get(
                label, label
            ),
            "source_french_name": definition.get("cond-name-fr"),
            "source_icd10": definition.get("icd10-id"),
            "training_source_group": (
                "additional_rule_generated"
                if label in extensions
                else "ddxplus_simulated"
            ),
            "raw_train_count": counts["train"][label],
            "raw_validation_count": counts["validation"][label],
            "v2_fitted_count_from_manifest": (
                manifest["training_counts_by_class"][label]
            ),
        })

    allergy = metadata["Allergic sinusitis"]
    original_codes = set(allergy["symptoms"])
    added_codes = set(
        profiles["Allergic rhinitis"]["possible_features"]
    )

    proposed_original = {
        PROPOSED_ALIASES.get(label, label) for label in original
    }
    proposed_with_covid = proposed_original | {"COVID-19"}

    assert len(proposed_original) == 21
    assert len(proposed_with_covid) == 22

    report = {
        "issue": 46,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "audit_only_candidate_mapping_proposed",
        "source_sha256": before,
        "current_v2_class_count": len(current),
        "conditions": rows,
        "raw_split_totals": {
            split: sum(values.values())
            for split, values in counts.items()
        },
        "original_21_split_totals": {
            split: sum(values[label] for label in original)
            for split, values in counts.items()
        },
        "raw_labels_outside_current_v2": [
            {
                "source_label": label,
                "raw_train_count": counts["train"][label],
                "raw_validation_count": counts["validation"][label],
            }
            for label in sorted(raw_labels - current)
        ],
        "allergy_label_review": {
            "source_label": "Allergic sinusitis",
            "source_french_name": allergy["cond-name-fr"],
            "source_icd10": allergy["icd10-id"],
            "source_symptom_codes": sorted(original_codes),
            "added_rhinitis_symptom_codes": sorted(added_codes),
            "symptom_code_sets_identical": original_codes == added_codes,
            "interpretation": (
                "Suspected label inconsistency based on French name, "
                "ICD category and matching symptom definitions. "
                "Matching symptom codes alone do not prove equivalence."
            ),
        },
        "proposed_source_label_mapping": {
            label: PROPOSED_ALIASES.get(label, label)
            for label in sorted(original)
        },
        "proposed_ddxplus_canonical_labels": sorted(proposed_original),
        "conditional_scope_with_covid": sorted(proposed_with_covid),
        "limitations": [
            "Mapping is proposed and has not been applied to any dataset.",
            "COVID-19 inclusion requires a separate data decision.",
            "Original DDXPlus records are simulated patients.",
            "Manifest counts are recorded counts, not retraining results.",
            "Raw counts precede symptom filtering and scenario generation.",
            "No model performance or clinical validity is established.",
            "Final test split was not inspected.",
        ],
    }

    after = fingerprints()
    if before != after:
        raise RuntimeError("An input changed during the audit; rerun it.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "scope_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (OUTPUT / "condition_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nCONDITION SCOPE AUDIT")
    print("Current V2 classes:", len(current))
    print("Original DDXPlus classes:", len(original))
    print("Proposed canonical DDXPlus classes:", len(proposed_original))
    print("Conditional scope including COVID-19:", len(proposed_with_covid))
    print("Allergy source French name:", allergy["cond-name-fr"])
    print("Allergy source ICD category:", allergy["icd10-id"])
    print("Allergy symptom sets identical:", original_codes == added_codes)
    print("Allergy raw training records:", counts["train"]["Allergic sinusitis"])
    print(
        "Allergy raw validation records:",
        counts["validation"]["Allergic sinusitis"],
    )
    print("Input fingerprints unchanged.")
    print("No labels changed and no model trained.")
    print("Reports saved to:", OUTPUT)


if __name__ == "__main__":
    main()