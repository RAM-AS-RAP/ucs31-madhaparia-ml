"""Audit canonical symptom mappings against DDXPlus and V2.

Creates inspection reports only.
Structural validity does not establish semantic or clinical correctness.
"""

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports/symptom_feature_audit"

INPUT_FILES = {
    "canonical": "nlp/canonical_symptoms.json",
    "english": "nlp/english_aliases.json",
    "swahili": "nlp/swahili_aliases.json",
    "schema": "models/v2_feature_schema.json",
    "evidences": "data/raw/release_evidences.json",
}


def load_json(relative_path):
    return json.loads(
        (ROOT / relative_path).read_text(encoding="utf-8-sig")
    )


def describe_feature(feature, evidences):
    base, separator, value = feature.partition("_@_")
    evidence = evidences.get(base)

    result = {
        "feature": feature,
        "base_evidence": base,
        "encoded_value": value if separator else None,
        "issues": [],
    }

    if evidence is None:
        result["issues"].append("missing_source_evidence")
        return result

    result.update({
        "source_question": evidence.get("question_en"),
        "source_type": evidence.get("data_type"),
        "source_parent_question": evidence.get("code_question"),
        "source_is_antecedent": evidence.get("is_antecedent"),
        "source_default_value": evidence.get("default_value"),
        "source_value_meaning": None,
    })

    if evidence.get("is_antecedent"):
        result["issues"].append("mapped_to_medical_history_evidence")

    if separator:
        if evidence.get("data_type") not in {"C", "M"}:
            result["issues"].append("value_suffix_on_non_categorical_evidence")

        if value not in evidence.get("possible-values", []):
            result["issues"].append("unknown_source_value")

        meaning = evidence.get("value_meaning", {}).get(value, {})
        result["source_value_meaning"] = meaning.get("en")

        if not result["source_value_meaning"]:
            result["issues"].append("missing_english_value_meaning")

        if value == evidence.get("default_value"):
            result["issues"].append("mapped_to_source_default_value")
    elif evidence.get("data_type") != "B":
        result["issues"].append("categorical_evidence_without_value")

    return result


def main():
    inputs = {
        name: load_json(path)
        for name, path in INPUT_FILES.items()
    }

    canonical = inputs["canonical"]
    supported = set(inputs["schema"]["symptom_features"])
    codes = [item["code"] for item in canonical]

    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate canonical symptom codes.")

    alias_index = defaultdict(list)
    unknown_alias_codes = []

    for language, dictionary in [
        ("en", inputs["english"]),
        ("sw", inputs["swahili"]),
    ]:
        for code, aliases in dictionary.items():
            if code not in codes:
                unknown_alias_codes.append({
                    "language": language,
                    "code": code,
                })

            for alias in aliases:
                normalized = " ".join(alias.lower().split())
                alias_index[normalized].append({
                    "code": code,
                    "language": language,
                })

    alias_conflicts = [
        {
            "normalized_alias": alias,
            "assignments": assignments,
        }
        for alias, assignments in sorted(alias_index.items())
        if len({item["code"] for item in assignments}) > 1
    ]

    rows = []
    feature_groups = defaultdict(list)

    for symptom in canonical:
        code = symptom["code"]
        features = symptom["features"]
        retained = sorted(set(features) & supported)
        excluded = sorted(set(features) - supported)

        source_details = [
            describe_feature(feature, inputs["evidences"])
            for feature in features
        ]

        if not features:
            coverage = (
                "safety_only"
                if symptom["is_red_flag"]
                else "nlp_only"
            )
        elif not retained:
            coverage = "no_v2_coverage"
        elif excluded:
            coverage = "partial_v2_coverage"
        else:
            coverage = "full_v2_coverage"

        if retained:
            feature_groups[tuple(retained)].append(code)

        rows.append({
            "code": code,
            "display_name": symptom["display_name"],
            "is_red_flag": symptom["is_red_flag"],
            "english_aliases": inputs["english"].get(code, []),
            "swahili_aliases": inputs["swahili"].get(code, []),
            "canonical_features": features,
            "retained_v2_features": retained,
            "excluded_features": excluded,
            "coverage": coverage,
            "source_details": source_details,
            "semantic_review": "pending",
        })

    collisions = [
        {
            "retained_v2_features": list(features),
            "canonical_codes": group,
        }
        for features, group in sorted(feature_groups.items())
        if len(group) > 1
    ]

    all_mapped = {
        feature
        for row in rows
        for feature in row["canonical_features"]
    }

    coverage_counts = dict(sorted(
        (status, sum(row["coverage"] == status for row in rows))
        for status in {row["coverage"] for row in rows}
    ))

    source_issue_count = sum(
        len(detail["issues"])
        for row in rows
        for detail in row["source_details"]
    )

    report = {
        "audit_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in INPUT_FILES.values()
        },
        "canonical_symptom_count": len(rows),
        "v2_symptom_feature_count": len(supported),
        "coverage_counts": coverage_counts,
        "source_issue_count": source_issue_count,
        "identical_v2_input_groups": collisions,
        "alias_conflicts": alias_conflicts,
        "unknown_alias_codes": unknown_alias_codes,
        "v2_features_without_canonical_mapping": sorted(
            supported - all_mapped
        ),
        "limitations": [
            "Full coverage means inclusion in the schema, not correctness.",
            "All symptom meanings and aliases still require manual review.",
            "Safety-only and NLP-only entries are not automatically errors.",
            "Identical vectors describe schema-level information loss.",
            "No training or model inference is performed.",
        ],
        "symptoms": rows,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "mapping_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    columns = [
        "code",
        "display_name",
        "is_red_flag",
        "coverage",
        "english_aliases",
        "swahili_aliases",
        "canonical_features",
        "retained_v2_features",
        "excluded_features",
        "source_details",
        "semantic_review",
    ]

    with (OUTPUT_DIR / "mapping_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(row[key], ensure_ascii=False)
                    if isinstance(row[key], list)
                    else row[key]
                )
                for key in columns
            })

    print("\nSYMPTOM-TO-FEATURE AUDIT")
    print("Canonical symptoms:", len(rows))
    print("V2 symptom features:", len(supported))
    print("Coverage:", coverage_counts)
    print("Source-definition issues:", source_issue_count)
    print("Conflicting aliases:", len(alias_conflicts))
    print("Unknown alias codes:", len(unknown_alias_codes))

    print("\nSymptoms with incomplete V2 coverage:")
    for row in rows:
        if row["coverage"] != "full_v2_coverage":
            print(
                f"  {row['code']}: {row['coverage']}"
                f" | retained={row['retained_v2_features']}"
                f" | excluded={row['excluded_features']}"
            )

    print("\nDifferent symptoms with identical V2 inputs:")
    for group in collisions:
        print(
            " ",
            group["canonical_codes"],
            "->",
            group["retained_v2_features"],
        )

    if not collisions:
        print("  None.")

    print("\nReports saved to:", OUTPUT_DIR)
    print("Semantic review remains pending.")


if __name__ == "__main__":
    main()