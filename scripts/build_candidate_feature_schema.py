"""Build a proposed respiratory feature schema for issue #45.

This is a design artifact, not a trained model schema.
No existing model, mapping or dataset is modified.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

V2_SCHEMA_PATH = ROOT / "models/v2_feature_schema.json"
EVIDENCE_PATH = ROOT / "data/raw/release_evidences.json"
OUTPUT_PATH = ROOT / "docs/candidate_feature_schema_v1.json"


# Explicit, reviewable source-value groupings.
REGION_VALUES = {
    "head_region_pain": {
        "V_25": "back of head",
        "V_62": "top of the head",
        "V_89": "forehead",
        "V_124": "occiput",
        "V_166": "temple(R)",
        "V_167": "temple(L)",
    },
    "chest_region_pain": {
        "V_29": "lower chest",
        "V_55": "side of the chest(R)",
        "V_56": "side of the chest(L)",
        "V_101": "upper chest",
        "V_170": "posterior chest wall(R)",
        "V_171": "posterior chest wall(L)",
    },
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    v2 = load_json(V2_SCHEMA_PATH)
    evidences = load_json(EVIDENCE_PATH)

    original_hashes = {
        "models/v2_feature_schema.json": fingerprint(V2_SCHEMA_PATH),
        "data/raw/release_evidences.json": fingerprint(EVIDENCE_PATH),
    }

    base_features = list(v2["symptom_features"])

    if len(base_features) != 32 or len(set(base_features)) != 32:
        raise ValueError("Expected 32 unique V2 symptom concepts.")

    for code in base_features:
        evidence = evidences.get(code)

        if evidence is None:
            raise ValueError(f"Missing source evidence: {code}")

        if evidence.get("data_type") != "B":
            raise ValueError(f"Expected binary source evidence: {code}")

        if evidence.get("is_antecedent"):
            raise ValueError(f"Unexpected medical-history feature: {code}")

    location = evidences["E_55"]

    if location["data_type"] != "M":
        raise ValueError("Expected multi-choice pain-location evidence.")

    if location["code_question"] != "E_53":
        raise ValueError("Unexpected pain-location parent evidence.")

    derived_features = {}

    for name, values in REGION_VALUES.items():
        for value, expected_meaning in values.items():
            if value not in location["possible-values"]:
                raise ValueError(f"Unsupported location value: {value}")

            actual_meaning = location["value_meaning"][value]["en"]

            if actual_meaning != expected_meaning:
                raise ValueError(
                    f"Source meaning changed for {value}: "
                    f"{actual_meaning!r}"
                )

        derived_features[name] = {
            "kind": "proposed_region_aggregation",
            "source_evidence": "E_55",
            "parent_present_feature": "E_53",
            "positive_source_features": [
                f"E_55_@_{value}"
                for value in values
            ],
            "source_value_meanings": values,
            "training_derivation": (
                "Present when at least one listed location is present "
                "in the original source record. Derive before masking "
                "reported symptom information."
            ),
            "missing_location_policy": (
                "No listed positive location does not by itself "
                "establish explicit absence of regional pain."
            ),
            "inference_requirement": (
                "Require confirmation of this region. Do not invent "
                "an exact source location from a broad region report."
            ),
        }

    head_values = set(REGION_VALUES["head_region_pain"])
    chest_values = set(REGION_VALUES["chest_region_pain"])

    if head_values & chest_values:
        raise ValueError("Head and chest location groups overlap.")

    concepts = base_features + list(REGION_VALUES)

    feature_names = (
        ["AGE", "SEX_M"]
        + [f"{code}__present" for code in concepts]
        + [f"{code}__absent" for code in concepts]
    )

    if len(feature_names) != 70 or len(set(feature_names)) != 70:
        raise ValueError("Candidate must have 70 unique numerical features.")

    schema = {
        "schema_id": "respiratory_candidate_v1",
        "status": "proposed_not_trained",
        "related_issue": 45,
        "compatible_with_existing_v2_model": False,
        "source_sha256": original_hashes,
        "base_symptom_features": base_features,
        "symptom_features": concepts,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "symptom_concept_count": len(concepts),
        "demographics": {
            "AGE": {
                "meaning": "Reported age in years",
                "minimum": 0,
                "maximum": 120,
                "missing_policy": "Do not invent a default age.",
            },
            "SEX_M": {
                "encoding": {"M": 1, "F": 0},
                "missing_policy": "Do not invent a default sex.",
                "scope": "Retains the current dataset encoding.",
            },
        },
        "symptom_state_encoding": {
            "present": {"present": 1, "absent": 0},
            "absent": {"present": 0, "absent": 1},
            "unknown": {"present": 0, "absent": 0},
            "invalid": {"present": 1, "absent": 1},
        },
        "negative_evidence_training_supported": False,
        "negative_evidence_note": (
            "Explicit absence is representable, but usable negative "
            "training evidence has not been established. Do not "
            "interpret unreported symptoms as confirmed absent."
        ),
        "derived_features": derived_features,
        "canonical_mapping_overrides": {
            "headache": {
                "requires_confirmation_of": "head_region_pain",
                "present_features": ["E_53", "head_region_pain"],
                "exact_location_required": False,
            },
            "chest_pain": {
                "requires_confirmation_of": "chest_region_pain",
                "present_features": ["E_53", "chest_region_pain"],
                "exact_location_required": False,
            },
        },
        "other_mapping_policy": (
            "Confirm the source-defined concept and its qualifiers "
            "before encoding. Existing aliases are not automatically "
            "approved by their feature's inclusion in this schema."
        ),
        "consistency_requirements": [
            "Present head-region pain requires present E_53.",
            "Present chest-region pain requires present E_53.",
            "Both pain regions may be present together.",
            "A symptom cannot be both present and absent.",
            "Contradictory reports require clarification.",
            "Denying one pain region does not deny all pain.",
            "Training masking must preserve parent-region consistency.",
        ],
        "safety_only_codes": [
            "blue_lips",
            "confusion",
            "fainting",
        ],
        "nlp_only_codes": ["sneezing"],
        "duration_policy": (
            "Retain reported duration as contextual information. "
            "Duration is not included in this numerical schema."
        ),
        "limitations": [
            "This specification is not a trained or validated model.",
            "Region groupings require source-data coverage evaluation.",
            "The schema does not fix dataset associations by itself.",
            "The 32 original concepts remain subject to semantic review.",
            "Disease scope is decided separately in issue #46.",
            "Training implementation belongs to issues #47 and #48.",
            "Safety handling remains independent of model features.",
        ],
    }

    for relative_path, expected_hash in original_hashes.items():
        if fingerprint(ROOT / relative_path) != expected_hash:
            raise RuntimeError("An input file changed during generation.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPROPOSED CANDIDATE SCHEMA")
    print("Original binary concepts:", len(base_features))
    print("Proposed regional concepts:", len(REGION_VALUES))
    print("Total symptom concepts:", len(concepts))
    print("Numerical features:", len(feature_names))

    for name, values in REGION_VALUES.items():
        print(f"\n{name}:")
        for value, meaning in values.items():
            print(f"  E_55_@_{value}: {meaning}")

    print("\nStatus: proposed, not trained")
    print("Existing V2 schema unchanged.")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()