"""Derive candidate symptom concepts from original DDXPlus evidence.

This module does not modify datasets, generate partial scenarios,
train models, or change deployed inference.
"""

import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "docs/candidate_feature_schema_v1.json"


def load_candidate_schema():
    schema = json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8")
    )

    base = schema["base_symptom_features"]
    concepts = schema["symptom_features"]
    derived = schema["derived_features"]

    if schema["schema_id"] != "respiratory_candidate_v1":
        raise ValueError("Unexpected candidate schema")

    if len(base) != 32 or len(set(base)) != 32:
        raise ValueError("Expected 32 unique base concepts")

    if len(concepts) != 34 or len(set(concepts)) != 34:
        raise ValueError("Expected 34 unique candidate concepts")

    if set(base) & set(derived):
        raise ValueError("Base and derived concepts must be distinct")

    if set(concepts) != set(base) | set(derived):
        raise ValueError("Candidate concepts do not match definitions")

    expected_names = (
        ["AGE", "SEX_M"]
        + [f"{code}__present" for code in concepts]
        + [f"{code}__absent" for code in concepts]
    )

    if schema["feature_names"] != expected_names:
        raise ValueError("Unexpected candidate feature order")

    if schema["feature_count"] != len(expected_names):
        raise ValueError("Incorrect candidate feature count")

    seen_locations = set()
    for concept, definition in derived.items():
        parent = definition["parent_present_feature"]
        locations = definition["positive_source_features"]

        if parent not in base:
            raise ValueError(f"Unsupported parent for {concept}")

        if not locations or len(locations) != len(set(locations)):
            raise ValueError(f"Invalid location list for {concept}")

        if seen_locations & set(locations):
            raise ValueError("Pain-region location groups overlap")

        seen_locations.update(locations)

    return schema


def derive_candidate_state(evidence_items, schema):
    """Derive positive concepts from an unmasked source evidence list.

    Missing supported evidence remains unknown.
    Unmodelled source evidence is not converted into candidate features.
    """

    if not isinstance(evidence_items, (list, tuple, set, frozenset)):
        raise ValueError("Evidence must be a collection of source codes")

    if any(
        not isinstance(code, str) or not code.strip()
        for code in evidence_items
    ):
        raise ValueError("Evidence codes must be non-empty strings")

    source = set(evidence_items)
    concepts = schema["symptom_features"]
    present = source & set(schema["base_symptom_features"])
    derived_from = {}

    for concept, definition in schema["derived_features"].items():
        matching_locations = (
            source & set(definition["positive_source_features"])
        )

        if not matching_locations:
            continue

        parent = definition["parent_present_feature"]
        if parent not in source:
            raise ValueError(
                f"{concept} has location evidence but lacks "
                f"required parent {parent}"
            )

        present.add(concept)
        derived_from[concept] = sorted(matching_locations)

    return {
        "present": [
            code for code in concepts if code in present
        ],
        "absent": [],
        "unknown": [
            code for code in concepts if code not in present
        ],
        "derived_from": derived_from,
    }

def get_reporting_units(state, schema):
    """Return positive concepts that can be sampled as reporting units.

    A region and its general-pain parent form one reporting unit.
    This preparation path supports positive-only source observations.
    """

    groups = {}
    supported = set(schema["symptom_features"])

    for name in ("present", "absent", "unknown"):
        values = state.get(name)

        if not isinstance(values, (list, tuple, set, frozenset)):
            raise ValueError(f"{name} must be a collection of concepts")

        if any(not isinstance(code, str) for code in values):
            raise ValueError(f"{name} contains an invalid concept")

        groups[name] = set(values)

        if len(groups[name]) != len(values):
            raise ValueError(f"{name} contains duplicate concepts")

    present = groups["present"]
    absent = groups["absent"]
    unknown = groups["unknown"]

    if present & absent or present & unknown or absent & unknown:
        raise ValueError("Symptom-state groups overlap")

    if present | absent | unknown != supported:
        raise ValueError("Symptom states must partition the schema")

    if absent:
        raise ValueError(
            "This preparation path does not support absent source evidence"
        )

    units = set(present)

    for concept, definition in schema["derived_features"].items():
        if concept not in present:
            continue

        parent = definition["parent_present_feature"]
        if parent not in present:
            raise ValueError(
                f"{concept} requires present parent {parent}"
            )

        # Avoid sampling the parent as an additional independent unit.
        units.discard(parent)

    return [
        code for code in schema["symptom_features"] if code in units
    ]


def mask_candidate_state(
    full_state,
    schema,
    *,
    reported_unit_count,
    seed,
):
    """Select a reproducible subset of positive reporting units.

    The caller must use a stable seed derived from the source group
    and variant identifier, without using the disease label.

    Selecting all available units is permitted. The dataset builder
    must avoid writing duplicate variants.
    """

    units = get_reporting_units(full_state, schema)

    if type(reported_unit_count) is not int:
        raise ValueError("reported_unit_count must be an integer")

    if not 1 <= reported_unit_count <= len(units):
        raise ValueError(
            f"reported_unit_count must be between 1 and {len(units)}"
        )

    if type(seed) not in (int, str):
        raise ValueError("seed must be an integer or string")

    rng = random.Random(seed)
    selected = set(rng.sample(units, reported_unit_count))
    present = set(selected)

    for concept, definition in schema["derived_features"].items():
        if concept in selected:
            present.add(definition["parent_present_feature"])

    if not present.issubset(full_state["present"]):
        raise ValueError("Masking introduced a new positive concept")

    concepts = schema["symptom_features"]
    source_derivations = full_state.get("derived_from", {})

    return {
        "present": [
            code for code in concepts if code in present
        ],
        "absent": [],
        "unknown": [
            code for code in concepts if code not in present
        ],
        "derived_from": {
            concept: list(locations)
            for concept, locations in source_derivations.items()
            if concept in present
        },
        "reported_units": [
            code for code in units if code in selected
        ],
        "reported_unit_count": reported_unit_count,
    }

def make_training_variants(
    full_state,
    schema,
    *,
    source_group_id,
    seed=42,
):
    """Create full and, where possible, strictly partial observations.

    The source group ID must come from normalized original inputs.
    It must not include the disease label or generated variant.
    """

    if not isinstance(source_group_id, str) or not source_group_id.strip():
        raise ValueError("source_group_id must be a non-empty string")

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    units = get_reporting_units(full_state, schema)
    if not units:
        raise ValueError("Cannot generate variants without positive units")

    seed_prefix = f"{seed}:{source_group_id}"

    full = mask_candidate_state(
        full_state,
        schema,
        reported_unit_count=len(units),
        seed=f"{seed_prefix}:full",
    )
    full["variant_id"] = "full"
    variants = [full]

    if len(units) >= 2:
        count_rng = random.Random(f"{seed_prefix}:partial:count")
        requested_count = count_rng.randint(
            1, min(5, len(units) - 1)
        )

        partial = mask_candidate_state(
            full_state,
            schema,
            reported_unit_count=requested_count,
            seed=f"{seed_prefix}:partial:selection",
        )
        partial["variant_id"] = "partial"

        if not set(partial["present"]) < set(full["present"]):
            raise ValueError("Partial variant must hide positive information")

        variants.append(partial)

    weight = 1.0 / len(variants)

    for variant in variants:
        variant["source_group_id"] = source_group_id
        variant["source_group_weight"] = weight

    return variants