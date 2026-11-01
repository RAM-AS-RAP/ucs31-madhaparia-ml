"""Portable feature validation and positive reporting-unit rules.

The caller must supply a schema already validated by the bundle loader.
This module does not read files or load models.
"""

import math
from numbers import Real

import numpy as np

from .errors import CandidateInputError


def normalize_context_codes(values, name):
    """Normalize safety or unsupported-symptom codes."""

    if not isinstance(values, (list, tuple, set, frozenset)):
        raise CandidateInputError(
            f"{name} must be a collection of strings"
        )

    if any(
        not isinstance(value, str) or not value.strip()
        for value in values
    ):
        raise CandidateInputError(
            f"{name} contains an invalid value"
        )

    return sorted({value.strip() for value in values})


def get_reporting_units(state, schema):
    """Return positive reporting units in schema order.

    Regional pain includes its required general-pain parent.
    The parent does not count as an additional reporting unit.
    """

    groups = {}
    supported = set(schema["symptom_features"])

    for name in ("present", "absent", "unknown"):
        values = state.get(name)

        if not isinstance(values, (list, tuple, set, frozenset)):
            raise CandidateInputError(
                f"{name} must be a collection of concepts"
            )

        if any(not isinstance(code, str) for code in values):
            raise CandidateInputError(
                f"{name} contains an invalid concept"
            )

        groups[name] = set(values)

        if len(groups[name]) != len(values):
            raise CandidateInputError(
                f"{name} contains duplicate concepts"
            )

    present = groups["present"]
    absent = groups["absent"]
    unknown = groups["unknown"]

    if present & absent or present & unknown or absent & unknown:
        raise CandidateInputError("Symptom-state groups overlap")

    if present | absent | unknown != supported:
        raise CandidateInputError(
            "Symptom states must partition the schema"
        )

    if absent:
        raise CandidateInputError(
            "This preparation path does not support absent source evidence"
        )

    units = set(present)

    for concept, definition in schema["derived_features"].items():
        if concept not in present:
            continue

        parent = definition["parent_present_feature"]

        if parent not in present:
            raise CandidateInputError(
                f"{concept} requires present parent {parent}"
            )

        units.discard(parent)

    return [
        code
        for code in schema["symptom_features"]
        if code in units
    ]


def validate_candidate_features(feature_values, schema):
    """Return ordered numerical values, unit count and absent concepts.

    Explicit negative evidence is returned separately. The prediction
    layer must withhold rather than score such observations.
    """

    if not isinstance(feature_values, dict):
        raise CandidateInputError(
            "Features must be supplied as a dictionary"
        )

    names = schema["feature_names"]

    if set(feature_values) != set(names):
        raise CandidateInputError(
            "Feature mismatch: supply exactly the candidate features"
        )

    row = {}

    for name in names:
        value = feature_values[name]

        if name == "AGE":
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
            ):
                raise CandidateInputError(
                    "AGE must be a numeric value between 0 and 120"
                )

        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise CandidateInputError(
                f"Invalid feature value: {name}"
            ) from None

        if not math.isfinite(value):
            raise CandidateInputError(
                f"Non-finite feature value: {name}"
            )

        if name == "AGE":
            if not 0 <= value <= 120:
                raise CandidateInputError(
                    "AGE must be between 0 and 120"
                )
        elif value not in (0.0, 1.0):
            raise CandidateInputError(
                f"{name} must be encoded as 0 or 1"
            )

        row[name] = value

    concepts = schema["symptom_features"]

    present = {
        code
        for code in concepts
        if row[f"{code}__present"] == 1
    }
    absent = {
        code
        for code in concepts
        if row[f"{code}__absent"] == 1
    }

    if present & absent:
        raise CandidateInputError(
            "Conflicting present and absent symptom states"
        )

    # Match the reference: count positive evidence only, while
    # retaining explicit negatives separately for the blocking gate.
    state = {
        "present": [
            code for code in concepts if code in present
        ],
        "absent": [],
        "unknown": [
            code for code in concepts if code not in present
        ],
    }

    units = get_reporting_units(state, schema)

    return row, len(units), sorted(absent)