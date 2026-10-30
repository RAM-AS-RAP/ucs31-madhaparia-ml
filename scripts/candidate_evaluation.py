"""Shared validation views and probability alignment for candidate evaluation."""

import numpy as np

from scripts.candidate_feature_builder import (
    get_reporting_units,
    mask_candidate_state,
)


VIEW_LIMITS = {
    "full": None,
    "up_to_1": 1,
    "up_to_2": 2,
    "up_to_3": 3,
    "up_to_5": 5,
}


def make_validation_view(full_frame, schema, setting, seed=42):
    """Create one deterministic observation per validation source group."""

    if setting not in VIEW_LIMITS:
        raise ValueError("Unknown validation setting")

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    required = {
        "source_group_id",
        "variant_id",
        "reported_unit_count",
        *schema["feature_names"],
    }
    if not required.issubset(full_frame.columns):
        raise ValueError("Missing validation columns")

    if full_frame.empty or not full_frame.columns.is_unique:
        raise ValueError("Empty validation data or duplicate columns")

    ids = full_frame["source_group_id"]

    if (
        ids.isna().any()
        or not ids.is_unique
        or any(not isinstance(value, str) or not value.strip() for value in ids)
    ):
        raise ValueError("Validation source-group IDs must be unique strings")

    if not full_frame["variant_id"].eq("full").all():
        raise ValueError("Views must be generated from full observations")

    features = schema["feature_names"]
    numeric = full_frame[features].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite validation features")

    if not ((numeric[:, 0] >= 0) & (numeric[:, 0] <= 120)).all():
        raise ValueError("Invalid validation age")

    if not np.isin(numeric[:, 1:], [0, 1]).all():
        raise ValueError("Non-binary symptom or sex features")

    concepts = schema["symptom_features"]
    present_columns = [f"{code}__present" for code in concepts]
    absent_columns = [f"{code}__absent" for code in concepts]

    if not (full_frame[absent_columns].to_numpy() == 0).all():
        raise ValueError("This evaluation path expects positive-only sources")

    original_present = full_frame[present_columns].to_numpy()
    recorded_counts = full_frame["reported_unit_count"].to_numpy()

    masked_rows = []
    actual_counts = []

    for group_id, flags, recorded_count in zip(
        ids, original_present, recorded_counts
    ):
        present = {
            code for code, flag in zip(concepts, flags) if flag == 1
        }
        state = {
            "present": [code for code in concepts if code in present],
            "absent": [],
            "unknown": [code for code in concepts if code not in present],
            "derived_from": {},
        }

        units = get_reporting_units(state, schema)

        if not units or recorded_count != len(units):
            raise ValueError("Invalid full-observation reporting-unit count")

        limit = VIEW_LIMITS[setting]
        requested = len(units) if limit is None else min(limit, len(units))

        masked = mask_candidate_state(
            state,
            schema,
            reported_unit_count=requested,
            seed=f"{seed}:{group_id}:evaluation:{setting}",
        )

        selected = set(masked["present"])
        masked_rows.append([
            int(code in selected) for code in concepts
        ])
        actual_counts.append(requested)

    result = full_frame.copy()
    result[present_columns] = np.asarray(masked_rows, dtype=np.uint8)
    result["reported_unit_count"] = actual_counts
    result["variant_id"] = setting

    return result


def project_v2_features(view_frame, v2_schema):
    """Select V2's supported features without changing the observation."""

    names = v2_schema["feature_names"]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate V2 feature names")

    missing = set(names) - set(view_frame.columns)
    if missing:
        raise ValueError(f"Missing V2 features: {sorted(missing)}")

    return view_frame.loc[:, names].copy()


def harmonize_probabilities(probabilities, source_classes, target_classes):
    """Combine allergy aliases without dropping or renormalizing scores."""

    source_classes = list(source_classes)
    target_classes = list(target_classes)
    values = np.asarray(probabilities, dtype=float)

    if len(source_classes) != len(set(source_classes)):
        raise ValueError("Duplicate source classes")

    if len(target_classes) != len(set(target_classes)):
        raise ValueError("Duplicate target classes")

    if values.ndim != 2 or values.shape[1] != len(source_classes):
        raise ValueError("Probability shape does not match source classes")

    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-8)
    ):
        raise ValueError("Invalid probability distribution")

    positions = {
        label: index for index, label in enumerate(target_classes)
    }
    aligned = np.zeros((len(values), len(target_classes)), dtype=float)

    for source_index, label in enumerate(source_classes):
        canonical = (
            "Allergic rhinitis" if label == "Allergic sinusitis" else label
        )

        if canonical not in positions:
            raise ValueError(f"No target class for {label}")

        aligned[:, positions[canonical]] += values[:, source_index]

    if not np.allclose(aligned.sum(axis=1), 1.0, rtol=0, atol=1e-8):
        raise ValueError("Probability mass was lost during alignment")

    return aligned