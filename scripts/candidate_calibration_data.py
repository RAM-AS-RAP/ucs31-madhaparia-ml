"""Load full observations for one reserved calibration partition."""

import numpy as np
import pandas as pd

from scripts.candidate_evaluation import make_validation_view


CALIBRATION_PARTITIONS = (
    "calibration_fit",
    "policy_development",
    "uncertainty_evaluation",
)


def validate_calibration_partitions(partitions):
    required = [
        "source_group_id",
        "source_split",
        "canonical_label",
        "parent_partition",
        "partition",
    ]

    if not partitions.columns.is_unique:
        raise ValueError("Duplicate partition columns")

    if not set(required).issubset(partitions.columns):
        raise ValueError("Missing calibration partition columns")

    if partitions.empty:
        raise ValueError("Empty calibration partition table")

    for name in required:
        valid = partitions[name].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not valid.all():
            raise ValueError(f"Invalid partition values: {name}")

    if not partitions["source_group_id"].str.fullmatch(
        r"[0-9a-f]{64}"
    ).all():
        raise ValueError("Invalid source-group signature")

    if not partitions["source_group_id"].is_unique:
        raise ValueError("Duplicate source-group IDs in partition table")

    if not partitions["source_split"].eq("train").all():
        raise ValueError("Only reserved training groups are permitted")

    if not partitions["parent_partition"].eq(
        "calibration_reserved"
    ).all():
        raise ValueError("Unexpected parent partition")

    if not partitions["partition"].isin(
        CALIBRATION_PARTITIONS
    ).all():
        raise ValueError("Unexpected calibration partition")


def load_calibration_full_rows(
    training_path,
    partitions,
    schema,
    *,
    partition,
    chunksize=50000,
):
    """Load exactly one full observation for each requested source group.

    The caller must verify source and partition fingerprints before use.
    This helper checks membership, labels, completeness and feature values.
    It does not read models or calculate predictions.
    """

    if partition not in CALIBRATION_PARTITIONS:
        raise ValueError("Unknown requested calibration partition")

    if type(chunksize) is not int or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")

    validate_calibration_partitions(partitions)

    selected_groups = partitions.loc[
        partitions["partition"].eq(partition)
    ].copy()

    if selected_groups.empty:
        raise ValueError("No groups in requested partition")

    expected_labels = selected_groups.set_index(
        "source_group_id"
    )["canonical_label"]
    selected_ids = set(expected_labels.index)

    metadata = [
        "source_group_id",
        "variant_id",
        "canonical_label",
        "reported_unit_count",
    ]
    features = list(schema["feature_names"])
    columns = metadata + features

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate features or overlap with metadata")

    # Keep original numeric precision. Training sample weights are not
    # loaded: the calibration-view weights will be assigned separately.
    dtypes = {name: np.float64 for name in features}
    dtypes.update({
        "source_group_id": str,
        "variant_id": str,
        "canonical_label": str,
        "reported_unit_count": np.float64,
    })

    selected_chunks = []

    for chunk in pd.read_csv(
        training_path,
        usecols=columns,
        dtype=dtypes,
        chunksize=chunksize,
    ):
        members = chunk.loc[
            chunk["source_group_id"].isin(selected_ids), columns
        ]

        if not members["variant_id"].isin(["full", "partial"]).all():
            raise ValueError("Unexpected variant in requested groups")

        full = members.loc[members["variant_id"].eq("full")]
        if not full.empty:
            selected_chunks.append(full.copy())

    if not selected_chunks:
        raise ValueError("No full observations found")

    result = pd.concat(selected_chunks, ignore_index=True)

    if not result["source_group_id"].is_unique:
        raise ValueError("Duplicate full observations")

    if set(result["source_group_id"]) != selected_ids:
        raise ValueError("Some requested groups have no full observation")

    labels = result["source_group_id"].map(expected_labels)
    if not labels.eq(result["canonical_label"]).all():
        raise ValueError("Observation labels disagree with partition labels")

    result = result.sort_values(
        "source_group_id", kind="stable"
    ).reset_index(drop=True)

    # Reuse the established checks for age, binary features,
    # positive-only evidence, reporting units and regional-pain parents.
    result = make_validation_view(
        result, schema, setting="full", seed=42
    )

    result["calibration_partition"] = partition
    return result