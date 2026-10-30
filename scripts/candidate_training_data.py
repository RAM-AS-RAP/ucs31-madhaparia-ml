"""Load fitting observations and calculate candidate sample weights."""

import numpy as np
import pandas as pd


CANDIDATE_NAMES = (
    "rf_group_weighted",
    "rf_class_balanced",
)


def validate_fit_groups(fit_groups):
    required = {"source_group_id", "canonical_label", "partition"}

    if not required.issubset(fit_groups.columns):
        raise ValueError("Missing fitting-group columns")

    if fit_groups.empty:
        raise ValueError("No fitting groups")

    if fit_groups[list(required)].isna().any().any():
        raise ValueError("Missing fitting-group values")

    if not fit_groups["partition"].eq("fit").all():
        raise ValueError("Only fitting groups are permitted")

    if fit_groups["source_group_id"].duplicated().any():
        raise ValueError("Fitting source groups must be unique")


def load_fitting_rows(
    training_path,
    partitions,
    schema,
    *,
    chunksize=50000,
):
    """Load only observations assigned to fitting, preserving row order."""

    if partitions["source_group_id"].duplicated().any():
        raise ValueError("Partition table contains duplicate group IDs")

    fit_groups = partitions.loc[partitions["partition"].eq("fit")].copy()
    validate_fit_groups(fit_groups)
    fit_ids = set(fit_groups["source_group_id"])

    metadata_columns = [
        "source_group_id",
        "variant_id",
        "canonical_label",
        "source_group_weight",
    ]
    columns = metadata_columns + schema["feature_names"]

    if len(columns) != len(set(columns)):
        raise ValueError("Feature names overlap metadata or contain duplicates")

    dtypes = {
        name: np.float32 for name in schema["feature_names"]
    }
    dtypes["source_group_weight"] = np.float64

    selected_chunks = []

    for chunk in pd.read_csv(
        training_path,
        usecols=columns,
        dtype=dtypes,
        chunksize=chunksize,
    ):
        selected = chunk.loc[
            chunk["source_group_id"].isin(fit_ids), columns
        ]
        if not selected.empty:
            selected_chunks.append(selected.copy())

    if not selected_chunks:
        raise ValueError("No fitting observations found")

    return pd.concat(selected_chunks, ignore_index=True)


def sample_weights_for_candidate(rows, fit_groups, candidate_name):
    """Validate fitting membership and return weights plus class factors."""

    if candidate_name not in CANDIDATE_NAMES:
        raise ValueError("Unknown candidate name")

    validate_fit_groups(fit_groups)

    required = {
        "source_group_id",
        "variant_id",
        "canonical_label",
        "source_group_weight",
    }
    if not required.issubset(rows.columns):
        raise ValueError("Missing training metadata columns")

    if rows.empty or rows[list(required)].isna().any().any():
        raise ValueError("Empty or incomplete training metadata")

    expected_labels = fit_groups.set_index(
        "source_group_id"
    )["canonical_label"]

    if not rows["source_group_id"].isin(expected_labels.index).all():
        raise ValueError("Rows contain non-fitting source groups")

    if set(rows["source_group_id"]) != set(expected_labels.index):
        raise ValueError("Some fitting groups have no observations")

    mapped_labels = rows["source_group_id"].map(expected_labels)
    if not mapped_labels.eq(rows["canonical_label"]).all():
        raise ValueError("Observation labels disagree with group labels")

    if not rows["variant_id"].isin(["full", "partial"]).all():
        raise ValueError("Unexpected training variant")

    if rows.duplicated(["source_group_id", "variant_id"]).any():
        raise ValueError("Duplicate variant within a fitting group")

    full_counts = (
        rows.loc[rows["variant_id"].eq("full"), "source_group_id"]
        .value_counts()
        .reindex(expected_labels.index, fill_value=0)
    )
    if not full_counts.eq(1).all():
        raise ValueError("Each fitting group requires exactly one full variant")

    variant_counts = rows["source_group_id"].value_counts()
    expected_base = (
        1.0 / rows["source_group_id"].map(variant_counts).to_numpy(dtype=float)
    )
    base_weights = rows["source_group_weight"].to_numpy(dtype=float)

    if (
        not np.isfinite(base_weights).all()
        or not np.array_equal(base_weights, expected_base)
    ):
        raise ValueError("Invalid source-group weights")

    class_counts = fit_groups["canonical_label"].value_counts()
    total_groups = len(fit_groups)
    class_count = len(class_counts)

    if candidate_name == "rf_group_weighted":
        factors = {label: 1.0 for label in sorted(class_counts.index)}
    else:
        factors = {
            label: total_groups / (class_count * int(class_counts[label]))
            for label in sorted(class_counts.index)
        }

    weights = (
        base_weights
        * rows["canonical_label"].map(factors).to_numpy(dtype=float)
    )

    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Invalid final sample weights")

    if not np.isclose(weights.sum(), total_groups):
        raise ValueError("Unexpected total fitting weight")

    return weights, factors