"""Verify saved candidate CSVs and summarize remaining feature collisions."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_overlap import (
    add_record,
    new_index,
    summarize_index,
    summarize_overlap,
)
from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_feature_builder import load_candidate_schema


DATA = ROOT / "data/candidate_v1"
REPORT = ROOT / "reports/candidate_dataset/verification.json"


def main():
    manifest = json.loads(
        (DATA / "manifest.json").read_text(encoding="utf-8")
    )
    schema = load_candidate_schema()

    assert manifest["schema"] == schema

    scope = json.loads(
        (ROOT / "reports/condition_scope_audit/scope_audit.json")
        .read_text(encoding="utf-8")
    )
    assert (
        manifest["label_mapping"]
        == scope["proposed_source_label_mapping"]
    )

    inputs = {
        "train_source": ROOT / "data/raw/release_train_patients",
        "validation_source": ROOT / "data/raw/release_validate_patients",
        "candidate_feature_builder": (
            ROOT / "scripts/candidate_feature_builder.py"
        ),
        "source_audit_helpers": (
            ROOT / "scripts/audit_candidate_source_records.py"
        ),
        "overlap_audit_helpers": (
            ROOT / "scripts/audit_candidate_overlap.py"
        ),
        "dataset_builder": ROOT / "scripts/build_candidate_dataset.py",
    }

    for name, path in inputs.items():
        assert file_hash(path) == manifest["input_sha256"][name], name

    for name, expected in manifest["output_sha256"].items():
        assert file_hash(DATA / name) == expected, name

    print("PASS: source, preparation-code and output fingerprints.")

    groups = pd.read_csv(DATA / "source_groups.csv")
    groups = groups.set_index("source_group_id")
    assert groups.index.is_unique, "Duplicate source-group IDs"

    assert groups["split"].isin(["train", "validation"]).all()
    assert (groups["source_group_size"] >= 1).all()
    assert (groups["available_reporting_units"] >= 1).all()
    assert (
        groups["source_label"].map(manifest["label_mapping"])
        == groups["canonical_label"]
    ).all()

    features = schema["feature_names"]
    concepts = schema["symptom_features"]
    present_columns = [f"{code}__present" for code in concepts]
    absent_columns = [f"{code}__absent" for code in concepts]

    expected_columns = [
        "source_group_id",
        "variant_id",
        "canonical_label",
        "source_group_weight",
        "reported_unit_count",
        *features,
    ]

    head_index = concepts.index("head_region_pain")
    chest_index = concepts.index("chest_region_pain")
    pain_index = concepts.index("E_53")
    bit_weights = np.left_shift(
        np.uint64(1), np.arange(len(concepts), dtype=np.uint64)
    )

    indexes = {}
    verified = {}

    for split in ("train", "validation"):
        print(f"Verifying {split} CSV...", flush=True)

        expected_groups = groups[groups["split"] == split]
        expected_partial = (
            set(expected_groups.index[
                expected_groups["available_reporting_units"] >= 2
            ])
            if split == "train"
            else set()
        )

        full_records = {}
        partial_seen = set()
        collision_index = new_index()
        label_counts = Counter()
        variant_counts = Counter()
        unit_counts = Counter()
        row_count = 0

        for chunk in pd.read_csv(
            DATA / f"{split}.csv", chunksize=50000
        ):
            assert chunk.columns.tolist() == expected_columns

            ids = chunk["source_group_id"]
            assert ids.isin(expected_groups.index).all()

            source_groups = expected_groups.loc[ids]
            labels = chunk["canonical_label"].to_numpy()
            assert np.array_equal(
                labels, source_groups["canonical_label"].to_numpy()
            )

            numeric = chunk[features].to_numpy(dtype=float)
            assert np.isfinite(numeric).all()
            assert ((numeric[:, 0] >= 0) & (numeric[:, 0] <= 120)).all()
            assert np.isin(numeric[:, 1:], [0, 1]).all()
            assert (chunk[absent_columns].to_numpy() == 0).all()

            present = chunk[present_columns].to_numpy(dtype=np.uint64)
            has_region = (
                (present[:, head_index] == 1)
                | (present[:, chest_index] == 1)
            )
            assert (present[has_region, pain_index] == 1).all()

            actual_units = present.sum(axis=1).astype(int)
            actual_units -= has_region.astype(int)

            assert np.array_equal(
                actual_units, chunk["reported_unit_count"].to_numpy()
            )
            assert (actual_units >= 1).all()

            available = source_groups[
                "available_reporting_units"
            ].to_numpy()

            kinds = chunk["variant_id"].to_numpy()
            assert np.isin(kinds, ["full", "partial"]).all()

            is_full = kinds == "full"
            assert (actual_units[is_full] == available[is_full]).all()

            if split == "validation":
                assert is_full.all()

            is_partial = ~is_full
            assert (
                actual_units[is_partial]
                <= np.minimum(5, available[is_partial] - 1)
            ).all()

            expected_weights = (
                np.where(available >= 2, 0.5, 1.0)
                if split == "train"
                else np.ones(len(chunk))
            )
            assert np.array_equal(
                chunk["source_group_weight"].to_numpy(),
                expected_weights,
            )

            masks = (present * bit_weights).sum(axis=1)

            for group_id, kind, age, sex, mask, label in zip(
                ids,
                kinds,
                numeric[:, 0],
                numeric[:, 1],
                masks,
                labels,
            ):
                mask = int(mask)

                if kind == "full":
                    assert group_id not in full_records
                    full_records[group_id] = (age, sex, mask)

                    # All absent columns are zero, so this tuple uniquely
                    # describes the full candidate input vector.
                    add_record(
                        collision_index, (age, sex, mask), label
                    )
                else:
                    assert group_id in full_records
                    assert group_id not in partial_seen
                    full_age, full_sex, full_mask = full_records[group_id]

                    assert (age, sex) == (full_age, full_sex)
                    assert mask != full_mask
                    assert mask & full_mask == mask
                    partial_seen.add(group_id)

            row_count += len(chunk)
            label_counts.update(
                chunk["canonical_label"].value_counts().to_dict()
            )
            variant_counts.update(
                chunk["variant_id"].value_counts().to_dict()
            )
            unit_counts.update(
                chunk["reported_unit_count"].value_counts().to_dict()
            )

        assert set(full_records) == set(expected_groups.index)
        assert partial_seen == expected_partial

        recorded = manifest["splits"][split]
        assert row_count == recorded["totals"]["scenario_rows"]
        assert len(full_records) == recorded["totals"]["retained_source_groups"]
        assert dict(label_counts) == recorded["scenarios_by_canonical_label"]
        assert dict(variant_counts) == recorded["variants_by_type"]
        assert {
            str(count): frequency for count, frequency in unit_counts.items()
        } == recorded["reporting_unit_count_histogram"]

        indexes[split] = collision_index
        verified[split] = {
            "scenario_rows": row_count,
            "source_groups": len(full_records),
            "partial_variants": len(partial_seen),
            "full_candidate_input_collisions": summarize_index(
                collision_index
            ),
        }

        print(f"PASS: {split} rows, provenance, features and group weights.")

    report = {
        "status": "passed",
        "manifest_sha256": file_hash(DATA / "manifest.json"),
        "verification_script_sha256": file_hash(Path(__file__).resolve()),
        "python_libraries": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "verified": verified,
        "full_candidate_input_overlap_after_filtering": summarize_overlap(
            indexes["train"], indexes["validation"]
        ),
        "notes": [
            "Full candidate-input collisions are retained and reported.",
            "Input-vector matches do not establish shared patient identity.",
            "Source group IDs are unique across the retained splits.",
            "Each training group contributes total sample weight 1.",
            "Verification does not establish predictive performance.",
        ],
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFULL CANDIDATE INPUTS AFTER FILTERING")
    for split, values in verified.items():
        print(split, values["full_candidate_input_collisions"])

    print("\nCROSS-SPLIT CANDIDATE INPUT OVERLAP")
    print(report["full_candidate_input_overlap_after_filtering"])
    print("\nVerification report saved to:", REPORT)


if __name__ == "__main__":
    main()