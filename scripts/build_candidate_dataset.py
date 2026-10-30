"""Build the initial DDXPlus candidate dataset.

Uses training and validation sources only.
Existing datasets and model artifacts are never overwritten.
"""

import argparse
import ast
import csv
import json
import platform
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_overlap import input_signature
from scripts.audit_candidate_source_records import (
    file_hash,
    valid_demographics,
)
from scripts.candidate_feature_builder import (
    derive_candidate_state,
    get_reporting_units,
    load_candidate_schema,
    make_training_variants,
    mask_candidate_state,
)


def read_source(path, schema, label_mapping):
    """Yield in-scope records with their original source row numbers."""

    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"AGE", "SEX", "PATHOLOGY", "EVIDENCES"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns: {path}")

        for row_number, row in enumerate(reader, start=1):
            source_label = row["PATHOLOGY"]
            if source_label not in label_mapping:
                continue

            if not valid_demographics(row):
                raise ValueError(
                    f"Invalid demographics: {path.name}, row {row_number}"
                )

            evidence = ast.literal_eval(row["EVIDENCES"])
            state = derive_candidate_state(evidence, schema)
            age = float(row["AGE"])
            sex = row["SEX"]

            yield {
                "source_row": row_number,
                "source_label": source_label,
                "age": age,
                "sex": sex,
                "state": state,
                "signature": input_signature(age, sex, evidence),
            }


def collect_groups(path, schema, label_mapping):
    """Index eligible groups without retaining complete records in memory."""

    groups = {}
    counts = Counter()

    for record in read_source(path, schema, label_mapping):
        counts["in_scope_rows"] += 1

        if not record["state"]["present"]:
            counts["no_supported_positive_concepts"] += 1
            continue

        counts["eligible_rows"] += 1
        signature = record["signature"]
        label = record["source_label"]

        if signature not in groups:
            groups[signature] = {
                "first_source_row": record["source_row"],
                "source_label": label,
                "source_group_size": 1,
            }
        else:
            group = groups[signature]
            if group["source_label"] != label:
                raise ValueError(
                    "Conflicting labels within a normalized source group"
                )
            group["source_group_size"] += 1

    counts["unique_eligible_groups"] = len(groups)
    return groups, dict(counts)


def numerical_values(age, sex, state, schema):
    """Encode values in the exact candidate schema order."""

    # Validate the state partition and required pain parents.
    get_reporting_units(state, schema)

    present = set(state["present"])
    absent = set(state["absent"])
    codes = schema["symptom_features"]

    values = (
        [age, int(sex == "M")]
        + [int(code in present) for code in codes]
        + [int(code in absent) for code in codes]
    )

    if len(values) != schema["feature_count"]:
        raise ValueError("Encoded feature count does not match schema")

    return values


def build_candidate_dataset(
    *,
    train_path,
    validation_path,
    output_dir,
    schema,
    label_mapping,
    seed=42,
):
    """Build a fresh output directory; refuse to overwrite any directory."""

    csv.field_size_limit(2_147_483_647)

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    if not label_mapping or len(set(label_mapping.values())) != len(
        label_mapping
    ):
        raise ValueError("Expected a non-empty, one-to-one label mapping")

    paths = {
        "train": Path(train_path).resolve(),
        "validation": Path(validation_path).resolve(),
    }
    output_dir = Path(output_dir).resolve()

    if paths["train"] == paths["validation"]:
        raise ValueError("Training and validation must be different files")

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}"
        )

    dependency_paths = {
        "train_source": paths["train"],
        "validation_source": paths["validation"],
        "candidate_feature_builder": (
            ROOT / "scripts/candidate_feature_builder.py"
        ),
        "source_audit_helpers": (
            ROOT / "scripts/audit_candidate_source_records.py"
        ),
        "overlap_audit_helpers": (
            ROOT / "scripts/audit_candidate_overlap.py"
        ),
        "dataset_builder": Path(__file__).resolve(),
    }

    def fingerprints():
        return {
            name: file_hash(path)
            for name, path in dependency_paths.items()
        }

    before = fingerprints()

    indexes = {}
    source_counts = {}

    for split, path in paths.items():
        print(f"Indexing {split} source groups...", flush=True)
        indexes[split], source_counts[split] = collect_groups(
            path, schema, label_mapping
        )

    shared = indexes["train"].keys() & indexes["validation"].keys()

    for signature in shared:
        if (
            indexes["train"][signature]["source_label"]
            != indexes["validation"][signature]["source_label"]
        ):
            raise ValueError(
                "Conflicting labels across a shared source-input group"
            )

    print("Shared source groups excluded from training:", len(shared))

    output_dir.mkdir(parents=True, exist_ok=False)

    scenario_metadata = [
        "source_group_id",
        "variant_id",
        "canonical_label",
        "source_group_weight",
        "reported_unit_count",
    ]

    group_columns = [
        "source_group_id",
        "split",
        "source_file",
        "source_row",
        "source_label",
        "canonical_label",
        "source_group_size",
        "available_reporting_units",
    ]

    exclusion_columns = [
        "split",
        "source_file",
        "source_row",
        "source_label",
        "source_group_id",
        "reason",
    ]

    summaries = {}
    retained_ids = {}

    with ExitStack() as stack:
        def writer(filename, columns):
            handle = stack.enter_context(
                (output_dir / filename).open(
                    "w", encoding="utf-8", newline=""
                )
            )
            result = csv.writer(handle, lineterminator="\n")
            result.writerow(columns)
            return result

        group_writer = writer("source_groups.csv", group_columns)
        exclusion_writer = writer("excluded_rows.csv", exclusion_columns)

        for split, path in paths.items():
            print(f"Writing {split} candidate observations...", flush=True)

            scenario_writer = writer(
                f"{split}.csv",
                scenario_metadata + schema["feature_names"],
            )

            totals = Counter()
            groups_by_label = Counter()
            scenarios_by_label = Counter()
            variants_by_type = Counter()
            reporting_counts = Counter()
            weight_by_label = Counter()
            retained_ids[split] = set()

            for record in read_source(path, schema, label_mapping):
                signature = record["signature"]
                group_id = signature.hex()
                source_label = record["source_label"]
                state = record["state"]
                reason = None

                if not state["present"]:
                    reason = "no_supported_positive_concepts"
                elif split == "train" and signature in shared:
                    reason = "source_group_reserved_for_validation"
                elif (
                    record["source_row"]
                    != indexes[split][signature]["first_source_row"]
                ):
                    reason = "repeated_source_input"

                if reason:
                    totals[f"excluded_rows:{reason}"] += 1
                    exclusion_writer.writerow([
                        split,
                        path.name,
                        record["source_row"],
                        source_label,
                        group_id,
                        reason,
                    ])
                    continue

                canonical_label = label_mapping[source_label]
                units = get_reporting_units(state, schema)

                if split == "train":
                    variants = make_training_variants(
                        state,
                        schema,
                        source_group_id=group_id,
                        seed=seed,
                    )
                else:
                    full = mask_candidate_state(
                        state,
                        schema,
                        reported_unit_count=len(units),
                        seed=f"{seed}:{group_id}:full",
                    )
                    full.update({
                        "variant_id": "full",
                        "source_group_id": group_id,
                        "source_group_weight": 1.0,
                    })
                    variants = [full]

                if sum(
                    item["source_group_weight"] for item in variants
                ) != 1.0:
                    raise ValueError("Source-group weights do not sum to 1")

                seen_variants = set()

                for variant in variants:
                    values = numerical_values(
                        record["age"], record["sex"], variant, schema
                    )
                    vector = tuple(values)

                    if vector in seen_variants:
                        raise ValueError("Duplicate variant within a group")
                    seen_variants.add(vector)

                    scenario_writer.writerow([
                        group_id,
                        variant["variant_id"],
                        canonical_label,
                        variant["source_group_weight"],
                        variant["reported_unit_count"],
                        *values,
                    ])

                    totals["scenario_rows"] += 1
                    scenarios_by_label[canonical_label] += 1
                    variants_by_type[variant["variant_id"]] += 1
                    reporting_counts[variant["reported_unit_count"]] += 1
                    weight_by_label[canonical_label] += (
                        variant["source_group_weight"]
                    )

                group_writer.writerow([
                    group_id,
                    split,
                    path.name,
                    record["source_row"],
                    source_label,
                    canonical_label,
                    indexes[split][signature]["source_group_size"],
                    len(units),
                ])

                retained_ids[split].add(signature)
                totals["retained_source_groups"] += 1
                groups_by_label[canonical_label] += 1

            expected_groups = len(indexes[split])
            if split == "train":
                expected_groups -= len(shared)

            if totals["retained_source_groups"] != expected_groups:
                raise ValueError("Unexpected retained source-group count")

            if set(groups_by_label) != set(label_mapping.values()):
                raise ValueError(
                    f"{split} lost coverage of a candidate condition"
                )

            for label, count in groups_by_label.items():
                if weight_by_label[label] != count:
                    raise ValueError("Incorrect aggregate source-group weight")

            summaries[split] = {
                "totals": dict(totals),
                "groups_by_canonical_label": dict(sorted(
                    groups_by_label.items()
                )),
                "scenarios_by_canonical_label": dict(sorted(
                    scenarios_by_label.items()
                )),
                "variants_by_type": dict(variants_by_type),
                "reporting_unit_count_histogram": {
                    str(count): frequency
                    for count, frequency in sorted(reporting_counts.items())
                },
                "weight_by_canonical_label": dict(sorted(
                    weight_by_label.items()
                )),
            }

    if retained_ids["train"] & retained_ids["validation"]:
        raise ValueError("Retained source groups overlap between splits")

    if before != fingerprints():
        raise RuntimeError("An input changed during the build; discard outputs")

    output_names = [
        "train.csv",
        "validation.csv",
        "source_groups.csv",
        "excluded_rows.csv",
    ]

    manifest = {
        "status": "candidate_dataset_built_not_trained",
        "seed": seed,
        "python_version": platform.python_version(),
        "schema": schema,
        "label_mapping": label_mapping,
        "source_filenames": {
            split: path.name for split, path in paths.items()
        },
        "input_sha256": before,
        "source_scan_counts": source_counts,
        "shared_source_groups_removed_from_training": len(shared),
        "retained_train_validation_source_overlap": 0,
        "splits": summaries,
        "output_sha256": {
            name: file_hash(output_dir / name)
            for name in output_names
        },
        "limitations": [
            "DDXPlus is simulated data; Kenyan performance is unverified.",
            "Input groups are not verified patient identifiers.",
            "Candidate-feature collisions are retained.",
            "Only positive evidence is used; hidden concepts remain unknown.",
            "COVID-19 is not included in this initial dataset.",
            "The final test split was not inspected.",
            "Training must select schema.feature_names as model inputs.",
            "Training must consume source_group_weight as sample weight.",
        ],
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nCANDIDATE DATASET BUILT")
    for split, summary in summaries.items():
        print(split, summary["totals"])
    print("Retained source-group overlap: 0")
    print("Output:", output_dir)

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/candidate_v1",
    )
    args = parser.parse_args()

    schema = load_candidate_schema()

    scope = json.loads(
        (ROOT / "reports/condition_scope_audit/scope_audit.json")
        .read_text(encoding="utf-8")
    )
    mapping = scope["proposed_source_label_mapping"]

    original_labels = {
        item["name"]
        for item in json.loads(
            (ROOT / "data/processed/diseases.json")
            .read_text(encoding="utf-8")
        )
    }

    if set(mapping) != original_labels or len(mapping) != 21:
        raise ValueError("Unexpected source condition scope")

    if mapping["Allergic sinusitis"] != "Allergic rhinitis":
        raise ValueError("Expected allergy-label mapping")

    build_candidate_dataset(
        train_path=ROOT / "data/raw/release_train_patients",
        validation_path=ROOT / "data/raw/release_validate_patients",
        output_dir=args.output_dir,
        schema=schema,
        label_mapping=mapping,
        seed=42,
    )


if __name__ == "__main__":
    main()