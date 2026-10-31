"""Evaluate the frozen policy on the reserved uncertainty-evaluation groups."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_calibration_data import validate_calibration_partitions
from scripts.candidate_calibration_metrics import (
    SETTINGS,
    evaluate_probability_view,
)
from scripts.candidate_probability_calibration import temperature_scale
from scripts.candidate_selective_evaluation import evaluate_policy_view


EXPERIMENT = ROOT / "data/candidate_experiment_v1"
CALIBRATION = EXPERIMENT / "calibration"
VIEWS = CALIBRATION / "views"
POLICY_FOLDER = CALIBRATION / "uncertainty_policy"
PARTITION = "uncertainty_evaluation"
OUTPUT = CALIBRATION / "uncertainty_evaluation"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT}")

    inputs = {
        "policy": POLICY_FOLDER / "policy.json",
        "search_manifest": POLICY_FOLDER / "search_manifest.json",
        "calibration_selection": (
            CALIBRATION / "policy_development_comparison/selection.json"
        ),
        "temperature_fit": (
            CALIBRATION / "temperature_fit/temperature_fit.json"
        ),
        "model": EXPERIMENT / "models/rf_group_weighted/model.joblib",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "protocol": ROOT / "docs/CANDIDATE_CALIBRATION_PROTOCOL.md",
        "calibration_manifest": CALIBRATION / "partition_manifest.json",
        "calibration_partitions": CALIBRATION / "partitions.csv",
        "view_manifest": VIEWS / "view_manifest.json",
        "calibration_helper": (
            ROOT / "scripts/candidate_probability_calibration.py"
        ),
        "calibration_data_helper": (
            ROOT / "scripts/candidate_calibration_data.py"
        ),
        "metrics_helper": ROOT / "scripts/candidate_calibration_metrics.py",
        "classification_metrics_helper": (
            ROOT / "scripts/candidate_comparison_metrics.py"
        ),
        "uncertainty_helper": ROOT / "scripts/candidate_uncertainty_policy.py",
        "search_helper": ROOT / "scripts/candidate_policy_search.py",
        "selection_script": (
            ROOT / "scripts/select_candidate_uncertainty_policy.py"
        ),
        "selective_evaluation_helper": (
            ROOT / "scripts/candidate_selective_evaluation.py"
        ),
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
        "evaluation_script": Path(__file__).resolve(),
    }

    for setting in SETTINGS:
        inputs[f"evaluation_view_{setting}"] = (
            VIEWS / PARTITION / f"{setting}.csv"
        )

    def fingerprints():
        return {name: file_hash(path) for name, path in inputs.items()}

    print("Checking frozen policy and evaluation inputs...", flush=True)
    before = fingerprints()

    policy = read_json(inputs["policy"])
    search_manifest = read_json(inputs["search_manifest"])
    selection = read_json(inputs["calibration_selection"])
    fitted = read_json(inputs["temperature_fit"])
    partition_manifest = read_json(inputs["calibration_manifest"])
    view_manifest = read_json(inputs["view_manifest"])
    schema = read_json(inputs["candidate_schema"])

    if before["policy"] != search_manifest["output_sha256"]["policy.json"]:
        raise ValueError("Frozen policy fingerprint mismatch")

    if search_manifest["selection_partition"] != "policy_development":
        raise ValueError("Policy was not selected on the expected partition")

    for name in (
        "calibration_selection",
        "temperature_fit",
        "model",
        "candidate_schema",
        "protocol",
        "calibration_manifest",
        "calibration_partitions",
        "calibration_helper",
        "calibration_data_helper",
        "metrics_helper",
        "classification_metrics_helper",
        "uncertainty_helper",
        "search_helper",
        "selection_script",
        "hash_helper",
    ):
        if before[name] != search_manifest["input_sha256"][name]:
            raise ValueError(f"Policy dependency changed: {name}")

    for name in (
        "model",
        "candidate_schema",
        "protocol",
        "temperature_fit",
        "calibration_selection",
    ):
        if before[name] != policy[f"{name}_sha256"]:
            raise ValueError(f"Policy fingerprint reference mismatch: {name}")

    if (
        before["view_manifest"]
        != fitted["input_sha256"]["view_manifest"]
    ):
        raise ValueError("Observation-view manifest changed")

    if (
        before["calibration_partitions"]
        != partition_manifest["partitions_sha256"]
    ):
        raise ValueError("Calibration partition CSV changed")

    if policy["probability_representation"] != selection["selected_representation"]:
        raise ValueError("Probability representation changed")

    representation = policy["probability_representation"]
    if representation not in {"original", "temperature_scaled"}:
        raise ValueError("Unknown probability representation")

    temperature = float(fitted["fit"]["temperature"])

    if representation == "temperature_scaled":
        if policy["temperature"] != temperature:
            raise ValueError("Frozen temperature changed")
    elif policy["temperature"] is not None:
        raise ValueError("Original-probability policy must not apply temperature")

    classes = list(policy["classes"])
    features = list(policy["feature_names"])

    if classes != fitted["classes"] or len(classes) != 21:
        raise ValueError("Frozen class definitions changed")

    if (
        features != fitted["feature_names"]
        or features != schema["feature_names"]
        or len(features) != 70
    ):
        raise ValueError("Frozen feature definitions changed")

    required_views = ("full", "up_to_2", "up_to_3", "up_to_5")
    if policy["required_views"] != list(required_views):
        raise ValueError("Required evaluation views differ from the protocol")

    partitions = pd.read_csv(inputs["calibration_partitions"], dtype=str)
    validate_calibration_partitions(partitions)

    evaluation_groups = partitions.loc[
        partitions["partition"].eq(PARTITION)
    ].set_index("source_group_id")

    expected_count = partition_manifest["group_counts"][PARTITION]
    if len(evaluation_groups) != expected_count or expected_count != 16741:
        raise ValueError("Unexpected uncertainty-evaluation group count")

    expected_ids = set(evaluation_groups.index)
    expected_support = partition_manifest[
        "groups_by_partition_and_class"
    ][PARTITION]

    if (
        evaluation_groups["canonical_label"].value_counts().to_dict()
        != expected_support
    ):
        raise ValueError("Evaluation condition support mismatch")

    frames = {}

    # Validate every evaluation view before calculating predictions.
    for setting in SETTINGS:
        details = view_manifest["views"][PARTITION][setting]

        if details["filename"] != f"{PARTITION}/{setting}.csv":
            raise ValueError(f"Unexpected evaluation filename: {setting}")

        if before[f"evaluation_view_{setting}"] != details["sha256"]:
            raise ValueError(f"Evaluation view fingerprint mismatch: {setting}")

        frame = pd.read_csv(inputs[f"evaluation_view_{setting}"])

        if len(frame) != expected_count or len(frame) != details["rows"]:
            raise ValueError(f"Unexpected evaluation view size: {setting}")

        if (
            not frame["source_group_id"].is_unique
            or set(frame["source_group_id"]) != expected_ids
        ):
            raise ValueError(f"Missing or foreign evaluation groups: {setting}")

        if not frame["calibration_partition"].eq(PARTITION).all():
            raise ValueError(f"Unexpected partition in evaluation view: {setting}")

        if not frame["variant_id"].eq(setting).all():
            raise ValueError(f"Unexpected evaluation variant: {setting}")

        expected_labels = frame["source_group_id"].map(
            evaluation_groups["canonical_label"]
        )
        if not expected_labels.eq(frame["canonical_label"]).all():
            raise ValueError(f"Evaluation label mismatch: {setting}")

        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite evaluation features: {setting}")

        frames[setting] = frame

    model = joblib.load(inputs["model"])

    if list(model.classes_) != classes:
        raise ValueError("Loaded model class order mismatch")

    if model.n_features_in_ != len(features):
        raise ValueError("Loaded model feature count mismatch")

    if (
        hasattr(model, "feature_names_in_")
        and list(model.feature_names_in_) != features
    ):
        raise ValueError("Loaded model feature order mismatch")

    print("Evaluation groups per view:", expected_count)
    print("Frozen policy mode:", policy["mode"])
    print("Probability representation:", representation)
    print("No calibration or threshold fitting will be performed.")

    results = {}
    archives = {}
    policy_rows = []
    condition_rows = []
    unit_rows = []
    probability_rows = []
    probability_condition_rows = []
    reliability_rows = []

    for setting in SETTINGS:
        print(f"\nEvaluating frozen policy / {setting}...", flush=True)
        frame = frames[setting]
        blocks = []

        for start in range(0, len(frame), 5000):
            batch = frame.iloc[start:start + 5000]
            blocks.append(model.predict_proba(batch[features]))

        original = np.vstack(blocks)
        scaled = temperature_scale(original, temperature)
        actual = frame["canonical_label"].to_numpy(dtype=str)
        units = frame["reported_unit_count"].to_numpy()

        probability_reports = {}

        for name, probabilities in (
            ("original", original),
            ("temperature_scaled", scaled),
        ):
            report = evaluate_probability_view(
                actual, probabilities, classes
            )
            probability_reports[name] = report

            context = {
                "partition": PARTITION,
                "view": setting,
                "representation": name,
            }
            probability_rows.append({**context, **report["summary"]})
            probability_condition_rows.extend(
                {**context, **row} for row in report["per_condition"]
            )
            reliability_rows.extend(
                {**context, **row} for row in report["reliability_bins"]
            )

        selected_probabilities = (
            scaled if representation == "temperature_scaled" else original
        )

        policy_report, accepted = evaluate_policy_view(
            actual,
            selected_probabilities,
            classes,
            units,
            policy,
        )

        required = setting in required_views
        policy_report["summary"]["required_for_research_target"] = required

        context = {"partition": PARTITION, "view": setting}
        policy_rows.append({**context, **policy_report["summary"]})
        condition_rows.extend(
            {**context, **row} for row in policy_report["per_condition"]
        )
        unit_rows.extend(
            {**context, **row} for row in policy_report["by_reporting_units"]
        )

        results[setting] = {
            "probability_metrics": probability_reports,
            "policy_metrics": policy_report,
        }

        archives[setting] = {
            "source_group_ids": frame["source_group_id"].to_numpy(dtype=str),
            "actual": actual,
            "classes": np.asarray(classes, dtype=str),
            "original_probabilities": original,
            "temperature_scaled_probabilities": scaled,
            "reported_unit_count": units,
            "accepted": accepted,
        }

        summary = policy_report["summary"]
        accuracy = summary["accepted_accuracy"]
        lower = summary["accepted_accuracy_wilson_lower"]

        accuracy_text = (
            "undefined" if accuracy is None else f"{100 * accuracy:.2f}%"
        )
        lower_text = (
            "undefined" if lower is None else f"{100 * lower:.2f}%"
        )

        print(
            f"  Coverage={100 * summary['coverage']:.2f}% | "
            f"accepted accuracy={accuracy_text} | "
            f"Wilson lower={lower_text}"
        )

        if required:
            print(
                "  Research targets met:",
                summary["meets_count_and_accuracy_targets"],
            )
        else:
            print("  One-unit view: withholding is required by the policy.")

    failed_views = [
        setting
        for setting in required_views
        if not results[setting]["policy_metrics"]["summary"][
            "meets_count_and_accuracy_targets"
        ]
    ]

    decision = {
        "evaluation_partition": PARTITION,
        "source_groups": expected_count,
        "policy_sha256": before["policy"],
        "required_views": list(required_views),
        "failed_required_views": failed_views,
        "research_targets_met_on_all_required_views": not failed_views,
        "thresholds_changed": False,
        "temperature_refitted": False,
        "deployment_approved": False,
        "interpretation": (
            "These are research targets on held-out simulated-data groups. "
            "Per-condition support, recall and withheld cases must also "
            "be reviewed. This result does not establish clinical safety."
        ),
    }

    if before != fingerprints():
        raise RuntimeError("An input changed during frozen-policy evaluation")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "predictions").mkdir()
    output_hashes = {}

    for setting, arrays in archives.items():
        filename = f"predictions/{setting}.npz"
        path = OUTPUT / filename
        np.savez_compressed(path, **arrays)
        output_hashes[filename] = file_hash(path)

    tables = (
        ("policy_summary.csv", policy_rows),
        ("per_condition.csv", condition_rows),
        ("by_reporting_units.csv", unit_rows),
        ("probability_summary.csv", probability_rows),
        ("probability_per_condition.csv", probability_condition_rows),
        ("reliability_bins.csv", reliability_rows),
    )

    for filename, rows in tables:
        path = OUTPUT / filename
        pd.DataFrame(rows).to_csv(
            path, index=False, lineterminator="\n"
        )
        output_hashes[filename] = file_hash(path)

    for filename, content in (
        ("per_view_results.json", results),
        ("evaluation_decision.json", decision),
    ):
        path = OUTPUT / filename
        write_json(path, content)
        output_hashes[filename] = file_hash(path)

    write_json(
        OUTPUT / "evaluation_manifest.json",
        {
            "issue": 49,
            "status": "frozen_policy_evaluation_completed",
            "partition": PARTITION,
            "source_groups": expected_count,
            "settings": list(SETTINGS),
            "input_sha256": before,
            "output_sha256": output_hashes,
            "notes": [
                "Only uncertainty-evaluation groups were evaluated here.",
                "The previously saved policy was applied without retuning.",
                "Calibration parameters and model weights were unchanged.",
                "Accepted-case accuracy must be interpreted with coverage.",
                "All-case per-condition recall retains withheld cases.",
                "Repeated views are not independent patients.",
                "The final test split was not inspected.",
                "No deployed model or application was changed.",
            ],
        },
    )

    print("\nFROZEN POLICY EVALUATION COMPLETE")
    print(
        "Research targets met on all required views:",
        decision["research_targets_met_on_all_required_views"],
    )
    print("Failed required views:", failed_views)
    print("Input/model/policy fingerprints unchanged.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()