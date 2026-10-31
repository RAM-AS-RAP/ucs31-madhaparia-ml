"""Select and freeze the research uncertainty policy on development data."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash
from scripts.candidate_calibration_data import validate_calibration_partitions
from scripts.candidate_policy_search import SETTINGS, search_thresholds
from scripts.candidate_probability_calibration import temperature_scale


EXPERIMENT = ROOT / "data/candidate_experiment_v1"
CALIBRATION = EXPERIMENT / "calibration"
COMPARISON = CALIBRATION / "policy_development_comparison"
OUTPUT = CALIBRATION / "uncertainty_policy"


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
        "comparison_manifest": COMPARISON / "comparison_manifest.json",
        "calibration_selection": COMPARISON / "selection.json",
        "temperature_fit": (
            CALIBRATION / "temperature_fit/temperature_fit.json"
        ),
        "model": EXPERIMENT / "models/rf_group_weighted/model.joblib",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "protocol": ROOT / "docs/CANDIDATE_CALIBRATION_PROTOCOL.md",
        "calibration_manifest": CALIBRATION / "partition_manifest.json",
        "calibration_partitions": CALIBRATION / "partitions.csv",
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
        "comparison_script": ROOT / "scripts/compare_candidate_calibration.py",
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
        "uncertainty_helper": ROOT / "scripts/candidate_uncertainty_policy.py",
        "search_helper": ROOT / "scripts/candidate_policy_search.py",
        "selection_script": Path(__file__).resolve(),
    }

    for setting in SETTINGS:
        inputs[f"predictions_{setting}"] = (
            COMPARISON / "predictions" / f"{setting}.npz"
        )

    def fingerprints():
        return {name: file_hash(path) for name, path in inputs.items()}

    print("Checking development predictions and dependencies...", flush=True)
    before = fingerprints()

    comparison = read_json(inputs["comparison_manifest"])
    selection = read_json(inputs["calibration_selection"])
    fitted = read_json(inputs["temperature_fit"])
    partition_manifest = read_json(inputs["calibration_manifest"])

    if (
        comparison["partition"] != "policy_development"
        or comparison["source_groups"] != 16720
        or comparison["settings"] != list(SETTINGS)
    ):
        raise ValueError("Unexpected comparison provenance")

    if (
        before["calibration_selection"]
        != comparison["output_sha256"]["selection.json"]
    ):
        raise ValueError("Calibration selection fingerprint mismatch")

    for name in (
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
        "comparison_script",
        "hash_helper",
    ):
        if before[name] != comparison["input_sha256"][name]:
            raise ValueError(f"Comparison dependency changed: {name}")

    if (
        before["calibration_partitions"]
        != partition_manifest["partitions_sha256"]
    ):
        raise ValueError("Calibration partition fingerprint mismatch")

    if (
        selection["selection_partition"] != "policy_development"
        or selection["candidate_name"] != "rf_group_weighted"
        or selection["model_sha256"] != before["model"]
        or selection["protocol_sha256"] != before["protocol"]
        or selection["temperature_fit_sha256"] != before["temperature_fit"]
    ):
        raise ValueError("Calibration selection provenance mismatch")

    representation = selection["selected_representation"]
    if representation not in {"original", "temperature_scaled"}:
        raise ValueError("Unknown selected probability representation")

    temperature = float(fitted["fit"]["temperature"])
    if selection["fitted_temperature"] != temperature:
        raise ValueError("Fitted temperature changed")

    if representation == "temperature_scaled":
        if selection["selected_temperature"] != temperature:
            raise ValueError("Selected temperature mismatch")
        probability_key = "temperature_scaled_probabilities"
    else:
        if selection["selected_temperature"] is not None:
            raise ValueError("Original probabilities must not apply temperature")
        probability_key = "original_probabilities"

    partitions = pd.read_csv(inputs["calibration_partitions"], dtype=str)
    validate_calibration_partitions(partitions)

    development = partitions.loc[
        partitions["partition"].eq("policy_development")
    ].set_index("source_group_id")

    expected_count = partition_manifest["group_counts"]["policy_development"]
    if expected_count != 16720 or len(development) != expected_count:
        raise ValueError("Unexpected development group count")

    expected_ids = set(development.index)
    expected_classes = list(fitted["classes"])
    class_positions = {
        label: index for index, label in enumerate(expected_classes)
    }

    views = {}

    for setting in SETTINGS:
        filename = f"predictions/{setting}.npz"
        if (
            before[f"predictions_{setting}"]
            != comparison["output_sha256"][filename]
        ):
            raise ValueError(f"Prediction archive changed: {setting}")

        with np.load(
            inputs[f"predictions_{setting}"], allow_pickle=False
        ) as archive:
            ids = archive["source_group_ids"].copy()
            actual = archive["actual"].copy()
            classes = archive["classes"].tolist()
            original = archive["original_probabilities"].copy()
            scaled = archive["temperature_scaled_probabilities"].copy()
            units = archive["reported_unit_count"].copy()

        if classes != expected_classes:
            raise ValueError(f"Prediction class order mismatch: {setting}")

        if (
            ids.shape != (expected_count,)
            or len(set(ids.tolist())) != expected_count
            or set(ids.tolist()) != expected_ids
        ):
            raise ValueError(f"Missing or foreign development groups: {setting}")

        expected_actual = development.loc[
            ids.tolist(), "canonical_label"
        ].to_numpy(dtype=str)

        if not np.array_equal(actual, expected_actual):
            raise ValueError(f"Prediction labels disagree with groups: {setting}")

        recomputed = temperature_scale(original, temperature)
        if scaled.shape != recomputed.shape or not np.allclose(
            scaled, recomputed, rtol=0, atol=1e-12
        ):
            raise ValueError(f"Calibrated prediction mismatch: {setting}")

        views[setting] = {
            "partition": "policy_development",
            "source_group_ids": ids,
            "classes": classes,
            "target_indices": np.asarray(
                [class_positions[label] for label in actual],
                dtype=np.int64,
            ),
            "probabilities": (
                scaled
                if probability_key == "temperature_scaled_probabilities"
                else original
            ),
            "reported_unit_count": units,
        }

    print("Development groups per view:", expected_count)
    print("Probability representation:", representation)
    print("Searching the 250 predefined threshold pairs...", flush=True)

    result = search_thresholds(views)
    policy = result["policy"]

    policy.update({
        "policy_id": "candidate_v1_research_uncertainty_v1",
        "candidate_name": "rf_group_weighted",
        "probability_representation": representation,
        "temperature": selection["selected_temperature"],
        "classes": expected_classes,
        "feature_names": fitted["feature_names"],
        "model_sha256": before["model"],
        "candidate_schema_sha256": before["candidate_schema"],
        "protocol_sha256": before["protocol"],
        "temperature_fit_sha256": before["temperature_fit"],
        "calibration_selection_sha256": before["calibration_selection"],
        "evaluation_status": "not_yet_evaluated_on_uncertainty_evaluation",
    })

    if before != fingerprints():
        raise RuntimeError("An input changed during policy selection")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    output_hashes = {}

    for filename, rows in (
        ("threshold_grid.csv", result["grid_summary"]),
        ("threshold_grid_by_view.csv", result["grid_by_view"]),
        ("development_summary.csv", result["selected_view_summary"]),
    ):
        path = OUTPUT / filename
        pd.DataFrame(rows).to_csv(
            path, index=False, lineterminator="\n"
        )
        output_hashes[filename] = file_hash(path)

    policy_path = OUTPUT / "policy.json"
    write_json(policy_path, policy)
    output_hashes["policy.json"] = file_hash(policy_path)

    write_json(
        OUTPUT / "search_manifest.json",
        {
            "issue": 49,
            "status": "research_policy_frozen_before_evaluation",
            "selection_partition": "policy_development",
            "source_groups": expected_count,
            "input_sha256": before,
            "output_sha256": output_hashes,
            "notes": [
                "Only policy-development predictions entered threshold search.",
                "All 250 prespecified threshold pairs were evaluated.",
                "Every required view must satisfy the development criteria.",
                "Development intervals are descriptive after selection.",
                "The model and fitted temperature were not changed.",
                "Uncertainty-evaluation predictions were not inspected.",
                "The final test split was not inspected.",
                "The saved policy does not authorize deployment.",
            ],
        },
    )

    print("\nRESEARCH UNCERTAINTY POLICY SAVED")
    print("Mode:", policy["mode"])
    print("Qualifying pairs:", policy["qualifying_threshold_pairs"])

    if policy["mode"] == "thresholds":
        print("Minimum reporting units:", policy["min_reporting_units"])
        print("Minimum top score:", policy["min_score"])
        print("Minimum top-two margin:", policy["min_margin"])
    else:
        print("No threshold pair met the requirements.")
        print("Default action: withhold all leading conditions.")

    for row in result["selected_view_summary"]:
        accuracy = row["accepted_accuracy"]
        lower = row["accepted_accuracy_wilson_lower"]

        accuracy_text = (
            "undefined" if accuracy is None else f"{100 * accuracy:.2f}%"
        )
        lower_text = (
            "undefined" if lower is None else f"{100 * lower:.2f}%"
        )

        print(
            f"  {row['view']}: "
            f"coverage={100 * row['coverage']:.2f}% | "
            f"accepted accuracy={accuracy_text} | "
            f"Wilson lower={lower_text}"
        )

    print("Input/model fingerprints unchanged.")
    print("Policy frozen; independent evaluation remains pending.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()