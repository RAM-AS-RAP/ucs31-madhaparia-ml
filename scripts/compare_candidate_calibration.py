"""Compare original and calibrated probabilities on policy-development data."""

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
    select_probability_representation,
)
from scripts.candidate_probability_calibration import (
    TEMPERATURE_BOUNDS,
    temperature_scale,
)


EXPERIMENT = ROOT / "data/candidate_experiment_v1"
CALIBRATION = EXPERIMENT / "calibration"
VIEWS = CALIBRATION / "views"
PARTITION = "policy_development"
OUTPUT = CALIBRATION / "policy_development_comparison"


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
        "temperature_fit": (
            CALIBRATION / "temperature_fit/temperature_fit.json"
        ),
        "model": EXPERIMENT / "models/rf_group_weighted/model.joblib",
        "training_manifest": (
            EXPERIMENT / "models/rf_group_weighted/training_manifest.json"
        ),
        "view_manifest": VIEWS / "view_manifest.json",
        "calibration_manifest": CALIBRATION / "partition_manifest.json",
        "calibration_partitions": CALIBRATION / "partitions.csv",
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "protocol": ROOT / "docs/CANDIDATE_CALIBRATION_PROTOCOL.md",
        "calibration_helper": (
            ROOT / "scripts/candidate_probability_calibration.py"
        ),
        "calibration_data_helper": (
            ROOT / "scripts/candidate_calibration_data.py"
        ),
        "fitting_script": ROOT / "scripts/fit_candidate_temperature.py",
        "metrics_helper": ROOT / "scripts/candidate_calibration_metrics.py",
        "classification_metrics_helper": (
            ROOT / "scripts/candidate_comparison_metrics.py"
        ),
        "hash_helper": ROOT / "scripts/audit_candidate_source_records.py",
        "comparison_script": Path(__file__).resolve(),
    }

    for setting in SETTINGS:
        inputs[f"development_view_{setting}"] = (
            VIEWS / PARTITION / f"{setting}.csv"
        )

    def fingerprints():
        return {name: file_hash(path) for name, path in inputs.items()}

    print("Verifying frozen temperature, model and development views...", flush=True)
    before = fingerprints()

    fitted = read_json(inputs["temperature_fit"])
    view_manifest = read_json(inputs["view_manifest"])
    partition_manifest = read_json(inputs["calibration_manifest"])
    training_manifest = read_json(inputs["training_manifest"])
    schema = read_json(inputs["candidate_schema"])

    if (
        fitted["candidate_name"] != "rf_group_weighted"
        or fitted["partition"] != "calibration_fit"
        or fitted["source_groups"] != 50174
    ):
        raise ValueError("Unexpected temperature-fitting provenance")

    for name in (
        "model",
        "training_manifest",
        "view_manifest",
        "calibration_manifest",
        "calibration_partitions",
        "candidate_schema",
        "protocol",
        "calibration_helper",
        "calibration_data_helper",
        "fitting_script",
        "hash_helper",
    ):
        if before[name] != fitted["input_sha256"][name]:
            raise ValueError(f"Temperature-fitting dependency changed: {name}")

    if before["model"] != training_manifest["model_sha256"]:
        raise ValueError("Model fingerprint differs from its training manifest")

    if (
        before["calibration_partitions"]
        != partition_manifest["partitions_sha256"]
    ):
        raise ValueError("Calibration partition fingerprint mismatch")

    temperature = float(fitted["fit"]["temperature"])
    lower, upper = TEMPERATURE_BOUNDS
    if not np.isfinite(temperature) or not lower <= temperature <= upper:
        raise ValueError("Fitted temperature is outside the protocol bounds")

    classes = list(fitted["classes"])
    features = list(fitted["feature_names"])

    if classes != training_manifest["classes"] or len(classes) != 21:
        raise ValueError("Class order differs from the frozen model")

    if (
        features != schema["feature_names"]
        or features != training_manifest["feature_names"]
        or len(features) != 70
    ):
        raise ValueError("Feature schema differs from the frozen model")

    partitions = pd.read_csv(inputs["calibration_partitions"], dtype=str)
    validate_calibration_partitions(partitions)

    development = partitions.loc[
        partitions["partition"].eq(PARTITION)
    ].set_index("source_group_id")

    expected_count = partition_manifest["group_counts"][PARTITION]
    if len(development) != expected_count or expected_count != 16720:
        raise ValueError("Unexpected policy-development group count")

    expected_ids = set(development.index)
    expected_support = partition_manifest[
        "groups_by_partition_and_class"
    ][PARTITION]

    if (
        development["canonical_label"].value_counts().to_dict()
        != expected_support
    ):
        raise ValueError("Policy-development condition support mismatch")

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

    print("Policy-development groups:", expected_count)
    print(f"Frozen temperature: {temperature:.6f}")
    print("Uncertainty-evaluation groups used: 0")

    results = {}
    archives = {}
    summary_rows = []
    condition_rows = []
    reliability_rows = []
    original_losses = {}
    scaled_losses = {}

    for setting in SETTINGS:
        print(f"\nEvaluating {setting}...", flush=True)
        details = view_manifest["views"][PARTITION][setting]

        if details["filename"] != f"{PARTITION}/{setting}.csv":
            raise ValueError(f"Unexpected view filename: {setting}")

        if before[f"development_view_{setting}"] != details["sha256"]:
            raise ValueError(f"Development view fingerprint mismatch: {setting}")

        frame = pd.read_csv(inputs[f"development_view_{setting}"])

        if len(frame) != expected_count or len(frame) != details["rows"]:
            raise ValueError(f"Unexpected observation count: {setting}")

        if (
            not frame["source_group_id"].is_unique
            or set(frame["source_group_id"]) != expected_ids
        ):
            raise ValueError(f"Missing or foreign source groups: {setting}")

        if not frame["calibration_partition"].eq(PARTITION).all():
            raise ValueError(f"Unexpected partition in view: {setting}")

        if not frame["variant_id"].eq(setting).all():
            raise ValueError(f"Unexpected variant in view: {setting}")

        expected_labels = frame["source_group_id"].map(
            development["canonical_label"]
        )
        if not expected_labels.eq(frame["canonical_label"]).all():
            raise ValueError(f"Observation label mismatch: {setting}")

        blocks = []
        for start in range(0, len(frame), 5000):
            batch = frame.iloc[start:start + 5000]
            blocks.append(model.predict_proba(batch[features]))

        original = np.vstack(blocks)
        scaled = temperature_scale(original, temperature)
        actual = frame["canonical_label"].to_numpy(dtype=str)

        results[setting] = {}

        for representation, probabilities in (
            ("original", original),
            ("temperature_scaled", scaled),
        ):
            report = evaluate_probability_view(
                actual, probabilities, classes
            )
            results[setting][representation] = report

            context = {
                "partition": PARTITION,
                "view": setting,
                "representation": representation,
            }
            summary_rows.append({**context, **report["summary"]})

            condition_rows.extend(
                {**context, **row} for row in report["per_condition"]
            )
            reliability_rows.extend(
                {**context, **row} for row in report["reliability_bins"]
            )

        original_report = results[setting]["original"]
        scaled_report = results[setting]["temperature_scaled"]

        if (
            original_report["confusion_matrix"]
            != scaled_report["confusion_matrix"]
            or original_report["per_condition"]
            != scaled_report["per_condition"]
        ):
            raise ValueError("Calibration changed classification results")

        original_losses[setting] = original_report["summary"]["log_loss"]
        scaled_losses[setting] = scaled_report["summary"]["log_loss"]

        archives[setting] = {
            "source_group_ids": frame["source_group_id"].to_numpy(dtype=str),
            "actual": actual,
            "classes": np.asarray(classes, dtype=str),
            "original_probabilities": original,
            "temperature_scaled_probabilities": scaled,
            "reported_unit_count": frame[
                "reported_unit_count"
            ].to_numpy(dtype=np.int64),
        }

        print(
            f"  Log loss: {original_losses[setting]:.6f}"
            f" -> {scaled_losses[setting]:.6f}"
        )
        print(
            f"  Accuracy unchanged:"
            f" {100 * original_report['summary']['accuracy']:.2f}%"
        )

    selection = select_probability_representation(
        original_losses, scaled_losses
    )
    selection.update({
        "candidate_name": "rf_group_weighted",
        "selection_partition": PARTITION,
        "fitted_temperature": temperature,
        "selected_temperature": (
            temperature
            if selection["selected_representation"] == "temperature_scaled"
            else None
        ),
        "temperature_fit_sha256": before["temperature_fit"],
        "model_sha256": before["model"],
        "protocol_sha256": before["protocol"],
        "status": "representation_selected_uncertainty_policy_pending",
    })

    if before != fingerprints():
        raise RuntimeError("An input changed during calibration comparison")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    prediction_folder = OUTPUT / "predictions"
    prediction_folder.mkdir()
    output_hashes = {}

    for setting, arrays in archives.items():
        filename = f"predictions/{setting}.npz"
        path = OUTPUT / filename
        np.savez_compressed(path, **arrays)
        output_hashes[filename] = file_hash(path)

    for filename, rows in (
        ("summary.csv", summary_rows),
        ("per_condition.csv", condition_rows),
        ("reliability_bins.csv", reliability_rows),
    ):
        path = OUTPUT / filename
        pd.DataFrame(rows).to_csv(
            path, index=False, lineterminator="\n"
        )
        output_hashes[filename] = file_hash(path)

    for filename, content in (
        ("per_view_metrics.json", results),
        ("selection.json", selection),
    ):
        path = OUTPUT / filename
        write_json(path, content)
        output_hashes[filename] = file_hash(path)

    write_json(
        OUTPUT / "comparison_manifest.json",
        {
            "issue": 49,
            "status": "policy_development_calibration_comparison_completed",
            "partition": PARTITION,
            "source_groups": expected_count,
            "settings": list(SETTINGS),
            "input_sha256": before,
            "output_sha256": output_hashes,
            "notes": [
                "The fitted temperature was not refitted on development data.",
                "Representation selection used the prespecified mean log loss.",
                "Classification results are unchanged by temperature scaling.",
                "Per-view calibration results may differ.",
                "Uncertainty thresholds have not yet been selected.",
                "Uncertainty-evaluation predictions were not calculated.",
                "The final test split was not inspected.",
                "No deployed model or application was changed.",
            ],
        },
    )

    print("\nCALIBRATION COMPARISON COMPLETED")
    print(
        "Original mean log loss:",
        f"{selection['original_mean_log_loss']:.6f}",
    )
    print(
        "Temperature-scaled mean log loss:",
        f"{selection['temperature_scaled_mean_log_loss']:.6f}",
    )
    print("Selected representation:", selection["selected_representation"])
    print("Input/model fingerprints unchanged.")
    print("Uncertainty policy remains pending.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()