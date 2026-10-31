"""Export recorded calibration results without fitting or predicting."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "data/candidate_experiment_v1/calibration"
REPORTS = ROOT / "reports/candidate_calibration"
OUTPUT = REPORTS / "recorded_results"
EXPORT_MANIFEST = REPORTS / "results_export_manifest.json"

MANIFESTS = (
    "temperature_fit/temperature_fit.json",
    "policy_development_comparison/comparison_manifest.json",
    "uncertainty_policy/search_manifest.json",
    "uncertainty_evaluation/evaluation_manifest.json",
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if OUTPUT.exists() or EXPORT_MANIFEST.exists():
        raise FileExistsError(
            "Calibration export already exists. Do not overwrite it."
        )

    checked = {}
    copies = {}

    def remember(path, expected=None):
        path = path.resolve()
        actual = file_hash(path)

        if expected is not None and actual != expected:
            raise RuntimeError(f"Fingerprint mismatch: {path}")

        if path in checked and checked[path] != actual:
            raise RuntimeError(f"File changed during export: {path}")

        checked[path] = actual

    def queue_copy(path):
        path = path.resolve()
        relative = path.relative_to(CALIBRATION.resolve())
        remember(path)
        copies[relative] = path

    linked_inputs = {
        "model": (
            ROOT
            / "data/candidate_experiment_v1/models"
            / "rf_group_weighted/model.joblib"
        ),
        "candidate_schema": ROOT / "docs/candidate_feature_schema_v1.json",
        "protocol": ROOT / "docs/CANDIDATE_CALIBRATION_PROTOCOL.md",
        "temperature_fit": CALIBRATION / MANIFESTS[0],
        "calibration_selection": (
            CALIBRATION
            / "policy_development_comparison/selection.json"
        ),
        "comparison_manifest": CALIBRATION / MANIFESTS[1],
        "policy": CALIBRATION / "uncertainty_policy/policy.json",
        "search_manifest": CALIBRATION / MANIFESTS[2],
        "calibration_manifest": CALIBRATION / "partition_manifest.json",
        "calibration_partitions": CALIBRATION / "partitions.csv",
        "view_manifest": CALIBRATION / "views/view_manifest.json",
    }

    print("Checking recorded outputs and linked experiment artifacts...")

    for relative in MANIFESTS:
        manifest_path = CALIBRATION / relative
        queue_copy(manifest_path)
        manifest = read_json(manifest_path)

        for name, path in linked_inputs.items():
            expected = manifest["input_sha256"].get(name)
            if expected is not None:
                remember(path, expected)

        for filename, expected in manifest["output_sha256"].items():
            path = (manifest_path.parent / filename).resolve()

            if not path.is_relative_to(CALIBRATION.resolve()):
                raise ValueError("Manifest output is outside calibration data")

            remember(path, expected)

            # Keep large per-observation prediction archives in ignored data.
            if path.suffix.lower() in {".json", ".csv"}:
                queue_copy(path)

    queue_copy(CALIBRATION / "partition_manifest.json")
    queue_copy(CALIBRATION / "views/view_manifest.json")

    policy_path = linked_inputs["policy"]
    decision_path = (
        CALIBRATION / "uncertainty_evaluation/evaluation_decision.json"
    )
    policy = read_json(policy_path)
    decision = read_json(decision_path)

    if decision["policy_sha256"] != checked[policy_path.resolve()]:
        raise RuntimeError("Evaluation references a different policy")

    if decision["thresholds_changed"] or decision["temperature_refitted"]:
        raise RuntimeError("Evaluation did not preserve the frozen policy")

    # Connect the inference probes to the same evaluated policy.
    probes_path = REPORTS / "inference_probes.json"
    probe_csv_path = REPORTS / "inference_probes.csv"
    remember(probes_path)
    remember(probe_csv_path)
    probes = read_json(probes_path)

    for name, path in {
        "policy": policy_path,
        "evaluation_manifest": CALIBRATION / MANIFESTS[3],
        "evaluation_decision": decision_path,
        "predictor": ROOT / "inference/candidate_predictor.py",
        "assessment": ROOT / "inference/candidate_assessment.py",
        "audit_script": ROOT / "scripts/audit_candidate_inference.py",
    }.items():
        remember(path, probes["source_sha256"][name])

    remember(Path(__file__))

    # Read exact bytes before creating any output.
    payloads = {}
    for relative, source in copies.items():
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != checked[source]:
            raise RuntimeError(f"Source changed before copying: {source}")
        payloads[relative] = content

    for path, expected in checked.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input changed during export: {path}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    output_hashes = {}

    for relative, content in sorted(payloads.items()):
        destination = OUTPUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        if file_hash(destination) != expected:
            raise RuntimeError(f"Export verification failed: {destination}")

        output_hashes[destination.relative_to(REPORTS).as_posix()] = expected

    export = {
        "issue": 49,
        "status": "recorded_results_exported",
        "candidate_name": policy["candidate_name"],
        "probability_representation": policy["probability_representation"],
        "temperature": policy["temperature"],
        "policy_mode": policy["mode"],
        "evaluation_groups_per_view": decision["source_groups"],
        "research_targets_met_on_all_required_views": decision[
            "research_targets_met_on_all_required_views"
        ],
        "source_sha256": {
            path.relative_to(ROOT.resolve()).as_posix(): expected
            for path, expected in sorted(checked.items())
        },
        "output_sha256": output_hashes,
        "notes": [
            "Exported JSON and CSV files preserve their original bytes.",
            "Recorded prediction-archive fingerprints were checked.",
            "Prediction archives and model artifacts remain in ignored data.",
            "Historical manifest statuses describe their original stages.",
            "The separate evaluation decision records the completed evaluation.",
            "This export checks integrity; it does not recompute metrics.",
            "No model prediction, calibration fitting or policy search occurred.",
            "Repeated views contain the same groups, not independent patients.",
            "Research targets do not establish clinical performance in Kenya.",
            "Final-test evaluation and deployment remain separate work.",
        ],
    }

    EXPORT_MANIFEST.write_text(
        json.dumps(export, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nCALIBRATION RESULTS EXPORTED")
    print("Copied JSON/CSV files:", len(output_hashes))
    print("Evaluation groups per view:", decision["source_groups"])
    print("Probability representation:", policy["probability_representation"])
    print("PASS: exported files match the recorded source bytes.")
    print("No model trained, predictions calculated or thresholds changed.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()