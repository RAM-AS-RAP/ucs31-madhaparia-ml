"""Export verified final-evaluation reports without rerunning predictions."""

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_source_records import file_hash


BASE = ROOT / "data/candidate_experiment_v1/final_test"
REPORTS = ROOT / "reports/candidate_final_evaluation"
OUTPUT = REPORTS / "recorded_results"
EXPORT_MANIFEST = REPORTS / "results_export_manifest.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if OUTPUT.exists() or EXPORT_MANIFEST.exists():
        raise FileExistsError("Final-results export already exists.")

    checked = {}

    def remember(path, expected=None):
        path = path.resolve()
        actual = file_hash(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Fingerprint mismatch: {path}")
        if path in checked and checked[path] != actual:
            raise RuntimeError(f"Input changed during export: {path}")
        checked[path] = actual

    print("Checking completed metric verification...", flush=True)

    verification_path = REPORTS / "metrics_verification.json"
    remember(verification_path)
    verification = read_json(verification_path)

    if verification["status"] != "final_metrics_verified_against_saved_predictions":
        raise ValueError("Final metric verification is not complete")

    for relative, expected in verification["input_sha256"].items():
        remember(ROOT / relative, expected)

    evaluation_manifest_path = BASE / "evaluation/evaluation_manifest.json"
    remember(evaluation_manifest_path)
    evaluation = read_json(evaluation_manifest_path)

    if evaluation["status"] != "frozen_official_test_evaluation_completed":
        raise ValueError("Final evaluation is not complete")

    if verification["cohort_counts"] != evaluation["cohort_counts"]:
        raise ValueError("Verification and evaluation cohort counts differ")

    copies = {
        "evaluation/evaluation_manifest.json": evaluation_manifest_path,
        "views/view_manifest.json": BASE / "views/view_manifest.json",
    }

    for filename, expected in evaluation["output_sha256"].items():
        source = (BASE / "evaluation" / filename).resolve()
        if not source.is_relative_to((BASE / "evaluation").resolve()):
            raise ValueError("Unexpected evaluation output path")

        remember(source, expected)

        if source.suffix.lower() in {".json", ".csv"}:
            copies[f"evaluation/{filename}"] = source

    payloads = {}
    for relative, source in copies.items():
        remember(source)
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != checked[source.resolve()]:
            raise RuntimeError(f"Source changed before copying: {source}")
        payloads[relative] = content

    remember(Path(__file__))

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
            raise RuntimeError(f"Copied bytes differ: {destination}")

        output_hashes[destination.relative_to(REPORTS).as_posix()] = expected

    manifest = {
        "issue": 50,
        "status": "verified_final_results_exported",
        "cohort_counts": evaluation["cohort_counts"],
        "research_target_assessment": verification["research_target_assessment"],
        "input_sha256": {
            path.relative_to(ROOT.resolve()).as_posix(): expected
            for path, expected in sorted(checked.items())
        },
        "output_sha256": output_hashes,
        "notes": [
            "Exported files preserve their original bytes.",
            "Existing verification and evaluation fingerprints were checked.",
            "Large observation CSVs and prediction archives remain in ignored data.",
            "The existing source audit and metric verification remain alongside this export.",
            "Historical run_started status is superseded by the completed evaluation manifest.",
            "No models were loaded and no predictions or metrics were recalculated.",
            "No calibration parameters, thresholds or deployed application files changed.",
        ],
    }

    EXPORT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL RESULTS EXPORTED")
    print("Copied JSON/CSV files:", len(output_hashes))
    print("Cohort counts:", evaluation["cohort_counts"])
    print("PASS: exported bytes match verified source files.")
    print("No predictions calculated.")
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()