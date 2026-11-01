"""Verify installed-runtime parity without importing the ML repository."""

import argparse
import copy
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import zipfile


FORBIDDEN_IMPORTS = {"scripts", "inference", "nlp"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class BlockResearchImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN_IMPORTS:
            raise ImportError(
                f"Research-repository import is forbidden: {fullname}"
            )
        return None


def compare(actual, expected, location):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and actual.keys() == expected.keys(),
            f"Response fields differ: {location}",
        )
        for key in expected:
            compare(actual[key], expected[key], f"{location}.{key}")
        return

    if isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            f"Response list differs: {location}",
        )
        for index, (first, second) in enumerate(zip(actual, expected)):
            compare(first, second, f"{location}[{index}]")
        return

    if type(expected) is float:
        require(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            f"Score differs: {location}",
        )
        return

    require(
        type(actual) is type(expected) and actual == expected,
        f"Response value differs: {location}",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-directory", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    args = parser.parse_args()

    root = Path(args.release_directory).resolve()
    output = root / "isolated_verification.json"

    require(not output.exists(), "Isolated verification already exists.")
    require(sys.flags.isolated == 1, "Run Python with -I.")
    require(sys.prefix != sys.base_prefix, "Use the separate virtual environment.")
    require(Path.cwd().resolve() == root, "Run from the copied release directory.")
    require(
        Path(__file__).resolve().parent == root,
        "Run the copied verification script.",
    )

    guard = BlockResearchImports()
    sys.meta_path.insert(0, guard)

    wheel_name = "respiratory_candidate_runtime-0.1.0-py3-none-any.whl"
    wheel_path = root / wheel_name
    wheel_bytes = wheel_path.read_bytes()

    require(
        digest(wheel_bytes) == args.wheel_sha256,
        "Copied wheel differs from the trusted wheel digest.",
    )

    export_bytes = (root / "export_record.json").read_bytes()
    export = json.loads(export_bytes)
    wheel_report_bytes = (root / "wheel_verification.json").read_bytes()
    wheel_report = json.loads(wheel_report_bytes)

    require(
        wheel_report["export_record_sha256"] == digest(export_bytes),
        "Export record differs from the verified record.",
    )
    require(
        export["manifest_sha256"]
        == wheel_report["bundle_manifest_sha256"]
        == args.manifest_sha256,
        "Bundle identities disagree.",
    )
    require(
        wheel_report["wheel_sha256"] == args.wheel_sha256,
        "Wheel verification refers to another wheel.",
    )

    import respiratory_candidate_runtime as package

    package_directory = Path(package.__file__).resolve().parent
    require(
        package_directory.is_relative_to(Path(sys.prefix).resolve()),
        "Runtime was not imported from the separate environment.",
    )
    require(package.__version__ == "0.1.0", "Unexpected runtime version.")

    source_hashes = wheel_report["runtime_source_sha256"]
    require(len(source_hashes) == 10, "Expected 10 runtime modules.")

    with zipfile.ZipFile(wheel_path) as archive:
        for name, expected in source_hashes.items():
            parts = Path(name).parts
            require(
                len(parts) == 2
                and parts[0] == "respiratory_candidate_runtime"
                and parts[1].endswith(".py"),
                f"Unexpected runtime source entry: {name}",
            )
            installed = package_directory / parts[1]
            require(
                digest(archive.read(name)) == expected
                and digest(installed.read_bytes()) == expected,
                f"Installed source differs from the verified wheel: {name}",
            )

    from respiratory_candidate_runtime.errors import CandidateInputError
    from respiratory_candidate_runtime.integrity import (
        decode_json_bytes,
        read_verified_bundle,
    )
    from respiratory_candidate_runtime.loader import load_runtime

    bundle = root / "bundle"
    manifest, payloads = read_verified_bundle(
        bundle,
        args.manifest_sha256,
    )
    require(
        manifest["source_sha256"] == export["source_sha256"],
        "Bundle and export source fingerprints disagree.",
    )

    cases = decode_json_bytes(
        payloads["reference_cases.json"],
        "reference_cases.json",
    )["cases"]
    require(
        len(cases) == 13
        and len({case["case_id"] for case in cases}) == 13,
        "Reference cases are missing or duplicated.",
    )

    print("Loading the separately installed runtime...")
    runtime = load_runtime(
        bundle,
        expected_manifest_sha256=args.manifest_sha256,
    )

    results = []
    for case in cases:
        for diagnostics in (False, True):
            kwargs = copy.deepcopy(case["supplied_context"])
            kwargs["include_diagnostics"] = diagnostics
            features = copy.deepcopy(case["encoded_features"])

            if case["audit_status"] == "rejected_invalid_input":
                try:
                    runtime.predict(features, **kwargs)
                except CandidateInputError as error:
                    require(
                        str(error) == case["validation_error"],
                        f"Validation error changed: {case['case_id']}",
                    )
                else:
                    raise AssertionError(
                        f"Invalid input accepted: {case['case_id']}"
                    )
            else:
                expected = copy.deepcopy(case["public_response"])
                if diagnostics and case["research_diagnostics"] is not None:
                    expected["diagnostics"] = copy.deepcopy(
                        case["research_diagnostics"]
                    )
                actual = runtime.predict(features, **kwargs)
                compare(actual, expected, case["case_id"])

        results.append({
            "case_id": case["case_id"],
            "expected_status": case["audit_status"],
            "public_and_diagnostic_calls_verified": True,
        })
        print(f"PASS: {case['case_id']}")

    require(
        not any(
            name.split(".")[0] in FORBIDDEN_IMPORTS
            for name in sys.modules
        ),
        "A research-repository module was imported.",
    )

    # Recheck the copied artifacts after inference.
    read_verified_bundle(bundle, args.manifest_sha256)
    require(
        digest(wheel_path.read_bytes()) == args.wheel_sha256,
        "Wheel changed during verification.",
    )

    installed_versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }

    report = {
        "issue": 58,
        "status": "isolated_installed_runtime_verified",
        "manifest_sha256": args.manifest_sha256,
        "wheel_sha256": args.wheel_sha256,
        "export_record_sha256": digest(export_bytes),
        "wheel_verification_sha256": digest(wheel_report_bytes),
        "verification_script_sha256": digest(
            Path(__file__).read_bytes()
        ),
        "runtime_version": package.__version__,
        "environment": runtime.provenance["environment"],
        "installed_distributions": dict(sorted(installed_versions.items())),
        "python_isolated_mode": True,
        "runtime_loaded_from_separate_environment": True,
        "research_imports_blocked": sorted(FORBIDDEN_IMPORTS),
        "runtime_modules_verified": len(source_hashes),
        "cases_verified": len(cases),
        "calls_verified": len(cases) * 2,
        "absolute_float_tolerance": 1e-12,
        "relative_float_tolerance": 0,
        "decisions_compared_exactly": True,
        "cases": results,
        "deployment_approved": False,
    }

    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print("\nISOLATED RUNTIME VERIFICATION PASSED")
    print("PASS: 10 installed modules match the verified wheel.")
    print("PASS: 13 cases, 26 public/diagnostic calls.")
    print("PASS: no ML research modules imported.")
    print("PASS: bundle and wheel fingerprints unchanged.")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()