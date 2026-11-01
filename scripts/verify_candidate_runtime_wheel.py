"""Verify the wheel against the exported runtime source fingerprints."""

import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT / "data/candidate_experiment_v1/portable_runtime_v1"
)
WHEEL_NAME = "respiratory_candidate_runtime-0.1.0-py3-none-any.whl"
DIST_INFO = "respiratory_candidate_runtime-0.1.0.dist-info"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    output = RELEASE / "wheel_verification.json"
    if output.exists():
        raise FileExistsError("Wheel verification already exists.")

    export_bytes = (RELEASE / "export_record.json").read_bytes()
    export = json.loads(export_bytes)

    manifest_bytes = (RELEASE / "bundle/manifest.json").read_bytes()
    require(
        digest(manifest_bytes) == export["manifest_sha256"],
        "Bundle manifest differs from the export record.",
    )
    manifest = json.loads(manifest_bytes)
    require(
        manifest["source_sha256"] == export["source_sha256"],
        "Manifest and export source fingerprints disagree.",
    )

    source_hashes = {
        name: expected
        for name, expected in export["source_sha256"].items()
        if name.startswith("respiratory_candidate_runtime/")
        and name.endswith(".py")
    }
    require(len(source_hashes) == 10, "Expected 10 runtime modules.")

    wheel_path = RELEASE / "wheels" / WHEEL_NAME
    wheel_bytes = wheel_path.read_bytes()

    metadata_files = {
        f"{DIST_INFO}/METADATA",
        f"{DIST_INFO}/WHEEL",
        f"{DIST_INFO}/top_level.txt",
        f"{DIST_INFO}/RECORD",
    }

    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
        names = archive.namelist()

        require(
            len(names) == len(set(names)),
            "Wheel contains duplicate entries.",
        )
        require(
            set(names) == set(source_hashes) | metadata_files,
            "Wheel contains missing or unexpected files.",
        )

        for name, expected in source_hashes.items():
            require(
                digest(archive.read(name)) == expected,
                f"Wheel source differs from the verified export: {name}",
            )
            require(
                digest((ROOT / name).read_bytes()) == expected,
                f"Local runtime source changed after export: {name}",
            )

        metadata = BytesParser().parsebytes(
            archive.read(f"{DIST_INFO}/METADATA")
        )

        require(
            metadata["Name"] == "respiratory-candidate-runtime",
            "Unexpected distribution name.",
        )
        require(
            metadata["Version"] == export["runtime_version"] == "0.1.0",
            "Runtime version mismatch.",
        )

        environment = export["environment"]
        python_requirement = f"=={environment['python']}"
        requirements = {
            f"{name}=={version}"
            for name, version in environment["packages"].items()
        }
        declared_requirements = metadata.get_all("Requires-Dist", [])

        require(
            metadata["Requires-Python"] == python_requirement,
            "Python requirement differs from the verified environment.",
        )
        require(
            len(declared_requirements) == len(requirements)
            and set(declared_requirements) == requirements,
            "Dependency requirements differ from the verified environment.",
        )

        record_name = f"{DIST_INFO}/RECORD"
        rows = list(
            csv.reader(
                io.StringIO(archive.read(record_name).decode("utf-8"))
            )
        )
        require(
            all(len(row) == 3 for row in rows),
            "Invalid wheel RECORD structure.",
        )
        require(
            len(rows) == len(names)
            and {row[0] for row in rows} == set(names),
            "Wheel RECORD inventory mismatch.",
        )

        for name, recorded_hash, recorded_size in rows:
            if name == record_name:
                require(
                    recorded_hash == recorded_size == "",
                    "RECORD must not contain its own digest.",
                )
                continue

            data = archive.read(name)
            expected_hash = (
                "sha256="
                + base64.urlsafe_b64encode(
                    hashlib.sha256(data).digest()
                ).rstrip(b"=").decode("ascii")
            )
            require(
                recorded_hash == expected_hash,
                f"Wheel RECORD digest mismatch: {name}",
            )
            require(
                recorded_size == str(len(data)),
                f"Wheel RECORD size mismatch: {name}",
            )

    configuration_bytes = (ROOT / "pyproject.toml").read_bytes()
    configuration = tomllib.loads(configuration_bytes.decode("utf-8"))
    project = configuration["project"]

    require(
        project["name"] == metadata["Name"]
        and project["version"] == metadata["Version"]
        and project["requires-python"] == python_requirement
        and set(project["dependencies"]) == requirements,
        "Packaging configuration differs from the wheel metadata.",
    )
    require(
        configuration["tool"]["setuptools"]["packages"]
        == ["respiratory_candidate_runtime"],
        "Unexpected package selection.",
    )

    report = {
        "issue": 58,
        "status": "wheel_verified_isolated_installation_pending",
        "wheel_filename": WHEEL_NAME,
        "wheel_sha256": digest(wheel_bytes),
        "wheel_size_bytes": len(wheel_bytes),
        "bundle_manifest_sha256": export["manifest_sha256"],
        "export_record_sha256": digest(export_bytes),
        "pyproject_sha256": digest(configuration_bytes),
        "runtime_version": export["runtime_version"],
        "environment": environment,
        "runtime_modules_verified": len(source_hashes),
        "runtime_source_sha256": source_hashes,
        "wheel_record_verified": True,
        "deployment_approved": False,
    }

    # Refuse to replace a report if another process created it.
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")

    print("WHEEL VERIFICATION PASSED")
    print("PASS: all 10 modules match the exported source fingerprints.")
    print("PASS: wheel inventory and RECORD digests.")
    print("PASS: Python and dependency requirements.")
    print(f"Wheel SHA-256: {report['wheel_sha256']}")
    print(f"Saved to: {output}")
    print("Isolated installation remains pending.")


if __name__ == "__main__":
    main()