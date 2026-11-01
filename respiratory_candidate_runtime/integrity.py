"""Verify bundle files before model deserialization.

This module does not import joblib or load model objects.
"""

import hashlib
import json
from pathlib import Path
import re

from . import __version__
from .errors import BundleValidationError


BUNDLE_FORMAT_VERSION = 1

REQUIRED_PAYLOADS = {
    "model.joblib",
    "feature_schema.json",
    "policy.json",
    "classes.json",
    "reference_cases.json",
}

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value

    return result


def _reject_constant(value):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def decode_json_bytes(data, description):
    """Decode strict JSON, rejecting duplicate keys and NaN/Infinity."""

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError) as error:
        raise BundleValidationError(
            f"Invalid JSON in {description}."
        ) from error


def _validate_digest(value, description):
    if (
        not isinstance(value, str)
        or SHA256_PATTERN.fullmatch(value) is None
    ):
        raise BundleValidationError(
            f"{description} must be a lowercase SHA-256 digest."
        )


def _safe_payload_path(root, name):
    if not isinstance(name, str) or not name:
        raise BundleValidationError("Invalid bundle payload name.")

    parts = name.split("/")

    if (
        "\\" in name
        or ":" in name
        or any(
            part in ("", ".", "..")
            or part.endswith((" ", "."))
            for part in parts
        )
        or name == "manifest.json"
    ):
        raise BundleValidationError(
            f"Unsafe bundle payload path: {name}"
        )

    path = root

    for part in parts:
        path = path / part

        if path.is_symlink():
            raise BundleValidationError(
                f"Linked bundle paths are not supported: {name}"
            )

    resolved = path.resolve()

    if not resolved.is_relative_to(root):
        raise BundleValidationError(
            f"Bundle payload escapes its directory: {name}"
        )

    return resolved


def read_verified_bundle(bundle_directory, expected_manifest_sha256):
    """Return the manifest and verified payload bytes.

    The expected digest must come from a trusted release record or
    deployment configuration, independently of the supplied bundle.
    """

    _validate_digest(
        expected_manifest_sha256,
        "Expected manifest fingerprint",
    )

    try:
        root = Path(bundle_directory).resolve()

        if not root.is_dir():
            raise BundleValidationError(
                "Bundle directory does not exist."
            )

        manifest_path = root / "manifest.json"

        if manifest_path.is_symlink():
            raise BundleValidationError(
                "The manifest must not be a symbolic link."
            )

        # Also reject links/junctions resolving outside the bundle.
        if not manifest_path.resolve().is_relative_to(root):
            raise BundleValidationError(
                "The manifest escapes the bundle directory."
            )

        manifest_bytes = manifest_path.read_bytes()

        if sha256_bytes(manifest_bytes) != expected_manifest_sha256:
            raise BundleValidationError(
                "Bundle manifest fingerprint mismatch."
            )

        manifest = decode_json_bytes(
            manifest_bytes,
            "manifest.json",
        )

        if not isinstance(manifest, dict):
            raise BundleValidationError(
                "Bundle manifest must be a JSON object."
            )

        if (
            type(manifest.get("bundle_format_version")) is not int
            or manifest["bundle_format_version"] != BUNDLE_FORMAT_VERSION
        ):
            raise BundleValidationError(
                "Unsupported bundle format version."
            )

        bundle_id = manifest.get("bundle_id")

        if not isinstance(bundle_id, str) or not bundle_id.strip():
            raise BundleValidationError("Missing bundle identity.")

        if manifest.get("runtime_version") != __version__:
            raise BundleValidationError(
                "Bundle requires a different runtime version."
            )

        files = manifest.get("files")

        if not isinstance(files, dict):
            raise BundleValidationError(
                "Manifest files must be a JSON object."
            )

        if not REQUIRED_PAYLOADS.issubset(files):
            raise BundleValidationError(
                "Manifest is missing required payload entries."
            )

        # Reject ambiguous names on case-insensitive filesystems.
        names = list(files)

        if len({name.casefold() for name in names}) != len(names):
            raise BundleValidationError(
                "Bundle contains case-conflicting payload names."
            )

        payloads = {}

        for name, entry in files.items():
            path = _safe_payload_path(root, name)

            if not isinstance(entry, dict):
                raise BundleValidationError(
                    f"Invalid payload entry: {name}"
                )

            digest = entry.get("sha256")
            size = entry.get("size_bytes")

            _validate_digest(digest, f"Fingerprint for {name}")

            if type(size) is not int or size < 0:
                raise BundleValidationError(
                    f"Invalid payload byte size: {name}"
                )

            if not path.is_file():
                raise BundleValidationError(
                    f"Required bundle file is missing: {name}"
                )

            if path.stat().st_size != size:
                raise BundleValidationError(
                    f"Bundle file size mismatch: {name}"
                )

            data = path.read_bytes()

            if len(data) != size or sha256_bytes(data) != digest:
                raise BundleValidationError(
                    f"Bundle file fingerprint mismatch: {name}"
                )

            payloads[name] = data

        return manifest, payloads

    except BundleValidationError:
        raise
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        raise BundleValidationError(
            "Unable to read the candidate bundle."
        ) from error