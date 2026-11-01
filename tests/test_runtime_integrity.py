"""Integrity checks using dummy bytes, without model loading."""

import hashlib
import json

import pytest

from respiratory_candidate_runtime import __version__
from respiratory_candidate_runtime.errors import BundleValidationError
from respiratory_candidate_runtime.integrity import (
    read_verified_bundle,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_manifest(root, manifest):
    data = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")

    (root / "manifest.json").write_bytes(data)

    return digest(data)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()

    payloads = {
        "model.joblib": b"dummy model bytes; never deserialized",
        "feature_schema.json": b"{}",
        "policy.json": b"{}",
        "classes.json": b"[]",
        "reference_cases.json": b"[]",
        "provenance/evaluation.json": b'{"status":"test-only"}',
    }

    files = {}

    for name, data in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        files[name] = {
            "sha256": digest(data),
            "size_bytes": len(data),
        }

    manifest = {
        "bundle_format_version": 1,
        "bundle_id": "integrity-test-bundle",
        "runtime_version": __version__,
        "files": files,
    }

    expected = write_manifest(root, manifest)

    return root, manifest, payloads, expected


def test_valid_bundle_returns_exact_verified_bytes(bundle):
    root, manifest, payloads, expected = bundle

    actual_manifest, actual_payloads = read_verified_bundle(
        root,
        expected,
    )

    assert actual_manifest == manifest
    assert actual_payloads == payloads


def test_loading_does_not_depend_on_working_directory(
    bundle,
    tmp_path,
    monkeypatch,
):
    root, _, payloads, expected = bundle
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    _, actual = read_verified_bundle(root, expected)

    assert actual == payloads


def test_changed_manifest_fails_trusted_digest_check(bundle):
    root, manifest, _, expected = bundle
    manifest["bundle_id"] = "changed"
    write_manifest(root, manifest)

    with pytest.raises(BundleValidationError, match="manifest fingerprint"):
        read_verified_bundle(root, expected)


@pytest.mark.parametrize(
    "expected",
    [None, "", "not-a-digest", "A" * 64, "0" * 64],
)
def test_invalid_or_wrong_expected_digest_is_rejected(bundle, expected):
    root, _, _, _ = bundle

    with pytest.raises(BundleValidationError):
        read_verified_bundle(root, expected)


@pytest.mark.parametrize(
    "name",
    ["model.joblib", "feature_schema.json", "policy.json"],
)
def test_missing_payload_is_rejected(bundle, name):
    root, _, _, expected = bundle
    (root / name).unlink()

    with pytest.raises(BundleValidationError, match="missing"):
        read_verified_bundle(root, expected)


def test_same_length_model_tampering_is_rejected(bundle):
    root, _, payloads, expected = bundle
    original = payloads["model.joblib"]

    (root / "model.joblib").write_bytes(
        b"X" + original[1:]
    )

    with pytest.raises(BundleValidationError, match="fingerprint mismatch"):
        read_verified_bundle(root, expected)


def test_changed_file_size_is_rejected(bundle):
    root, _, payloads, expected = bundle

    (root / "model.joblib").write_bytes(
        payloads["model.joblib"] + b"extra"
    )

    with pytest.raises(BundleValidationError, match="size mismatch"):
        read_verified_bundle(root, expected)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.joblib",
        "/absolute.joblib",
        "C:/outside.joblib",
        "provenance\\outside.json",
        "provenance/../outside.json",
        "provenance//outside.json",
        "policy.json.",
        "manifest.json",
    ],
)
def test_unsafe_manifest_paths_are_rejected(bundle, name):
    root, manifest, _, _ = bundle
    manifest["files"][name] = {
        "sha256": digest(b"x"),
        "size_bytes": 1,
    }
    expected = write_manifest(root, manifest)

    with pytest.raises(BundleValidationError, match="Unsafe"):
        read_verified_bundle(root, expected)


@pytest.mark.parametrize(
    "field, value",
    [
        ("bundle_format_version", 2),
        ("bundle_format_version", True),
        ("runtime_version", "999.0.0"),
        ("bundle_id", ""),
        ("files", []),
    ],
)
def test_incompatible_manifest_metadata_is_rejected(bundle, field, value):
    root, manifest, _, _ = bundle
    manifest[field] = value
    expected = write_manifest(root, manifest)

    with pytest.raises(BundleValidationError):
        read_verified_bundle(root, expected)


def test_required_entry_cannot_be_omitted(bundle):
    root, manifest, _, _ = bundle
    del manifest["files"]["model.joblib"]
    expected = write_manifest(root, manifest)

    with pytest.raises(BundleValidationError, match="required payload"):
        read_verified_bundle(root, expected)


def test_case_conflicting_names_are_rejected(bundle):
    root, manifest, _, _ = bundle
    manifest["files"]["MODEL.joblib"] = manifest["files"]["model.joblib"]
    expected = write_manifest(root, manifest)

    with pytest.raises(BundleValidationError, match="case-conflicting"):
        read_verified_bundle(root, expected)


def test_duplicate_json_keys_are_rejected(bundle):
    root, _, _, _ = bundle
    data = b'{"bundle_format_version":1,"bundle_format_version":2}'
    (root / "manifest.json").write_bytes(data)

    with pytest.raises(BundleValidationError, match="Invalid JSON"):
        read_verified_bundle(root, digest(data))