"""Test loading order, metadata checks and verified-byte consumption."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from respiratory_candidate_runtime import __version__
from respiratory_candidate_runtime import loader
from respiratory_candidate_runtime.errors import BundleValidationError


ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")


def save_manifest(root, manifest):
    data = json_bytes(manifest)
    (root / "manifest.json").write_bytes(data)
    return digest(data)


def replace_json_payload(root, manifest, name, value):
    data = json_bytes(value)
    (root / name).write_bytes(data)

    manifest["files"][name] = {
        "sha256": digest(data),
        "size_bytes": len(data),
    }


class StubModel:
    def __init__(self, schema, classes):
        self.n_features_in_ = 70
        self.feature_names_in_ = np.asarray(schema["feature_names"])
        self.classes_ = np.asarray(classes)
        self.calls = []

    def predict_proba(self, frame):
        self.calls.append(frame.copy())

        values = np.full((1, 21), 0.05 / 20)
        values[0, 0] = 0.95
        return values


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()

    schema_bytes = (
        ROOT / "docs/candidate_feature_schema_v1.json"
    ).read_bytes()
    schema = json.loads(schema_bytes)

    policy = json.loads(
        (
            ROOT
            / "reports/candidate_calibration/recorded_results"
            / "uncertainty_policy/policy.json"
        ).read_bytes()
    )
    classes = list(policy["classes"])

    # Synthetic artifact references are used only in this test fixture.
    model_bytes = b"dummy model bytes"
    policy["model_sha256"] = digest(model_bytes)
    policy["candidate_schema_sha256"] = digest(schema_bytes)

    payloads = {
        "model.joblib": model_bytes,
        "feature_schema.json": schema_bytes,
        "policy.json": json_bytes(policy),
        "classes.json": json_bytes(classes),
        "reference_cases.json": b"[]",
    }

    files = {}

    for name, data in payloads.items():
        (root / name).write_bytes(data)
        files[name] = {
            "sha256": digest(data),
            "size_bytes": len(data),
        }

    manifest = {
        "bundle_format_version": 1,
        "bundle_id": "loader-test-bundle",
        "runtime_version": __version__,
        "model_version": "candidate_v1_research",
        "schema_id": schema["schema_id"],
        "policy_id": policy["policy_id"],
        "environment": loader.current_environment(),
        "files": files,
    }

    expected = save_manifest(root, manifest)
    model = StubModel(schema, classes)
    deserializer = Mock(return_value=model)

    monkeypatch.setattr(loader.joblib, "load", deserializer)

    return {
        "root": root,
        "manifest": manifest,
        "expected": expected,
        "model": model,
        "deserializer": deserializer,
        "model_bytes": model_bytes,
        "schema": schema,
        "policy": policy,
        "classes": classes,
    }


def load(data):
    return loader.load_runtime(
        data["root"],
        expected_manifest_sha256=data["expected"],
    )


def test_valid_load_uses_verified_bytes_without_prediction(bundle):
    runtime = load(bundle)

    bundle["deserializer"].assert_called_once()
    source = bundle["deserializer"].call_args.args[0]

    assert source.getvalue() == bundle["model_bytes"]
    assert bundle["model"].calls == []
    assert runtime.classes == tuple(bundle["classes"])
    assert runtime.provenance["manifest_sha256"] == bundle["expected"]
    assert runtime.provenance["model_sha256"] == digest(
        bundle["model_bytes"]
    )


def test_public_metadata_cannot_mutate_loaded_resources(bundle):
    runtime = load(bundle)

    schema_copy = runtime.schema
    schema_copy["feature_names"].clear()

    provenance_copy = runtime.provenance
    provenance_copy["environment"]["packages"].clear()

    assert len(runtime.schema["feature_names"]) == 70
    assert runtime.provenance["environment"]["packages"]


def test_loaded_runtime_can_predict_and_withhold(bundle):
    runtime = load(bundle)

    features = {
        name: 0.0
        for name in runtime.schema["feature_names"]
    }
    features.update({
        "AGE": 22,
        "SEX_M": 0,
        "E_89__present": 1,
        "E_91__present": 1,
    })

    accepted = runtime.predict(features)

    assert accepted["status"] == "prediction_available"
    assert accepted["predicted_condition"] == bundle["classes"][0]
    assert len(bundle["model"].calls) == 1

    features["E_91__present"] = 0
    withheld = runtime.predict(features)

    assert withheld["status"] == "insufficient_information"
    assert withheld["predicted_condition"] is None
    assert withheld["prediction_performed"] is False
    assert len(bundle["model"].calls) == 1


@pytest.mark.parametrize("problem", ["missing", "modified", "manifest"])
def test_integrity_failure_prevents_deserialization(bundle, problem):
    if problem == "missing":
        (bundle["root"] / "model.joblib").unlink()
    elif problem == "modified":
        (bundle["root"] / "model.joblib").write_bytes(b"changed")
    else:
        bundle["expected"] = "0" * 64

    with pytest.raises(BundleValidationError):
        load(bundle)

    bundle["deserializer"].assert_not_called()


@pytest.mark.parametrize(
    "problem",
    ["environment", "identity", "schema", "policy", "class_order"],
)
def test_metadata_failure_prevents_deserialization(bundle, problem):
    root = bundle["root"]
    manifest = bundle["manifest"]

    if problem == "environment":
        manifest["environment"]["packages"]["numpy"] = "0.0.0"
    elif problem == "identity":
        manifest["model_version"] = "different"
    elif problem == "schema":
        schema = bundle["schema"]
        schema["feature_count"] = 66
        replace_json_payload(
            root,
            manifest,
            "feature_schema.json",
            schema,
        )
    elif problem == "policy":
        policy = bundle["policy"]
        policy["min_reporting_units"] = 1
        replace_json_payload(root, manifest, "policy.json", policy)
    else:
        replace_json_payload(
            root,
            manifest,
            "classes.json",
            list(reversed(bundle["classes"])),
        )

    bundle["expected"] = save_manifest(root, manifest)

    with pytest.raises(BundleValidationError):
        load(bundle)

    bundle["deserializer"].assert_not_called()


@pytest.mark.parametrize(
    "problem",
    ["feature_count", "feature_order", "class_order", "no_predict"],
)
def test_incompatible_loaded_model_is_rejected(bundle, problem):
    model = bundle["model"]

    if problem == "feature_count":
        model.n_features_in_ = 66
    elif problem == "feature_order":
        model.feature_names_in_ = model.feature_names_in_[::-1]
    elif problem == "class_order":
        model.classes_ = model.classes_[::-1]
    else:
        model.predict_proba = None

    with pytest.raises(BundleValidationError):
        load(bundle)

    bundle["deserializer"].assert_called_once()


def test_deserialization_failure_is_a_bundle_error(bundle):
    failure = RuntimeError("Controlled loading failure")
    bundle["deserializer"].side_effect = failure

    with pytest.raises(BundleValidationError) as captured:
        load(bundle)

    assert captured.value.__cause__ is failure


def test_loader_does_not_reopen_model_path(bundle):
    def replace_file_during_deserialization(source):
        # The loader must use the already-verified in-memory bytes.
        (bundle["root"] / "model.joblib").write_bytes(b"changed later")
        assert source.getvalue() == bundle["model_bytes"]
        return bundle["model"]

    bundle["deserializer"].side_effect = (
        replace_file_during_deserialization
    )

    runtime = load(bundle)

    assert runtime.provenance["model_sha256"] == digest(
        bundle["model_bytes"]
    )

    # A subsequent load must detect the changed disk file.
    with pytest.raises(BundleValidationError):
        load(bundle)