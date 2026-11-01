"""Check export parity rules and failure handling."""

import copy
import json
from unittest.mock import Mock

import pytest

from scripts import export_candidate_runtime_bundle as exporter


@pytest.mark.parametrize(
    "actual, expected",
    [
        (0.7801629519255994, 0.7801629519255993),
        ({"score": None}, {"score": None}),
        ({"reasons": ["too_few"]}, {"reasons": ["too_few"]}),
    ],
)
def test_matching_values_pass(actual, expected):
    exporter.compare_values(actual, expected)


@pytest.mark.parametrize(
    "actual, expected",
    [
        (0.7802, 0.7801),
        (float("nan"), 0.78),
        (float("inf"), 0.78),
        (True, 1),
        (1, True),
        ("prediction_available", "insufficient_information"),
        ({"score": 0.78}, {"score": None}),
        ({"score": None, "extra": True}, {"score": None}),
        (["first", "second"], ["second", "first"]),
    ],
)
def test_changed_values_fail(actual, expected):
    with pytest.raises(ValueError):
        exporter.compare_values(actual, expected)


def example_case():
    return {
        "case_id": "example",
        "encoded_features": {"example_feature": 1},
        "supplied_context": {
            "safety_flags": [],
            "unsupported_symptoms": [],
        },
        "audit_status": "prediction_available",
        "public_response": {
            "status": "prediction_available",
            "model_score": 0.8,
        },
        "research_diagnostics": {
            "research_only": True,
            "score_margin": 0.5,
        },
        "validation_error": None,
    }


def test_public_and_diagnostic_calls_are_both_checked():
    case = example_case()
    calls = []

    def predict(features, **kwargs):
        calls.append((copy.deepcopy(features), copy.deepcopy(kwargs)))
        result = copy.deepcopy(case["public_response"])
        if kwargs["include_diagnostics"]:
            result["diagnostics"] = copy.deepcopy(
                case["research_diagnostics"]
            )
        return result

    result = exporter.verify_cases(predict, [case], ValueError)

    assert result["cases_verified"] == 1
    assert result["calls_verified"] == 2
    assert [item[1]["include_diagnostics"] for item in calls] == [
        False,
        True,
    ]
    assert all(item[0] == case["encoded_features"] for item in calls)


def test_diagnostics_cannot_change_public_decision():
    case = example_case()

    def predict(features, **kwargs):
        result = copy.deepcopy(case["public_response"])
        if kwargs["include_diagnostics"]:
            result["status"] = "insufficient_information"
            result["diagnostics"] = copy.deepcopy(
                case["research_diagnostics"]
            )
        return result

    with pytest.raises(ValueError, match="Exact parity failed"):
        exporter.verify_cases(predict, [case], ValueError)


def test_invalid_input_must_raise_expected_error():
    case = example_case()
    case["audit_status"] = "rejected_invalid_input"
    case["validation_error"] = "Conflicting symptom states"

    def reject(features, **kwargs):
        raise ValueError("Conflicting symptom states")

    result = exporter.verify_cases(reject, [case], ValueError)
    assert result["calls_verified"] == 2

    with pytest.raises(AssertionError, match="Invalid input was accepted"):
        exporter.verify_cases(lambda *args, **kwargs: {}, [case], ValueError)


def test_source_change_is_detected(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b"original")
    snapshot = {"source.json": exporter.sha256_bytes(b"original")}

    exporter.check_sources_unchanged(snapshot, root=tmp_path)

    source.write_bytes(b"changed")

    with pytest.raises(ValueError, match="Source changed"):
        exporter.check_sources_unchanged(snapshot, root=tmp_path)


def test_existing_export_is_preserved(tmp_path, monkeypatch):
    destination = tmp_path / "release"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_bytes(b"existing release")
    monkeypatch.setattr(exporter, "DESTINATION", destination)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        exporter.export_bundle({}, {}, {})

    assert marker.read_bytes() == b"existing release"
    assert list(destination.iterdir()) == [marker]


def test_failed_loading_leaves_no_completed_export(tmp_path, monkeypatch):
    from inference import candidate_predictor as reference

    destination = tmp_path / "release"
    monkeypatch.setattr(exporter, "DESTINATION", destination)
    monkeypatch.setattr(reference, "load_candidate_resources", Mock())

    # This test isolates export cleanup; real parity is checked separately.
    monkeypatch.setattr(exporter, "verify_cases", lambda *args: {})
    monkeypatch.setattr(
        exporter,
        "load_runtime",
        Mock(side_effect=RuntimeError("Simulated loader failure")),
    )

    payloads = {
        "reference_cases.json": json.dumps({"cases": []}).encode(),
        "policy.json": json.dumps({"policy_id": "test_policy"}).encode(),
        "feature_schema.json": json.dumps({"schema_id": "test_schema"}).encode(),
    }

    with pytest.raises(RuntimeError, match="Simulated loader failure"):
        exporter.export_bundle(payloads, {}, {})

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []