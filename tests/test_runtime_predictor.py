"""Prediction-flow parity and error-boundary tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from inference import candidate_predictor as reference
from respiratory_candidate_runtime.errors import (
    CandidateInferenceError,
    CandidateInputError,
)
from respiratory_candidate_runtime.predictor import (
    _predict_with_resources,
)


ROOT = Path(__file__).resolve().parents[1]


class StubModel:
    classes_ = np.asarray(["A", "B"])

    def __init__(self):
        self.calls = []
        self.probabilities = np.asarray([[0.95, 0.05]])
        self.failure = None

    def predict_proba(self, frame):
        self.calls.append(frame.copy())

        if self.failure is not None:
            raise self.failure

        return self.probabilities.copy()


@pytest.fixture
def example(monkeypatch):
    schema = json.loads(
        (ROOT / "docs/candidate_feature_schema_v1.json").read_text(
            encoding="utf-8"
        )
    )

    policy = {
        "policy_id": "candidate_v1_research_uncertainty_v1",
        "classes": ["A", "B"],
        "mode": "thresholds",
        "min_reporting_units": 2,
        "min_score": 0.78,
        "min_margin": 0.20,
        "probability_representation": "temperature_scaled",
        "temperature": 0.6408387217626598,
    }

    features = {
        name: 0.0
        for name in schema["feature_names"]
    }
    features.update({
        "AGE": 22,
        "SEX_M": 0,
        "E_89__present": 1,
        "E_91__present": 1,
    })

    model = StubModel()

    # The reference uses the same controlled resources without
    # loading or changing any saved model artifact.
    monkeypatch.setattr(
        reference,
        "load_candidate_resources",
        lambda: (model, schema, policy),
    )

    return model, schema, policy, features


def assert_response_equal(actual, expected):
    assert type(actual) is type(expected)

    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_response_equal(actual[key], expected[key])

    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_response_equal(first, second)

    elif isinstance(expected, float):
        assert np.isfinite(actual)
        assert actual == pytest.approx(
            expected,
            rel=0,
            abs=1e-12,
        )

    else:
        assert actual == expected


@pytest.mark.parametrize(
    "case, expected_status, expected_scoring",
    [
        ("accepted", "prediction_available", True),
        ("low_score", "insufficient_information", True),
        ("one_unit", "insufficient_information", False),
        ("zero_units", "insufficient_information", False),
        ("region_only", "insufficient_information", False),
        ("unsupported_only", "unsupported_input", False),
        ("mixed_unsupported", "unsupported_input", False),
        ("safety_only", "safety_review_required", False),
        ("supplied_safety", "safety_review_required", False),
        ("negative", "unsupported_input", False),
    ],
)
@pytest.mark.parametrize("diagnostics", [False, True])
def test_flow_matches_reference(
    example,
    case,
    expected_status,
    expected_scoring,
    diagnostics,
):
    model, schema, policy, features = example
    context = {}

    if case == "low_score":
        model.probabilities = np.asarray([[0.55, 0.45]])

    if case == "one_unit":
        features["E_91__present"] = 0

    if case in ("zero_units", "region_only", "unsupported_only"):
        features["E_89__present"] = 0
        features["E_91__present"] = 0

    if case == "region_only":
        features["E_53__present"] = 1
        features["head_region_pain__present"] = 1

    if case in ("unsupported_only", "mixed_unsupported"):
        context["unsupported_symptoms"] = ["sneezing"]

    if case == "safety_only":
        context["unsupported_symptoms"] = ["blue_lips"]

    if case == "supplied_safety":
        context["safety_flags"] = ["existing_safety_flag"]

    if case == "negative":
        features["E_201__absent"] = 1

    before = copy.deepcopy(features)

    actual = _predict_with_resources(
        features,
        model=model,
        schema=schema,
        policy=policy,
        include_diagnostics=diagnostics,
        **context,
    )

    assert actual["status"] == expected_status
    assert actual["prediction_performed"] is expected_scoring
    assert len(model.calls) == int(expected_scoring)

    if expected_scoring:
        frame = model.calls[0]
        assert list(frame.columns) == schema["feature_names"]
        assert frame.iloc[0]["AGE"] == 22
        assert frame.iloc[0]["SEX_M"] == 0
    else:
        assert "diagnostics" not in actual

    model.calls.clear()

    expected = reference.predict_candidate(
        features,
        include_diagnostics=diagnostics,
        **context,
    )

    assert_response_equal(actual, expected)
    assert len(model.calls) == int(expected_scoring)
    assert features == before

    if expected_status != "prediction_available":
        assert actual["predicted_condition"] is None
        assert actual["model_score"] is None


def test_safety_takes_priority_and_preserves_all_reasons(example):
    model, schema, policy, features = example
    features["E_91__present"] = 0
    features["E_201__absent"] = 1

    result = _predict_with_resources(
        features,
        model=model,
        schema=schema,
        policy=policy,
        unsupported_symptoms=["sneezing", "blue_lips"],
    )

    assert result["status"] == "safety_review_required"
    assert result["safety_flags"] == ["blue_lips"]
    assert result["unsupported_symptoms"] == ["blue_lips", "sneezing"]
    assert result["uncertainty_reasons"] == [
        "safety_flags_present",
        "unsupported_symptoms_present",
        "explicit_negative_evidence_not_evaluated",
        "too_few_supported_reporting_units",
    ]
    assert model.calls == []


@pytest.mark.parametrize(
    "problem",
    ["contradiction", "missing_parent", "context_string", "diagnostics"],
)
def test_invalid_input_raises_input_error_before_scoring(example, problem):
    model, schema, policy, features = example
    context = {}

    if problem == "contradiction":
        features["E_89__absent"] = 1
    elif problem == "missing_parent":
        features["head_region_pain__present"] = 1
    elif problem == "context_string":
        context["safety_flags"] = "blue_lips"
    else:
        context["include_diagnostics"] = "false"

    with pytest.raises(CandidateInputError):
        _predict_with_resources(
            features,
            model=model,
            schema=schema,
            policy=policy,
            **context,
        )

    assert model.calls == []


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.9, 0.1],
        [[1.0]],
        [[np.nan, 0.1]],
        [[0.2, 0.2]],
        [[-0.1, 1.1]],
    ],
)
def test_invalid_model_output_is_an_inference_failure(
    example,
    probabilities,
):
    model, schema, policy, features = example
    model.probabilities = np.asarray(probabilities)

    with pytest.raises(CandidateInferenceError) as captured:
        _predict_with_resources(
            features,
            model=model,
            schema=schema,
            policy=policy,
        )

    assert not isinstance(captured.value, CandidateInputError)
    assert captured.value.__cause__ is not None
    assert len(model.calls) == 1


def test_model_exception_is_not_converted_to_withholding(example):
    model, schema, policy, features = example
    model.failure = RuntimeError("Controlled model failure")

    with pytest.raises(CandidateInferenceError) as captured:
        _predict_with_resources(
            features,
            model=model,
            schema=schema,
            policy=policy,
        )

    assert captured.value.__cause__ is model.failure
    assert len(model.calls) == 1