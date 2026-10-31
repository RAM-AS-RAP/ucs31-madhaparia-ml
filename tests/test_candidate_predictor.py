"""Candidate input and safety behavior without loading model artifacts."""

import numpy as np
import pytest

from inference import candidate_predictor
from scripts.candidate_feature_builder import load_candidate_schema


class FakeModel:
    classes_ = np.asarray(["A", "B"])

    def __init__(self):
        self.calls = []
        self.probabilities = np.asarray([[0.9, 0.1]])

    def predict_proba(self, frame):
        self.calls.append(frame.copy())
        return self.probabilities.copy()


@pytest.fixture
def example(monkeypatch):
    schema = load_candidate_schema()
    model = FakeModel()
    policy = {
        "policy_id": "test_research_policy",
        "classes": ["A", "B"],
        "mode": "thresholds",
        "min_reporting_units": 2,
        "min_score": 0.78,
        "min_margin": 0.20,
        "probability_representation": "original",
        "temperature": None,
    }

    monkeypatch.setattr(
        candidate_predictor,
        "load_candidate_resources",
        lambda: (model, schema, policy),
    )

    features = {name: 0.0 for name in schema["feature_names"]}
    features.update({
        "AGE": 22,
        "SEX_M": 0,
        "E_89__present": 1,
        "E_91__present": 1,
    })

    return model, schema, features


def test_valid_input_uses_schema_order_and_returns_assessment(example):
    model, schema, features = example

    result = candidate_predictor.predict_candidate(features)

    assert result["status"] == "prediction_available"
    assert result["predicted_condition"] == "A"
    assert result["reported_unit_count"] == 2
    assert result["prediction_performed"] is True
    assert list(model.calls[0].columns) == schema["feature_names"]
    assert result["safety_flags"] == []


def test_one_unit_skips_model_prediction(example):
    model, _, features = example
    features["E_91__present"] = 0

    result = candidate_predictor.predict_candidate(features)

    assert result["status"] == "insufficient_information"
    assert result["predicted_condition"] is None
    assert result["model_score"] is None
    assert result["prediction_performed"] is False
    assert model.calls == []


def test_regional_pain_and_parent_count_as_one_unit(example):
    model, schema, features = example

    for concept in schema["symptom_features"]:
        features[f"{concept}__present"] = 0

    features["E_53__present"] = 1
    features["head_region_pain__present"] = 1

    result = candidate_predictor.predict_candidate(features)

    assert result["reported_unit_count"] == 1
    assert result["status"] == "insufficient_information"
    assert model.calls == []


def test_safety_flags_are_preserved_and_prediction_is_skipped(example):
    model, _, features = example

    result = candidate_predictor.predict_candidate(
        features,
        safety_flags=["existing_safety_flag"],
        include_diagnostics=True,
    )

    assert result["status"] == "safety_review_required"
    assert result["safety_flags"] == ["existing_safety_flag"]
    assert result["safety_review_required"] is True
    assert result["predicted_condition"] is None
    assert "diagnostics" not in result
    assert model.calls == []


def test_safety_only_symptom_cannot_be_dropped_as_merely_unsupported(example):
    model, _, features = example

    result = candidate_predictor.predict_candidate(
        features,
        unsupported_symptoms=["blue_lips"],
    )

    assert result["status"] == "safety_review_required"
    assert result["safety_flags"] == ["blue_lips"]
    assert result["unsupported_symptoms"] == ["blue_lips"]
    assert model.calls == []


def test_unsupported_symptom_is_not_silently_ignored(example):
    model, _, features = example

    result = candidate_predictor.predict_candidate(
        features,
        unsupported_symptoms=["sneezing"],
    )

    assert result["status"] == "unsupported_input"
    assert result["predicted_condition"] is None
    assert result["unsupported_symptoms"] == ["sneezing"]
    assert model.calls == []


def test_explicit_negative_evidence_blocks_prediction(example):
    model, _, features = example
    features["E_201__absent"] = 1

    result = candidate_predictor.predict_candidate(features)

    assert result["status"] == "unsupported_input"
    assert result["unsupported_absent_concepts"] == ["E_201"]
    assert result["predicted_condition"] is None
    assert model.calls == []


def test_contradictory_states_are_rejected(example):
    model, _, features = example
    features["E_89__absent"] = 1

    with pytest.raises(ValueError, match="Conflicting"):
        candidate_predictor.predict_candidate(features)

    assert model.calls == []


def test_region_without_required_parent_is_rejected(example):
    model, _, features = example
    features["head_region_pain__present"] = 1

    with pytest.raises(ValueError, match="requires present parent"):
        candidate_predictor.predict_candidate(features)

    assert model.calls == []


@pytest.mark.parametrize("age", [-1, 121, True, np.nan])
def test_invalid_age_is_rejected(example, age):
    model, _, features = example
    features["AGE"] = age

    with pytest.raises(ValueError):
        candidate_predictor.predict_candidate(features)

    assert model.calls == []


def test_extra_feature_is_rejected(example):
    _, _, features = example
    features["sneezing"] = 1

    with pytest.raises(ValueError, match="Feature mismatch"):
        candidate_predictor.predict_candidate(features)


def test_missing_feature_is_rejected(example):
    _, _, features = example
    del features["AGE"]

    with pytest.raises(ValueError, match="Feature mismatch"):
        candidate_predictor.predict_candidate(features)


def test_invalid_binary_value_is_rejected(example):
    _, _, features = example
    features["SEX_M"] = 2

    with pytest.raises(ValueError, match="0 or 1"):
        candidate_predictor.predict_candidate(features)


def test_context_string_is_not_mistaken_for_a_list_of_flags(example):
    _, _, features = example

    with pytest.raises(ValueError, match="collection"):
        candidate_predictor.predict_candidate(
            features, safety_flags="blue_lips"
        )


def test_low_score_withholds_public_prediction_after_model_call(example):
    model, _, features = example
    model.probabilities = np.asarray([[0.6, 0.4]])

    result = candidate_predictor.predict_candidate(
        features, include_diagnostics=True
    )

    assert result["prediction_performed"] is True
    assert result["status"] == "insufficient_information"
    assert result["predicted_condition"] is None
    assert result["model_score"] is None
    assert result["diagnostics"]["research_only"] is True
    assert len(model.calls) == 1