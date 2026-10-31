# Candidate ML Integration Contract

Issue: #50

Status: research interface implemented and evaluated on simulated data.
Application integration and deployment remain separate work.

## 1. Interface and model identity

The Python entry point is:

```python
from inference.candidate_predictor import predict_candidate

result = predict_candidate(
    feature_values,
    safety_flags=(),
    unsupported_symptoms=(),
    include_diagnostics=False,
)
```

This function accepts one confirmed, encoded observation.

It does not accept raw symptom text, perform NLP, expose an HTTP
endpoint, or provide a complete application safety assessment.

The evaluated configuration is:

| Setting | Value |
|---|---|
| Candidate | rf_group_weighted |
| Response model version | candidate_v1_research |
| Policy ID | candidate_v1_research_uncertainty_v1 |
| Numerical features | 70 |
| Symptom concepts | 34 |
| Conditions | 21 |
| Probability representation | temperature_scaled |
| Temperature | 0.6408387217626598 |
| Minimum supported reporting units | 2 |
| Minimum leading score | 0.78 |
| Minimum top-two score margin | 0.20 |

Use the stored policy and schema as the authoritative configuration.
Do not independently round or replace their values.

The candidate has a different schema and condition scope from V2.
It is not a direct replacement for the existing V2 model artifact.

## 2. Feature dictionary

`feature_values` must be a dictionary containing exactly the names
in the candidate schema's `feature_names` list.

The 70 features consist of:

- AGE.
- SEX_M.
- 34 symptom-present indicators.
- 34 symptom-absent indicators.

The schema controls the model's column order. Dictionary insertion
order does not determine the order passed to the model.

### Demographics

AGE must be a finite numeric value between 0 and 120, inclusive.
Boolean ages are rejected.

SEX_M uses the source dataset's encoding:

- 0: female.
- 1: male.

There is no evaluated unknown-sex encoding. A caller must not silently
default an unknown value to either category.

### Symptom states

For each concept:

| State | `__present` | `__absent` |
|---|---:|---:|
| Present | 1 | 0 |
| Explicitly absent | 0 | 1 |
| Unknown or unreported | 0 | 0 |

A concept cannot be both present and absent.

Supply numerical 0/1 indicators. Missing features, extra features,
non-finite values and invalid indicator values are rejected.

Unreported symptoms must remain unknown. They must not be converted
into explicit negative evidence.

Although the schema represents explicit absence, the current candidate
policy was not evaluated for such inputs. Any supplied absent concept
blocks prediction with `unsupported_input`.

Do not discard an explicit denial merely to obtain a prediction.

### Regional pain

Present `head_region_pain` requires present `E_53`.

Present `chest_region_pain` requires present `E_53`.

Confirm the reported region before encoding it. Do not invent an exact
pain location from a broad region report.

A region and its required general-pain parent do not count as two
independent reporting units. The feature builder determines reporting
units; callers must not substitute the number of positive columns.

For example, tiredness plus confirmed head-region pain produces two
reporting units despite having three positive concept indicators.

## 3. Caller responsibilities

Before calling the predictor, the integrating application must:

1. Extract and confirm the intended symptom concepts.
2. Preserve the source definitions and relevant qualifiers.
3. Apply the candidate schema's regional-pain mapping rules.
4. Preserve unreported concepts as unknown.
5. Retain unsupported symptoms separately.
6. Run its safety assessment and pass the resulting safety flags.

Existing NLP aliases are not automatically validated by their inclusion
in a numerical schema.

The current interface does not establish that NLP, language handling,
symptom confirmation or the complete application safety workflow works
correctly with the candidate. Those require integration testing.

Duration is not a numerical feature in this schema. Preserve it as
context rather than inventing a model feature.

## 4. Context arguments

`safety_flags` and `unsupported_symptoms` accept a list, tuple, set or
frozenset of non-empty strings.

A bare string is rejected.

Values are stripped of surrounding whitespace, deduplicated and sorted.
They are not converted to lowercase or translated into canonical codes.

Any supplied safety flag blocks a model conclusion.

The following exact unsupported-symptom codes are also promoted to
safety flags:

- blue_lips
- confusion
- fainting

Other unsupported symptoms also block prediction. For example, sneezing
has no supported numerical feature in this candidate.

A mixture of supported and unsupported symptoms remains
`unsupported_input`; unsupported evidence must not be silently dropped.

These checks do not constitute a complete safety assessment.
The absence of a supplied flag does not establish that a case is safe.

`include_diagnostics` must be a boolean and defaults to false.

## 5. Response statuses

For structurally valid input and available, verified resources:

| Status | Meaning | Public condition and score |
|---|---|---|
| prediction_available | The leading condition passed the frozen research policy | Populated |
| insufficient_information | Too few reporting units, or score/margin requirements failed | None |
| unsupported_input | Unsupported symptoms or unevaluated explicit negative evidence were supplied | None |
| safety_review_required | Safety flags require review | None |

For valid input, safety flags take priority over unsupported-input and
information-gate statuses.

Malformed feature input raises an exception; it does not produce one
of these normal response statuses.

### Common response fields

| Field | Meaning |
|---|---|
| model_version | Candidate response version |
| policy_id | Frozen uncertainty-policy identifier |
| status | One of the statuses above |
| predicted_condition | Leading condition, or None when withheld |
| model_score | Assessed leading score from 0 to 1, or None when withheld |
| uncertain | False only when the research policy accepts the result |
| uncertainty_reasons | Machine-readable reasons for withholding |
| reported_unit_count | Supported positive reporting units |
| probability_representation | Selected score representation |
| message | Explanatory text |
| prediction_performed | Whether model probabilities were calculated |
| safety_flags | Normalized safety flags |
| safety_review_required | Whether safety flags are present |
| unsupported_symptoms | Normalized unsupported symptom codes |
| unsupported_absent_concepts | Explicitly absent concepts that block prediction |

`uncertain=False` means the result passed the research policy.
It does not mean the condition is confirmed.

The `score_note` field is present after model scoring. It is not present
on responses blocked before scoring.

### Prediction performed versus prediction available

These are different properties.

A response can have:

- `prediction_performed=True`, because probabilities were calculated.
- `status="insufficient_information"`, because the result failed the
  score or margin requirement.
- `predicted_condition=None` and `model_score=None`.

For early blocking due to safety flags, unsupported evidence or fewer
than two reporting units, `prediction_performed=False`.

A false prediction-performed value does not mean resource loading was
skipped. The current implementation loads and validates resources
before returning these early responses.

### Reason codes

The interface can return:

- safety_flags_present
- unsupported_symptoms_present
- explicit_negative_evidence_not_evaluated
- too_few_supported_reporting_units
- below_score_threshold
- close_alternative
- no_qualifying_research_policy

The last code supports a withhold-all policy. The currently selected
policy uses thresholds.

Consumers should use status and reason codes for application logic,
rather than matching the explanatory message text.

## 6. Research diagnostics

When enabled and model scoring occurs, `diagnostics` contains:

- `research_only=True`.
- The top-two assessed score margin.
- Ranked conditions with rank, condition, original_model_score and
  assessed_model_score.

Early blocked responses do not contain diagnostic rankings.

Diagnostics must not be used to bypass a withheld public result.
A research ranking is not an alternative public diagnosis.

Repeated floating-point calculations can differ at machine precision.
Tests comparing numerical scores should use an appropriate tolerance,
while checking statuses, labels and decisions exactly.

## 7. Recorded example behaviour

The inference audit used age 22 and female sex with direct confirmed
feature inputs.

| Example | Recorded public result |
|---|---|
| Tiredness and head-region pain | insufficient_information |
| Fever and cough | insufficient_information |
| Tiredness only | insufficient_information |
| No confirmed symptoms | insufficient_information |
| Unsupported symptom only | unsupported_input |
| Supplied safety flag | safety_review_required |
| Explicit negative evidence | unsupported_input |

Tiredness and chest-region pain passed the policy with Pulmonary
neoplasm as the leading condition and an assessed score of approximately
0.780163.

That example has no verified diagnosis. Passing the policy does not
establish that the prediction is correct.

These probes evaluate interface behaviour, not diagnostic accuracy.

## 8. Runtime artifacts and verification

The current loader requires the candidate schema, model and linked
calibration artifacts.

Under the candidate experiment directory, it reads:

- models/rf_group_weighted/model.joblib
- calibration/uncertainty_policy/policy.json
- calibration/uncertainty_policy/search_manifest.json
- calibration/views/view_manifest.json
- calibration/uncertainty_evaluation/evaluation_manifest.json
- calibration/uncertainty_evaluation/evaluation_decision.json

It also reads the candidate schema and fingerprints the probability
calibration, uncertainty-policy and feature-builder implementations.

The loader checks linked fingerprints, model/schema references,
feature order, class order and the recorded uncertainty-evaluation
decision.

Its runtime evaluation dependency is the reserved uncertainty
evaluation from issue #49. Issue #50's final-test reports provide
additional evidence; they are not new runtime-loader dependencies.

The experiment directory is Git-ignored. A repository clone alone
therefore does not supply the complete runtime artifact set.

Artifact packaging and deployment must preserve the matching files
and their expected layout. Copying only the model file is insufficient.

Resources are cached after successful loading within a process.
Replacing files does not automatically reload the model or policy.
A controlled artifact change requires a fresh process and verification.

## 9. Failure handling

Invalid arguments or inconsistent features raise validation errors.

Missing files, fingerprint mismatches, incompatible artifacts and model
loading failures are operational failures.

An integrating application must distinguish input-validation failures
from operational failures. Not every ValueError represents a user
input mistake.

Do not convert operational failures into an ordinary
`insufficient_information` response.

Do not silently fall back to V2 or bypass resource checks.

This Python contract does not prescribe HTTP routes or status codes.
Those belong to a separately implemented application adapter.

## 10. Interpretation and integration boundaries

The candidate predicts among its 21 included conditions. COVID-19 is
not a candidate output class.

The model does not determine whether the real condition lies outside
its condition scope. A score threshold is not a validated detector of
all out-of-scope cases.

Model scores are not accuracy percentages or clinically validated
individual disease risks.

Official-test-split results use simulated DDXPlus records. Earlier
project-level test exposure is documented or uncertain, so the split
must not be described as never inspected throughout the project.

The evaluation does not establish clinical performance for Kenyan
users. Performance and withholding also vary by condition.

Application integration requires separate validation of:

- Confirmed symptom mapping into the candidate schema.
- Safety and unsupported-evidence propagation.
- Every response status and nullable field.
- Operational-error handling.
- Availability of matching runtime artifacts.
- Presentation of withheld results without exposing research rankings
  as conclusions.

Issue #50 documents the interface and final evaluation. It does not
change the active backend/frontend prediction flow or approve
deployment.