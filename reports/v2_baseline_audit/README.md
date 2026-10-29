# V2 Baseline Audit

Related issue: #44

## Purpose

Establish a reproducible baseline for investigating misleading V2
predictions from limited symptom information.

This audit records existing behaviour. It does not retrain the model,
change its features or implement fixes.

## Model and audit scope

- Model: rf_v2_23class_prototype.joblib
- Model input features: 66
- Prediction classes: 23
- Audit demographics: age 22, female
- Direct confirmed-symptom cases: 9
- Free-text NLP cases: 12
- Prediction records across both reports: 14

The remaining cases were skipped by the audit because they had no
supported model features or triggered its red-flag guard.

The examples have no verified disease labels. They are behaviour
probes, not diagnostic accuracy measurements.

## Reproduction

Run from the ML repository root with the project environment active:

    python -B scripts/audit_v2_baseline.py
    python -B scripts/audit_v2_nlp_baseline.py

Generated reports:

- baseline.json
- nlp_baseline.json

Both reports record source/model SHA-256 fingerprints, runtime versions
and the Git HEAD commit. File fingerprints identify the actual files
used, including scripts that may not yet have been committed.

The reports also record timestamps, so repeated executions are not
expected to produce byte-identical JSON files. Compare case inputs,
encoded features, scores and source fingerprints.

Running the commands again overwrites the corresponding reports.
Retain the committed issue #44 results as the original baseline when
evaluating later candidates.

## Direct prediction findings

| Confirmed symptoms | Leading condition | Model score | Uncertain |
|---|---|---:|---|
| Tiredness and headache | Pulmonary neoplasm | 33.61% | Yes |
| Fever and cough | Tuberculosis | 16.45% | Yes |
| Tiredness and chest pain | Pulmonary neoplasm | 33.61% | Yes |
| Tiredness only | COVID-19 | 28.48% | Yes |
| Headache only | Viral pharyngitis | 16.14% | Yes |
| Fever only | Tuberculosis | 13.21% | Yes |
| Cough only | Tuberculosis | 11.12% | Yes |

These scores are uncalibrated model outputs, not established
probabilities that a patient has a condition.

### Loss of pain location

The canonical headache mapping contains:

- E_53: general pain
- E_55_@_V_89: forehead pain location

The canonical chest-pain mapping contains:

- E_53: general pain
- E_55_@_V_101: upper-chest pain location

The V2 schema retains E_53 but excludes both location features.

Consequently, headache and chest pain become identical V2 inputs.
Adding tiredness to either produces the same encoded vector and
ranked prediction scores.

The audit explicitly records excluded features. This exclusion is
part of the audit's representation of the current V2-supported inputs;
it is not a correction to the schema.

### Uncertainty does not withhold a condition

All seven direct cases that reached inference were flagged uncertain
by the existing confidence assessment.

The assessment still returned a leading condition. This establishes
the need to evaluate an insufficient-information outcome in later work.

### Symptoms without model features

Sneezing exists in the canonical catalogue but has no model features.
The audit skipped the sneezing-only and empty-confirmation cases.

These skip outcomes belong to the audit harness.

## NLP findings

The ML repository's NLP mapper:

- Recognised tiredness and headache correctly in the tested sentence.
- Produced equivalent fever/cough inputs for English, mixed
  English/Kiswahili and the tested spelling errors.
- Recognised denied cough as absent.
- Extracted 21 days from "cough for three weeks".
- Identified the sister's cough as belonging to another person.
- Flagged the positive coughing-blood example.
- Recognised denied coughing up blood as absent.

The audit simulated confirmation of present patient symptoms only.
Negated symptoms were recorded but not passed as explicit negative
model evidence.

Duration was recorded by NLP but was not included in V2 inference.

Some interpreted wording remained in unmatched output, including
"no" and "for three weeks". This should be reviewed in issue #45.

The ML NLP results do not establish equivalent behaviour in the
running backend. Backend and frontend code were not changed.

## Existing labelled evaluation

The following figures come from previously saved evaluation reports.
These evaluations were reviewed, not rerun during this audit.

All rows below use 55,808 simulated DDXPlus validation cases whose
true labels cover the original 21 conditions. V2 can predict 23 classes.

| Evaluation input | V2 accuracy | V2 macro F1 |
|---|---:|---:|
| Full available V2 representation | 89.34% | 86.42% |
| Up to 1 supported positive evidence code | 29.75% | 34.62% |
| Up to 2 supported positive evidence codes | 60.77% | 61.62% |
| Up to 3 supported positive evidence codes | 78.07% | 77.52% |

Sources:

- [Full-input comparison](../comparison_v1_v2/comparison_metrics.json)
- [Sparse-input comparison](../comparison_v1_v2/sparse_all_conditions_metrics.json)
- [V2 validation metrics](../validation_v2_23class/validation_metrics.json)

The sparse-input experiments use structured evidence codes, not
free-text extraction. Age and sex are also supplied.

COVID-19 and allergic rhinitis have no positive ground-truth cases
in these validation evaluations, so their sensitivity is not established.

A previous overlap audit identified 2,886 validation cases matching
original training inputs, approximately 5.17% of validation records.
This measures overlap with the original training partition, not the
exact selected V2 training subset.

Source:

- [Original-partition overlap audit](../comparison_v1_v2/duplicate_impact_audit.json)

These results do not establish real-world clinical performance.

## Verification performed

- Existing ML tests: 97 passed, 5 subtests passed.
- Both audit scripts completed successfully.
- Report case counts and case-ID uniqueness were checked.
- All 14 prediction records contained 23 unique ranked conditions.
- Scores were finite, within bounds, correctly ordered and summed to 1.
- Encoded feature order and count matched the saved schema.
- Equivalent direct and NLP inputs produced matching predictions.
- Recorded source/model fingerprints matched the current files.

The existing test suite is separate from the report-consistency checks.

## Follow-up work

- #45: Audit symptom meaning and correct the proposed feature schema.
- #46: Review condition coverage and evidence quality.
- #47: Prepare reproducible candidate data and evaluation partitions.
- #48: Train and compare candidate models.
- #49: Evaluate calibration and insufficient-information handling.
- #50: Complete final evaluation and document the integration contract.

Issue #44 establishes the baseline. The recorded prediction and
representation problems remain unresolved at this stage.