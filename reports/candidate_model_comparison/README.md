# Candidate Model Comparison

Issue: #48
Status: validation comparison completed; candidate not deployed

## Selected research candidate

rf_group_weighted was selected using the prespecified mean macro F1
across four sparse validation settings.

| Model | Mean sparse macro F1 | Full-observation accuracy | Full-observation macro F1 |
| --- | ---: | ---: | ---: |
| Frozen V2, harmonized | 64.46% | 90.73% | 87.60% |
| rf_group_weighted | 69.78% | 94.10% | 92.59% |
| rf_class_balanced | 67.05% | 94.53% | 91.32% |

The selected candidate improved mean sparse macro F1 by approximately
5.32 percentage points over harmonized V2.

The class-balanced model had higher full-observation accuracy, but did
not win under the predefined sparse-input selection criterion.

These are validation results on simulated DDXPlus data, not final-test
results or evidence of clinical performance in Kenyan users.

## Evaluation design

The primary cohort contains 52,844 validation source-input groups without
a normalized original-input match in the original training source pool.

Additional reports cover all 55,680 prepared validation groups and the
2,836 legacy-overlap groups separately.

Every model received the same full and sparse observations.
Sparse settings retain up to 1, 2, 3 or 5 reporting units.

A regional pain concept and its required general-pain parent count as
one reporting unit, although they activate two positive feature columns.

V2 receives the supported 66-feature projection; candidates receive
70 features.

V2's two allergy probabilities are combined before canonical ranking.
Its COVID-19 probability is retained without renormalizing other scores.

Macro metrics use 21 fixed ground-truth conditions. COVID-19 detection
sensitivity cannot be evaluated because there are no positive cases.

## Fitting and held-out groups

Both candidates used:
- 334,588 fitting source groups.
- 662,036 fitting observations.
- Identical forest settings and fitting groups.

Their sample-weight strategies differed.

83,635 prepared training groups remain reserved for calibration and
uncertainty-policy development in issue #49. They were not used for
candidate fitting or selection.

Training durations on the current machine:
- rf_group_weighted: approximately 190.0 seconds.
- rf_class_balanced: approximately 164.2 seconds.

These timings are descriptive, not a controlled benchmark.

## Important regressions

Aggregate improvement does not mean every condition improved.

In the up-to-1-unit primary validation setting:
- Tuberculosis recall decreased from 74.88% to 12.56%.
- The selected candidate detected 0 of 26 bronchiolitis cases.
- Overall macro recall decreased despite higher accuracy and macro F1.

The per-condition change tables and figures retain these regressions.

Sparse-input accuracy remains limited: the selected candidate achieved
43.25% accuracy in the up-to-1-unit setting.

## Original symptom examples

Direct feature-input probes used age 22 and female sex.

| Input | Frozen V2 top result | Selected candidate top result |
| --- | --- | --- |
| Tiredness and headache | Pulmonary neoplasm, 33.61% | URTI, 35.62% |
| Tiredness and chest pain | Pulmonary neoplasm, 33.61% | Pulmonary neoplasm, 54.96% |
| Fever and cough | Tuberculosis, 16.45% | Viral pharyngitis, 23.60% |

The candidate preserves the head/chest distinction that V2 loses.

These examples have no ground-truth diagnoses. Their model scores are
not accuracy measurements or calibrated clinical risks.

The probes assume confirmed source-defined concepts. They do not test
the NLP, frontend, backend or deployed prediction pipeline.

The chest-pain result demonstrates why a simple score cutoff of 50%
does not establish sufficient evidence for a disease conclusion.

## Documentation files

- comparison_table.md: percentage table for the primary cohort.
- summary.csv: accuracy, macro precision/recall/F1 and weighted F1.
- per_condition.csv: precision, recall, F1, support and prediction counts.
- confusion_counts.csv: underlying confusion-matrix counts.
- selected_vs_v2_per_condition.csv: per-condition changes.
- selection.json: selection scores and selected candidate.
- prediction_examples.json: encoded probe inputs and complete rankings.
- prediction_examples_top3.csv: concise probe results.
- Training, partition, view and comparison manifests: provenance.
- figures/: 300-dpi PNG and scalable SVG charts.

Raw prediction archives and model artifacts remain in the ignored
data/candidate_experiment_v1 directory.

## Limitations and next steps

Input groups are not verified patient identities. Reduced-feature
collisions remain, including identical inputs with different labels.

This experiment changes several aspects relative to V2, including data
preparation, condition labels, features, fitting data and forest settings.
It does not isolate the causal contribution of each change.

The initial candidate covers 21 conditions; COVID-19 remains a conditional
extension outside candidate training.

Issue #49 must evaluate calibration and insufficient-information handling
using the reserved groups. Calibration or abstention will not automatically
repair missed conditions or remove dataset biases.

If the candidate cannot meet the chosen reliability requirements, further
data or model work will be needed.

Final-test evaluation belongs to issue #50.
No model was deployed or substituted into the backend or frontend.