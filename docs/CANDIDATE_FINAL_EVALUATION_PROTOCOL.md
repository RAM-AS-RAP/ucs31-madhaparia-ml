# Candidate Final Evaluation Protocol

Issue: #50

Status: specified before issue #50 test predictions.

## Objective

Evaluate the frozen research candidate on the official DDXPlus test
split and document the candidate ML integration contract.

Report classification performance, probability calibration, uncertainty
coverage and per-condition limitations.

This evaluation does not authorize deployment or establish clinical
performance for Kenyan users.

## Previous test-set use

The repository documents earlier model evaluation on test data.

An older experimental script, evaluate_v2_final_test.py, also targets
the official test source. Whether that script was run and whether its
results influenced subsequent choices could not be confirmed.

Therefore, describe the present work as evaluation of the frozen
candidate on the official test split.

Do not claim that this split has never been inspected or provides a
completely untouched project-wide final test.

The candidate workflow in issues #47–#49 kept this split outside its
documented preparation, selection and calibration procedures. This
does not erase earlier project-level exposure.

## Frozen configuration

Preserve the selected configuration from issues #48 and #49:

- Candidate: rf_group_weighted.
- Condition scope: 21 canonical respiratory-related conditions.
- Input schema: 70 numerical features representing 34 symptom concepts
  plus age and recorded sex.
- Probability representation: temperature_scaled.
- Temperature: the exact stored value, approximately 0.640839.
- Minimum supported reporting units: 2.
- Minimum leading score: 0.78.
- Minimum top-two margin: 0.20.

Use exact stored artifacts and verify their fingerprints.
Do not recreate parameters from rounded documentation values.

Preserve the frozen V2 model and its 66-feature schema for comparison.

Do not evaluate rf_class_balanced to reconsider candidate selection.

## No tuning on test results

Do not use final-test results to:

- Retrain or choose a different model.
- Change feature definitions or condition mappings.
- Refit temperature.
- Change score, margin or reporting-unit thresholds.
- Select symptom masks, cohorts or demographic cutoffs.
- Suppress particular example predictions.

Report failed targets and condition-specific regressions.

If a defect prevents valid evaluation, document the defect, any test
results already seen, the correction and the invalidated run. Preserve
the audit history rather than silently overwriting results.

## Source and condition mapping

Use the official source:

data/raw/release_test_patients

Use the existing candidate derivation and canonical condition mapping.

Map source-labelled Allergic sinusitis to Allergic rhinitis, consistent
with the documented source-definition review.

COVID-19 remains outside the candidate's supported condition set.

Count out-of-scope source records separately. This in-scope evaluation
does not measure reliable recognition of out-of-scope diseases.

Do not invent symptoms, explicit negative evidence or demographic values.

## Preparation before predictions

First audit source structure, eligibility, repeated inputs and overlap.
Do not calculate model predictions during this preparation stage.

Record:

- Raw source fingerprint and row count.
- In-scope and out-of-scope record counts.
- Per-condition support.
- Invalid demographics or evidence.
- Records without supported positive concepts.
- Repeated normalized source-input groups.
- Conflicting labels within a normalized source-input group.
- Overlap with original training and validation source inputs.
- Candidate-feature collisions and their label ambiguity.

Invalid evidence, invalid demographics or conflicting source-group
labels require explicit review. Do not silently repair them.

## Source grouping and cohorts

Reuse the existing normalized source-input signature:

- Numeric age without truncation.
- Recorded sex.
- Sorted unique original evidence codes.
- No pathology or differential-diagnosis field in the signature.

Keep one deterministic representative per normalized test source-input
group: the first eligible source row.

Retain source row numbers, group sizes, source labels and canonical labels.

Define the cohorts before inspecting predictions:

1. Primary:
   Eligible unique test groups without a normalized source-input match
   in either the original in-scope training or validation source pools.

2. Complete eligible test cohort:
   All eligible unique test groups, including overlapping groups.

3. Overlap cohort:
   Eligible unique test groups matching either earlier source pool.

Identify training matches, validation matches and matches to both.
Do not change the frozen training data to remove newly discovered overlap.

A match with a source pool does not prove that a record was selected
for model fitting. These exclusions are conservative.

Do not delete groups merely because their reduced candidate features
match another group's features.

Normalized input groups are not verified patient identities.
Removing exact input overlap does not establish complete independence.

## Zero-supported-feature records

Count in-scope records with no supported positive concepts explicitly.

Keep them outside classifier accuracy and calibration calculations
that require eligible supported observations.

Report their counts and the resulting eligibility denominator.
Demonstrate that the candidate interface withholds a leading condition
for zero-supported-feature inputs.

Do not present coverage among eligible observations as coverage across
all raw records or all possible application users.

## Observation views

For each eligible source group, prepare:

- full
- up_to_1
- up_to_2
- up_to_3
- up_to_5

Reuse the existing deterministic view generator and seed 42.
Prepare each shared view once for both compared models.

A regional pain concept and its general-pain parent count together as
one reporting unit.

Hidden or unreported concepts remain unknown.
Explicit negative evidence is not introduced.

When fewer units are available than requested, retain the available
units and report the actual count.

Full means all supported positive concepts available from the source.
It does not mean complete clinical information.

Do not pool repeated views as independent patients.

## Frozen V2 comparison

Project the same observations into V2's supported feature columns.
V2 cannot use the added regional-pain features.

Apply the existing probability harmonization:

- Combine Allergic sinusitis and Allergic rhinitis into Allergic rhinitis.
- Retain COVID-19 as an additional possible V2 output.
- Do not remove COVID-19 probability and renormalize.
- Recompute the canonical leading class after combining allergy scores.

Report unfiltered classification metrics for harmonized V2 and the
selected candidate on the same cohorts and views.

Do not apply the candidate's thresholds to V2 as if they were evaluated
V2 thresholds.

This dataset cannot establish COVID-19 sensitivity.

## Classification and calibration reporting

For each cohort and view, report:

- Source-group count and per-condition support.
- Accuracy.
- Macro precision, recall and F1 over the fixed 21 ground-truth classes.
- Weighted F1.
- Per-condition precision, recall and F1.
- Confusion counts and clearly labelled normalized confusion matrices.

Retain COVID-19 as a possible erroneous V2 prediction.

For the candidate, report original and temperature-scaled:

- Multiclass log loss.
- Multiclass Brier score, summed across classes.
- Top-label ECE using ten equal-width bins.
- Reliability-bin counts, mean scores and observed accuracy.

Check that temperature scaling preserves leading-class indices.

Use undefined values where a metric's denominator is absent.
Document the existing zero-contribution convention for macro metrics.

## Frozen uncertainty-policy reporting

For each cohort and view, report:

- Total, accepted and withheld counts.
- Coverage and abstention rate.
- Accuracy and error rate among accepted observations.
- Two-sided 95% Wilson accuracy intervals.
- Correct accepted predictions divided by all observations.
- Per-condition acceptance and withholding.
- Correct accepted predictions divided by all true cases of each class.
- Accepted-case per-condition metrics with explicit denominators.
- Results grouped by actual reporting-unit count.

Zero accepted observations mean undefined accepted accuracy.

On the primary cohort, assess the existing research targets separately
for full, up_to_2, up_to_3 and up_to_5:

- At least 100 accepted observations.
- Accepted-accuracy Wilson lower endpoint at least 0.90.

The up_to_1 view must withhold all predictions.

If a required cohort or denominator is unavailable, report the result
as not assessable. Do not silently replace it with the complete cohort.

Do not treat meeting aggregate targets as clinical readiness.

## Limitations requiring explicit review

Retain and examine the previously identified limitations:

- Sparse-input tuberculosis recall.
- Small bronchiolitis sample support.
- Complete withholding of source-labelled Larygospasm in the reserved
  uncertainty evaluation.
- The unlabelled tiredness/chest-pain probe returning pulmonary neoplasm
  at approximately 78.02%.
- Temperature scaling sharpening an incorrect or unsupported conclusion.
- One-unit calibration limitations.
- Reduced-feature collisions across different condition labels.
- Simulated data and unestablished performance for Kenyan users.

Do not tune the frozen configuration to remove these findings.

## ML integration contract

Document the existing candidate interface, including:

- Required feature names, order, values and schema version.
- Supported age and recorded-sex encoding.
- Present, absent and unknown symptom semantics.
- Regional-pain parent consistency and reporting-unit counting.
- Unsupported symptoms and explicit negative evidence.
- Safety flags and the separate application safety responsibilities.
- Response statuses, null fields and uncertainty reasons.
- Invalid-input errors.
- Default responses versus optional research diagnostics.
- Exact model, schema, calibration and policy dependencies.
- Local artifact locations and current loading requirements.
- Differences from the deployed V2 interface.

The contract must distinguish implemented behavior from future
integration work.

Document that model artifacts and prediction archives are currently
stored in ignored local data folders. A repository checkout alone
does not supply these runtime artifacts.

Do not change frontend or backend code, replace deployed models or
claim that the candidate is production-ready.

## Reproducibility and completion

Preserve fingerprints for sources, preparation code, model, schema,
temperature, policy, views, predictions and reports.

Keep final-evaluation artifacts separate from earlier experiments.
Refuse silent overwrites.

Verify reported metrics against saved prediction arrays.
Export documentation tables and PNG/SVG figures.

Complete issue #50 with an explicit research conclusion, integration
contract and remaining limitations, regardless of whether the existing
research targets are met.