# Candidate Calibration and Uncertainty Protocol

Issue: #49
Status: specified before reserved-data prediction analysis

## Objective

Evaluate probability calibration and implement an experimental
insufficient-information policy for rf_group_weighted.

The candidate remains a research model. This work does not establish
clinical reliability or authorize deployment.

## Frozen inputs

Preserve:
- The selected rf_group_weighted model from issue #48.
- Its 21-condition scope and 70-feature schema.
- The prepared candidate dataset and existing experiment partitions.
- The existing V2 model and deployed inference path.

Do not retrain the forest during this experiment.

COVID-19 remains outside the candidate's supported conditions.

## Reserved-data subdivision

Use only the 83,635 calibration_reserved source groups.

Within each canonical condition, sort source-group IDs by:
SHA256("49:calibration_partition:" + source_group_id)
Break any hash tie using source_group_id.

Allocate:
- First floor(0.60 * class_group_count): calibration_fit.
- Next floor(0.20 * class_group_count): policy_development.
- Remaining groups: uncertainty_evaluation.

Require every condition in all three partitions.

All observations derived from a source group must stay together.
Record group counts, per-condition support and partition fingerprints.

Do not use the existing validation-selection cohorts to fit calibration
or choose uncertainty thresholds.

Do not inspect the final test split.

## Observation views

Start from each reserved group's full candidate observation.

Use the existing deterministic view-generation logic for:
- full
- up_to_1
- up_to_2
- up_to_3
- up_to_5

Keep hidden concepts unknown and preserve regional-pain consistency.

A regional-pain concept and its general-pain parent count as one
reporting unit.

For calibration fitting, give each view weight 1/5 per source group.
This defines an experimental mixture of five observation settings;
it is not an estimate of how Kenyan users report symptoms.

Report evaluation results separately for each view.
Do not treat repeated views as independent patients.

## Calibration experiment

Compare:
1. Original model probabilities.
2. Scalar temperature scaling of log probabilities.

For temperature scaling:
- Clip input probabilities below 1e-12 before taking logarithms.
- Compute softmax(log(probability) / temperature).
- Fit one positive temperature using calibration_fit only.
- Minimize the weighted multiclass negative log likelihood.
- Restrict temperature to [0.25, 4.0].
- Record optimizer settings, convergence and boundary solutions.

Temperature scaling is intended to adjust probability sharpness while
preserving the leading-class ranking. Verify this property.

Select temperature scaling only if it improves the mean log loss
across the five policy_development views by more than 1e-6.
Otherwise retain the original probabilities.

This selection does not establish that every condition is calibrated.

## Calibration measurements

For original and temperature-scaled probabilities, report:
- Multiclass log loss.
- Multiclass Brier score: mean sum of squared errors over 21 classes,
  without division by the number of classes.
- Top-label reliability diagrams using 10 equal-width bins.
- Top-label expected calibration error with the same bins.
- Accuracy, macro precision, macro recall and macro F1.
- Per-condition precision, recall, F1 and support.

Retain bin counts and mean scores in report tables.

Log loss and Brier score measure more than calibration alone.
Assess them alongside reliability diagrams.

## Insufficient-information policy

Apply the policy to the selected probability representation.

Require at least two supported positive reporting units before a
leading condition can be returned.

This is an experimental information gate, not evidence that two
symptoms are sufficient for diagnosis.

For eligible observations, evaluate a global threshold grid:
- Minimum top score: 0.50 through 0.99, in increments of 0.01.
- Minimum top-two margin: 0.00, 0.05, 0.10, 0.15 or 0.20.
- Accept only when both thresholds are met.

Choose thresholds using policy_development only.

A threshold pair is eligible when, separately in each of full,
up_to_2, up_to_3 and up_to_5:
- At least 100 source groups receive an accepted prediction.
- The 95% Wilson lower bound for accepted-prediction accuracy
  is at least 0.90.

These are prespecified research targets, not clinical safety standards.
Development intervals are descriptive after threshold selection;
they are not guarantees.

Among eligible pairs:
1. Maximize mean coverage across those four views.
2. Break ties using higher mean accepted-prediction accuracy.
3. Then prefer the higher score threshold.
4. Then prefer the higher margin threshold.

If no pair qualifies, record that no policy met the research targets.
Do not silently relax the criteria. Withhold leading conditions by
default until a revised experiment is specified.

## Locked evaluation

Freeze the selected calibration choice and thresholds before evaluating
uncertainty_evaluation.

Report for every view:
- Total, accepted and withheld counts.
- Coverage and abstention rate.
- Accuracy and error rate among accepted predictions.
- Confidence intervals with their denominators.
- Per-condition acceptance rates.
- Correct accepted predictions divided by all true cases of each class.
- Per-condition results among accepted cases, including support.
- Results grouped by actual reporting-unit count.

Treat zero accepted cases as undefined accuracy, not perfect accuracy.

Assess whether the development targets also hold in evaluation.
Report failures without retuning on uncertainty_evaluation.

Existing issue #48 validation results may be used for a secondary
comparison after the policy is frozen. They are not untouched test data.

## Inference behavior

Implement a separate candidate inference interface.

When information is insufficient, return a structured status and
reasons with no leading predicted condition.

Keep diagnostic rankings separate from any user-facing conclusion.

Reject invalid or contradictory inputs explicitly.
Do not silently encode unsupported symptoms as supported evidence.

Explicit negative symptoms are outside this experiment's evaluated
positive-only observation distribution. Do not present predictions
using them as validated.

Safety flags must remain separate from disease confidence and must
not be suppressed by abstention.

Low confidence cannot reliably identify every out-of-scope condition.
A high score does not prove that an input belongs to the model's scope.

## Regression probes

Re-run the original age-22, female examples:
- Tiredness and headache.
- Tiredness and chest pain.
- Fever and cough.

Also check empty, unsupported-only and one-unit inputs.

These are unlabelled behavior checks.
Do not tune thresholds to obtain preferred diseases for these examples.

## Reporting and boundaries

Preserve model, dataset, code and output fingerprints.
Export tables and reliability/coverage figures for documentation.

Retain the tuberculosis and bronchiolitis recall limitations from
issue #48. Calibration and abstention do not repair those limitations.

Keep generated artifacts separate from deployed models.
Do not change the frontend or backend.

Final-test evaluation and the integration contract belong to issue #50.