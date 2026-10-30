# Candidate Model Experiment Protocol

Issue: #48
Status: specified before candidate training

## Objective

Compare two candidate Random Forest models with frozen V2 using
documented full and sparse validation observations.

This experiment selects a research candidate. It does not authorize
deployment or establish clinical performance.

## Dataset

Use the verified data/candidate_v1 dataset from issue #47:

- 21 canonical conditions.
- 70 numerical features.
- 418,223 training source-input groups.
- 55,680 validation source-input groups.

COVID-19 remains outside the candidate training scope.

Preserve the generated dataset and its build manifest unchanged.
Store experiment partitions separately.

## Reserve data for issue #49

Partition the prepared training source groups into:

- Approximately 80% for candidate model fitting.
- Approximately 20% reserved for calibration and uncertainty-policy
  development in issue #49.

Stratify by canonical condition.

Within each condition, sort group IDs by a deterministic SHA-256
partition key using seed 42, then reserve floor(0.20 * group_count)
groups, with at least one reserved and one fitting group.

Keep every variant of a source group in the same partition.

Both candidate models must use exactly the same fitting groups.
Exclude all reserved-group variants from model fitting.

Do not inspect reserved-group model performance to select the candidate.
Issue #49 will specify its calibration-fitting and policy-development
subdivision before using those records.

Do not refit the selected model on these reserved groups before calibration.

## Candidate models

Train these candidates sequentially:

1. rf_group_weighted
   Use the dataset's source_group_weight directly.

2. rf_class_balanced
   Multiply source_group_weight by a class-balancing factor calculated
   from fitting source groups only:

       total_fitting_groups / (
           number_of_classes * fitting_groups_for_this_class
       )

The second candidate balances total class weight without allowing the
number of variants to determine class frequencies.

Set class_weight=None for both models. Apply the complete weighting
through sample_weight.

These weights are experimental choices, not Kenyan disease prevalence.

## Shared Random Forest settings

- n_estimators: 180
- criterion: gini
- max_depth: 18
- min_samples_split: 2
- min_samples_leaf: 2
- max_features: sqrt
- max_leaf_nodes: 4096
- bootstrap: true
- class_weight: null
- random_state: 42
- n_jobs: 2

The leaf limit and limited parallelism constrain resource use.
These settings are not identical to the original V2 configuration.

Do not run a broad parameter search in this initial comparison.
Record any necessary protocol amendment before running revised experiments.

## Model inputs

Select only schema.feature_names, in the recorded order.

Do not use labels, group IDs, variant IDs, reporting-unit counts or
weights as predictive features.

Use canonical_label as the target.

## Validation cohorts

Primary comparison:
Use validation source groups without a normalized original-input match
in the original DDXPlus training source.

The issue #47 exclusions identify 2,836 shared groups.
The expected primary cohort therefore contains 52,844 validation groups.

Secondary report:
Evaluate the complete prepared validation cohort of 55,680 groups,
with the 2,836 legacy-overlap groups identified separately.

This distinction matters because frozen V2 was trained before the new
overlap filtering. A source-pool match does not prove that a record was
included in V2's selected fitting subset.

Retained candidate-input collisions remain in both cohorts.
They are not automatically treated as duplicate patients.

Report per-condition support for every cohort.

## Validation observations

Evaluate each cohort using:

- Full available candidate observations.
- Up to 1 reporting unit.
- Up to 2 reporting units.
- Up to 3 reporting units.
- Up to 5 reporting units.

Use deterministic source-group-based seeds and generate each sparse view
once for use by every model.

When fewer units are available, retain all available units and report
the actual count.

Preserve pain-parent consistency. Hidden concepts remain unknown.

Do not combine these views as if they were independent patients.

## Frozen V2 comparison

Preserve the existing V2 artifact and its 66-feature schema.

For a shared observation, project candidate concepts into V2's supported
feature columns. V2 cannot use the two new regional-pain columns.

For canonical probability comparisons:
- Combine V2's Allergic sinusitis and Allergic rhinitis scores into the
  canonical Allergic rhinitis score.
- Retain V2's COVID-19 score as an additional output.
- Do not drop COVID-19 probability and renormalize the remaining scores.
- Recompute the canonical ranking after combining the allergy scores.

Label these results as harmonized V2 results.
Preserve original-label baseline reports separately.

The validation data cannot establish COVID-19 sensitivity.

## Metrics and selection

For each model, cohort and observation setting, report:

- Accuracy.
- Macro F1 across the 21 canonical ground-truth conditions.
- Per-condition precision, recall, F1 and support.
- Confusion counts, retaining COVID-19 as a possible V2 prediction.

Primary candidate-selection score:
The unweighted mean of macro F1 across the four sparse settings
on the primary comparison cohort.

If candidate scores tie exactly, prefer higher full-observation macro F1.
If still tied, prefer rf_group_weighted.

Also report full-observation performance and per-condition regressions.
Do not claim that a higher average improves every condition.

Compare the selected candidate with harmonized frozen V2.
A candidate can be selected for further investigation without outperforming
V2 or being suitable for deployment.

## Original failure examples

Re-run the age-22, female examples:
- Tiredness and headache.
- Tiredness and chest pain.
- Fever and cough.

Record encoded features, ranked model scores and differences from V2.
These unlabelled examples are regression probes, not accuracy tests.

Do not tune the model to produce a preferred disease for these examples.

## Reproducibility and boundaries

Record dataset, partition, code and model fingerprints; package versions;
training settings; sample-weight summaries; and training duration.

Keep candidate artifacts separate from existing model files.
Do not replace deployed models.

Do not inspect the final test split during this experiment.
Final-test evaluation belongs to issue #50.

Probability calibration and insufficient-information handling belong
to issue #49.