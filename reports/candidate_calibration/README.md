# Candidate calibration and uncertainty evaluation

Issue: #49

The selected `rf_group_weighted` research candidate now has a frozen
temperature-scaling configuration, an evaluated uncertainty policy,
and a separate candidate inference interface.

The deployed V2 model, frontend and backend were not changed.
This work does not authorize deployment.

## Experiment separation

The 83,635 source groups reserved during candidate training were divided into:

| Partition | Source groups | Purpose |
| --- | ---: | --- |
| calibration_fit | 50,174 | Fit scalar temperature |
| policy_development | 16,720 | Select probability representation and uncertainty thresholds |
| uncertainty_evaluation | 16,741 | Evaluate the frozen configuration without retuning |

These groups were excluded from candidate model fitting.

Each partition has five observation views: full, up to one, two, three
and five reporting units. Views reuse the same source groups; they must
not be pooled and described as independent patients.

The final test split remains separate for issue #50.

## Calibration

Scalar temperature scaling was fitted using five views per calibration
group, with weight 0.2 per observation and total weight 1 per group.

- Fitted temperature: approximately 0.640839.
- Calibration-fit log loss: 0.739102 before, 0.649942 after.
- Mean policy-development log loss: 0.735544 before, 0.646433 after.
- Selected representation: `temperature_scaled`.

The fitting loss is not held-out performance. Representation selection
used the prespecified policy-development criterion.

Temperature scaling preserved the leading class. It changes scores,
not which condition ranks first. Because the fitted temperature is below
one, it sharpens the probability distribution.

On the uncertainty-evaluation partition, log loss and Brier score improved
in all five views. Top-label ECE improved in four views but worsened in
the one-unit view, from approximately 0.090738 to 0.094155.

Scores are not clinically validated individual disease risks.

## Frozen uncertainty policy

The development search evaluated 250 predefined threshold pairs.
Of these, 110 met the development criteria.

The selected policy requires all of:

- At least two supported positive reporting units.
- Leading assessed score at least 0.78.
- Difference between the leading and second score at least 0.20.

A regional pain concept and its required general-pain parent count
together as one reporting unit.

For each required development view, a qualifying policy needed at least
100 accepted observations and a two-sided 95% Wilson accuracy interval
whose lower endpoint was at least 90%.

Required views were full, up to two, up to three and up to five units.
The one-unit view must withhold predictions.

These are research selection criteria, not clinical safety standards.

## Frozen-policy evaluation

The policy was evaluated on 16,741 reserved source groups per view.
No thresholds or temperature parameters were changed during evaluation.

| View | Coverage | Accepted accuracy | Wilson accuracy lower endpoint |
| --- | ---: | ---: | ---: |
| Full | 84.44% | 98.61% | 98.40% |
| Up to 1 | 0.00% | Undefined | Undefined |
| Up to 2 | 42.32% | 91.23% | 90.55% |
| Up to 3 | 71.48% | 93.92% | 93.48% |
| Up to 5 | 84.11% | 97.85% | 97.60% |

All required views met the frozen research targets.

Coverage is accepted observations divided by all observations.
Accepted accuracy excludes withheld observations.

Full-view accuracy before withholding was approximately 94.09%.
The 98.61% accepted accuracy applies only to the accepted 84.44% of cases.

These evaluation results use a different partition from issue #48's
model-selection results. The earlier 94.10% full-view accuracy should
not be presented as the same measurement.

See [evaluation_tables.md](evaluation_tables.md) for accuracy, macro
precision, macro recall, macro F1 and calibration metrics.

## Per-condition limitations

Aggregate targets do not guarantee adequate performance for each condition.

Examples from the uncertainty-evaluation partition:

- Tuberculosis, up-to-two-unit view: 110 correctly accepted cases out of
  614 true cases, giving correct-accepted recall over all cases of 17.92%.
- Bronchiolitis: only 11 true cases were available. In the up-to-two-unit
  view, two true cases were accepted, both incorrectly; correct-accepted
  recall was zero.
- The source-labelled `Larygospasm` condition had 216 true cases. All were
  withheld in every view, despite 100% leading-class recall before
  withholding.

Correct-accepted recall over all cases uses every true case as its
denominator, including withheld cases. It differs from recall calculated
only among accepted cases.

Full per-condition results are retained in
[per_condition.csv](recorded_results/uncertainty_evaluation/per_condition.csv).

## Unlabelled inference probes

Direct feature-input probes used age 22 and female sex.

| Input | Public response |
| --- | --- |
| Tiredness and headache | Insufficient information; condition and score withheld |
| Fever and cough | Insufficient information; condition and score withheld |
| Tiredness and chest pain | Pulmonary neoplasm; assessed model score approximately 78.02% |

The chest-pain example remains a limitation. Its original candidate
score was approximately 54.96%; temperature scaling increased it to
approximately 78.02%, slightly above the score threshold.

These examples have no ground-truth diagnosis. They do not establish
diagnostic correctness or measure accuracy. The frozen policy was not
retuned to suppress this example.

The audit also checked:

- Withholding for fewer than two reporting units.
- Blocking supplied safety flags.
- Blocking unsupported symptoms and unevaluated explicit negative evidence.
- Rejecting contradictory present/absent states.
- Rejecting regional pain without its required general-pain parent.
- Preserving distinct head-region and chest-region inputs.
- Keeping research diagnostic rankings out of default responses.

Default and diagnostic-mode responses must match, allowing only an
absolute numerical tolerance of 1e-12 for the public model score.
Statuses, conditions and other public fields must match exactly.

See [inference_probes.json](inference_probes.json) and
[inference_probes.csv](inference_probes.csv).

## Candidate inference boundary

The separate interface is implemented in
`inference/candidate_predictor.py` and `inference/candidate_assessment.py`.

Possible response statuses include:

- `prediction_available`
- `insufficient_information`
- `unsupported_input`
- `safety_review_required`

Malformed or contradictory feature inputs raise validation errors.

Withheld responses expose neither a public predicted condition nor a
public model score. Optional research diagnostics are not user conclusions.

The candidate interface accepts encoded features. It does not perform NLP
or a complete application safety assessment. Supplied safety flags and
recognised safety-only unsupported codes block prediction, but callers
must retain the application's complete safety checks.

Explicit negative evidence is blocked because it was not evaluated in
this candidate experiment.

## Documentation artifacts

- [Calibration metrics](figures/calibration_metrics.png)
- [Coverage and accuracy](figures/coverage_and_accuracy.png)
- [Reliability diagrams](figures/reliability_diagrams.png)
- [Per-condition recall](figures/per_condition_recall.png)

Each figure is also available as SVG in the same folder.

`recorded_results` preserves 23 original JSON/CSV files, including
partition details, calibration results, threshold-search results,
evaluation metrics and manifests.

Historical status fields describe the stage at which each artifact was
created. The separate evaluation decision records the completed
frozen-policy evaluation.

Large prediction archives and model artifacts remain in ignored local
data folders. They are required to reproduce the verification.

## Validation and reproduction

The full test suite passed with 344 tests and five subtests.

`verify_candidate_calibration.py` recomputed the uncertainty-evaluation
metrics from saved prediction arrays and verified:

- Source-group identities, labels and reporting-unit counts.
- Temperature transformation and unchanged leading-class indices.
- Exact acceptance masks.
- Nested JSON results.
- Ten probability summary rows.
- Five policy summary rows.
- 210 probability per-condition rows.
- 105 policy per-condition rows.
- 100 reliability-bin rows.
- 22 reporting-unit summary rows.
- Exported result and documentation fingerprints.

The verification uses the existing tested metric implementations.
It checks consistency with saved predictions; it is not an independent
clinical validation or a rerun of model inference.

Development and temperature-fitting metrics were not recomputed by this
verification.

Relevant commands, run from the ML repository root:

```powershell
python -B -m pytest tests/ -q
python -B scripts/audit_candidate_inference.py
python -B scripts/export_candidate_calibration_results.py
python -B scripts/plot_candidate_calibration.py
python -B scripts/verify_candidate_calibration.py
```

Report-producing scripts refuse to overwrite existing outputs.
These commands document the workflow; do not blindly rerun them over
completed artifacts.

## Remaining work

Issue #50 covers final-test evaluation and the ML integration contract.

The data is simulated and does not establish clinical performance for
Kenyan users. Sparse-input limitations, rare-condition support and
condition-specific withholding require explicit consideration before
any deployment decision.