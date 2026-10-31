# Final Candidate Evaluation

Issue: #50

Status: frozen candidate evaluated; saved metrics verified; documentation
exported. Deployment is not approved by this evaluation.

## Purpose

Compare the selected respiratory candidate with frozen V2 on the
official DDXPlus test split.

Evaluate classification, probability calibration and the frozen
uncertainty policy without retraining, changing thresholds or selecting
another candidate after observing these results.

The selected candidate is rf_group_weighted, using 70 numerical features
and 21 canonical conditions.

## Evaluation history

Earlier model evaluation on test data is documented in this repository.

Whether an older experimental final-test script was also run, and
whether its results influenced later decisions, could not be confirmed.

These results therefore represent evaluation on the official test
split. They are not claimed to come from a never-inspected project-wide
test set.

The detailed protocol is CANDIDATE_FINAL_EVALUATION_PROTOCOL.md in docs.

## Source groups and cohorts

Records were grouped using normalized source inputs. These groups are
not verified patient identities.

Repeated source inputs were represented once. Eligible groups contain
at least one supported positive reporting unit.

| Cohort | Eligible source groups | Purpose |
|---|---:|---|
| Primary | 53,272 | Excludes source-input groups matching training or validation |
| Complete eligible | 56,632 | All eligible unique test groups |
| Overlap | 3,360 | Eligible test groups matching training or validation |

The overlap cohort contains:

- 3,046 groups matching training only.
- 203 groups matching validation only.
- 111 groups matching both.

One additional unique test group had no supported positive reporting
units. It was evaluated separately and correctly withheld without
calculating model probabilities.

That zero-supported group is outside the eligible-cohort denominators.

Excluding identical source inputs does not eliminate identical
compressed feature vectors. Different source records can produce the
same candidate features, including records with different labels.
These feature collisions were retained.

## Shared observation views

Both models were evaluated using corresponding observations derived
from the same source groups:

- Full supported positive evidence.
- Up to one reporting unit.
- Up to two reporting units.
- Up to three reporting units.
- Up to five reporting units.

A reporting unit is not necessarily one positive feature column.
Regional pain and its required general-pain parent are handled together.

"Up to" does not mean every observation contains exactly that number
of reporting units.

Unreported concepts remain unknown. These views do not evaluate the
use of explicit negative evidence.

The five views reuse the same source groups. They must not be counted
as five independent sets of patients.

## Classification before withholding

The following results use the primary cohort.

| View | V2 accuracy | Candidate accuracy | V2 macro F1 | Candidate macro F1 |
|---|---:|---:|---:|---:|
| Full | 90.86% | 94.25% | 88.44% | 93.03% |
| Up to 1 | 29.56% | 43.28% | 32.99% | 38.24% |
| Up to 2 | 59.72% | 69.20% | 60.73% | 66.10% |
| Up to 3 | 78.42% | 84.78% | 78.42% | 82.83% |
| Up to 5 | 89.27% | 93.50% | 87.45% | 92.46% |

The candidate improves aggregate accuracy and macro F1 in all five
views. Improvement is not uniform across individual conditions.

Full-view candidate macro precision is 95.17%, and macro recall is
92.30%.

Macro metrics use the fixed 21 ground-truth conditions. The exported
tables also include per-condition metrics and confusion counts.

For the V2 comparison, allergy labels are harmonized. V2's COVID-19
prediction probability is retained rather than removed and renormalized.
The test cohort contains no COVID-19 ground-truth cases.

The candidate itself has no COVID-19 output class.

## Frozen uncertainty policy

The evaluated policy uses:

- Temperature: 0.6408387217626598.
- Minimum supported reporting units: 2.
- Minimum assessed leading score: 0.78.
- Minimum top-two assessed score margin: 0.20.

These settings were frozen before the final evaluation.

| View | Coverage | Accepted accuracy | 95% Wilson lower endpoint |
|---|---:|---:|---:|
| Full | 85.73% | 98.68% | 98.58% |
| Up to 1 | 0.00% | Undefined | Undefined |
| Up to 2 | 41.79% | 90.82% | 90.43% |
| Up to 3 | 71.98% | 93.84% | 93.59% |
| Up to 5 | 85.51% | 97.87% | 97.73% |

Coverage is accepted observations divided by all eligible observations.

Accepted accuracy is correct accepted predictions divided by accepted
observations. It excludes withheld observations.

For example, the full view accepts 45,668 of 53,272 observations.
Of those accepted predictions, 45,067 are correct and 601 are incorrect.

Therefore, 98.68% accepted accuracy must not be described as accuracy
on every input. Correct accepted predictions account for 84.60% of
all primary full-view observations.

The one-unit view withholds every prediction. Its accepted accuracy
is undefined, not zero.

All four required views met the predefined aggregate research targets:
at least 100 accepted observations and a lower endpoint of the
two-sided 95% Wilson accuracy interval of at least 90%.

This target assessment does not establish adequate performance for
every condition.

## Probability calibration

Temperature scaling preserves the leading class and therefore does
not change classification accuracy.

On the primary cohort, it reduces log loss and multiclass Brier score
in all five views.

For the full view:

| Metric | Original | Temperature scaled |
|---|---:|---:|
| Log loss | 0.208942 | 0.144749 |
| Multiclass Brier score | 0.094531 | 0.081073 |
| Top-label ECE | 0.073210 | 0.011945 |

Brier score is summed across classes. ECE uses ten equal-width bins.

Calibration improvement is not universal across metrics: one-unit ECE
increases from 0.089345 to 0.097865.

Temperature scaling does not establish that the scores are clinically
validated individual disease risks.

## Condition-specific limitations

Aggregate metrics conceal substantial differences between conditions.

### Tuberculosis

The primary cohort contains 1,861 tuberculosis groups.

In the up-to-two-unit view:

- Unfiltered recall is 41.75%.
- 322 true tuberculosis cases receive a correct accepted prediction.
- Correct accepted predictions cover only 17.30% of all true
  tuberculosis cases.

High accuracy among accepted cases does not remove this limitation.

### Bronchiolitis

The primary cohort contains only 34 bronchiolitis groups.

In the up-to-two-unit view:

- Unfiltered recall is 5.88%.
- Five true bronchiolitis cases receive accepted predictions.
- All five accepted predictions are incorrect.

The small sample size and sparse-input performance limit conclusions
about this condition.

### Larygospasm

All 186 primary cases are withheld in every evaluated view, despite
100% unfiltered recall.

Correct internal rankings therefore do not imply that the interface
returns useful accepted predictions for this condition.

### Unlabelled chest-pain example

The earlier age-22 female tiredness-and-chest-region-pain probe passes
the frozen policy with Pulmonary neoplasm as its leading condition
at approximately 0.780163.

There is no verified diagnosis for this example. Acceptance does not
establish correctness, and the example remains a documented concern.

The final-test results were not used to retune the policy around it.

## Documentation files

- final_evaluation_tables.md: classification, policy and calibration tables.
- source_audit.json: source counts, grouping and overlap audit.
- metrics_verification.json: verification against saved prediction archives.
- results_export_manifest.json: exported-file provenance and fingerprints.
- documentation_manifest.json: chart/table inputs and output fingerprints.
- recorded_results: verified JSON and CSV result copies.
- figures: PNG and SVG documentation figures.

The seven figures cover:

1. Classification accuracy and macro F1.
2. Coverage and accepted accuracy.
3. Calibration metrics.
4. Reliability diagrams.
5. Per-condition recall and correct accepted predictions.
6. Full-view confusion matrices.
7. Up-to-two-unit confusion matrices.

Confusion matrices describe classification before withholding.

Complete-eligible and overlap-cohort results remain in the recorded
tables; the main documentation figures use the primary cohort.

## Verification and reproducibility

The workflow records source, preparation-code, model, policy and
output fingerprints.

Saved observation views were checked against deterministic
reconstruction before prediction.

The metric verifier recomputed results from saved prediction archives
and checked the recorded JSON and CSV outputs. It did not retrain the
model or calculate new model predictions.

This checks consistency with the tested metric implementations; it is
not an independent clinical validation.

Exported JSON and CSV files preserve the verified source bytes.
Chart generation uses the exported results without new predictions.

Large observation files, prediction archives and model artifacts remain
in the ignored local experiment directory. The committed reports do not
alone provide everything required to rerun the experiment.

Completed evaluation outputs are protected against accidental
overwriting. Preserve the existing artifacts and manifests.

## Integration and conclusion

The Python interface is documented in
CANDIDATE_ML_INTEGRATION_CONTRACT.md in docs.

The candidate improves aggregate classification results and meets the
predefined aggregate uncertainty targets on the evaluated simulated
data. Sparse-input weaknesses, condition-specific withholding and
unsupported evidence remain material limitations.

These results do not establish clinical performance for Kenyan users
or authorize deployment.

The active backend/frontend prediction flow is unchanged.