# Final evaluation tables

Primary cohort: 53,272 eligible unique source-input groups from the official test split. Earlier project-level test exposure is documented in CANDIDATE_FINAL_EVALUATION_PROTOCOL.md. This is not claimed to be a never-inspected project-wide test.

## Classification before withholding

Precision, recall and F1 are macro averages over the fixed 21 ground-truth conditions.

| View | Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| Full | Frozen V2 (harmonized) | 90.86% | 88.42% | 89.96% | 88.44% |
| Full | Selected candidate | 94.25% | 95.17% | 92.30% | 93.03% |
| Up to 1 | Frozen V2 (harmonized) | 29.56% | 48.10% | 39.50% | 32.99% |
| Up to 1 | Selected candidate | 43.28% | 56.50% | 38.05% | 38.24% |
| Up to 2 | Frozen V2 (harmonized) | 59.72% | 64.96% | 64.79% | 60.73% |
| Up to 2 | Selected candidate | 69.20% | 75.07% | 65.46% | 66.10% |
| Up to 3 | Frozen V2 (harmonized) | 78.42% | 79.09% | 80.82% | 78.42% |
| Up to 3 | Selected candidate | 84.78% | 86.72% | 82.15% | 82.83% |
| Up to 5 | Frozen V2 (harmonized) | 89.27% | 87.55% | 88.98% | 87.45% |
| Up to 5 | Selected candidate | 93.50% | 94.59% | 91.77% | 92.46% |

## Candidate policy

Accepted accuracy excludes withheld cases. Coverage uses all eligible cases. Intervals are two-sided 95% Wilson accuracy intervals.

| View | Accepted / eligible | Coverage | Accepted accuracy | Accuracy interval | Correct accepted / all |
| --- | --- | --- | --- | --- | --- |
| Full | 45,668 / 53,272 | 85.73% | 98.68% | 98.58% to 98.78% | 84.60% |
| Up to 1 | 0 / 53,272 | 0.00% | undefined | undefined | 0.00% |
| Up to 2 | 22,260 / 53,272 | 41.79% | 90.82% | 90.43% to 91.19% | 37.95% |
| Up to 3 | 38,343 / 53,272 | 71.98% | 93.84% | 93.59% to 94.08% | 67.54% |
| Up to 5 | 45,553 / 53,272 | 85.51% | 97.87% | 97.73% to 98.00% | 83.69% |

## Candidate calibration

Lower is better. Brier score is summed across classes. ECE uses ten equal-width top-label score bins.

| View | Representation | Log loss | Brier score | ECE |
| --- | --- | --- | --- | --- |
| Full | original | 0.208942 | 0.094531 | 0.073210 |
| Full | temperature_scaled | 0.144749 | 0.081073 | 0.011945 |
| Up to 1 | original | 1.772563 | 0.723449 | 0.089345 |
| Up to 1 | temperature_scaled | 1.702499 | 0.720468 | 0.097865 |
| Up to 2 | original | 0.981871 | 0.445717 | 0.142177 |
| Up to 2 | temperature_scaled | 0.855644 | 0.419053 | 0.036138 |
| Up to 3 | original | 0.524117 | 0.243135 | 0.124749 |
| Up to 3 | temperature_scaled | 0.420712 | 0.219678 | 0.019897 |
| Up to 5 | original | 0.244573 | 0.109859 | 0.082098 |
| Up to 5 | temperature_scaled | 0.177100 | 0.095151 | 0.008410 |

## Interpretation

- Complete eligible and overlap-cohort results remain in the recorded CSVs.
- The separately recorded zero-supported group is outside these denominators.
- Repeated views reuse the same groups and are not independent patients.
- Aggregate target success does not establish adequate recall for each condition.
- Per-condition correct-accepted recall includes withheld true cases.
- One-unit ECE worsened after temperature scaling.
- Sparse tuberculosis and bronchiolitis results and complete Larygospasm withholding remain limitations.
- The earlier unlabelled chest-pain probe remains a documented concern.
- These simulated-data results do not establish clinical performance for Kenyan users or authorize deployment.
