# Calibration and uncertainty evaluation

16,741 reserved simulated source groups per view, across 21 conditions. The five views reuse the same groups and are not independent samples. These results are separate from the earlier model-selection validation results and from the final test.

## Classification before withholding

Temperature scaling preserves the leading class. Precision, recall and F1 below are macro averages across the 21 conditions.

| View | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Full | 94.09% | 95.21% | 91.23% | 92.37% |
| Up to 1 | 44.44% | 59.78% | 38.41% | 40.36% |
| Up to 2 | 69.85% | 72.94% | 65.39% | 66.17% |
| Up to 3 | 85.07% | 88.63% | 80.74% | 81.39% |
| Up to 5 | 93.38% | 94.89% | 90.70% | 91.94% |

## Frozen policy results

Accepted accuracy uses only accepted cases. Coverage uses all cases. The interval is a two-sided 95% Wilson interval.

| View | Accepted / total | Coverage | Accepted accuracy | Accuracy interval | Correct accepted / all |
| --- | --- | --- | --- | --- | --- |
| Full | 14,136 / 16,741 | 84.44% | 98.61% | 98.40% to 98.79% | 83.26% |
| Up to 1 | 0 / 16,741 | 0.00% | undefined | undefined | 0.00% |
| Up to 2 | 7,084 / 16,741 | 42.32% | 91.23% | 90.55% to 91.87% | 38.61% |
| Up to 3 | 11,967 / 16,741 | 71.48% | 93.92% | 93.48% to 94.34% | 67.14% |
| Up to 5 | 14,081 / 16,741 | 84.11% | 97.85% | 97.60% to 98.08% | 82.30% |

## Calibration comparison

Lower values are better. Brier score is summed across classes. ECE is top-label ECE with ten equal-width bins.

| View | Representation | Log loss | Brier score | ECE |
| --- | --- | --- | --- | --- |
| Full | original | 0.216518 | 0.097630 | 0.077023 |
| Full | temperature_scaled | 0.148697 | 0.083447 | 0.011012 |
| Up to 1 | original | 1.740056 | 0.709921 | 0.090738 |
| Up to 1 | temperature_scaled | 1.663813 | 0.705430 | 0.094155 |
| Up to 2 | original | 0.959075 | 0.434888 | 0.142243 |
| Up to 2 | temperature_scaled | 0.830190 | 0.407467 | 0.036665 |
| Up to 3 | original | 0.515728 | 0.238405 | 0.125633 |
| Up to 3 | temperature_scaled | 0.410649 | 0.215128 | 0.015982 |
| Up to 5 | original | 0.248965 | 0.111891 | 0.085595 |
| Up to 5 | temperature_scaled | 0.177148 | 0.096496 | 0.007690 |

## Interpretation limits

- Aggregate targets do not guarantee performance for each condition.
- Accepted-case accuracy must be reported alongside coverage and all-case per-condition recall.
- Withheld cases remain in the denominator of correct-accepted recall.
- Temperature scaling does not improve every calibration metric in every view; inspect the one-unit ECE results.
- The unlabelled tiredness/chest-pain probe still returns pulmonary neoplasm at approximately 78.02%, just above the score threshold. This is a remaining limitation, not evidence of diagnostic correctness.
- Probe handling does not constitute a complete application safety assessment.
- Clinical performance for Kenyan users has not been established.
