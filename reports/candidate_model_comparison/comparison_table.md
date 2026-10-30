# Candidate Model Validation Comparison

These are validation results, not final-test or clinical results.
Scores below are percentages.

## Primary comparison cohort

This cohort excludes normalized original-input matches with the
original DDXPlus training source pool.

| Model | View | Groups | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_v2_harmonized | full | 52,844 | 90.73% | 87.64% | 89.42% | 87.60% | 90.59% |
| frozen_v2_harmonized | up_to_1 | 52,844 | 28.94% | 47.75% | 39.66% | 33.52% | 29.98% |
| frozen_v2_harmonized | up_to_2 | 52,844 | 59.94% | 64.30% | 64.91% | 60.44% | 59.39% |
| frozen_v2_harmonized | up_to_3 | 52,844 | 77.99% | 78.08% | 79.89% | 77.24% | 77.70% |
| frozen_v2_harmonized | up_to_5 | 52,844 | 89.10% | 86.81% | 88.43% | 86.65% | 88.98% |
| rf_group_weighted | full | 52,844 | 94.10% | 95.02% | 91.67% | 92.59% | 93.80% |
| rf_group_weighted | up_to_1 | 52,844 | 43.25% | 56.42% | 38.01% | 38.49% | 40.10% |
| rf_group_weighted | up_to_2 | 52,844 | 68.91% | 76.39% | 65.36% | 65.95% | 67.31% |
| rf_group_weighted | up_to_3 | 52,844 | 84.62% | 86.94% | 81.87% | 82.53% | 83.95% |
| rf_group_weighted | up_to_5 | 52,844 | 93.32% | 94.72% | 91.12% | 92.16% | 93.02% |
| rf_class_balanced | full | 52,844 | 94.53% | 91.41% | 93.29% | 91.32% | 94.39% |
| rf_class_balanced | up_to_1 | 52,844 | 31.56% | 45.42% | 46.94% | 33.31% | 31.97% |
| rf_class_balanced | up_to_2 | 52,844 | 64.39% | 64.23% | 71.84% | 62.77% | 64.84% |
| rf_class_balanced | up_to_3 | 52,844 | 84.06% | 81.32% | 85.64% | 81.31% | 83.91% |
| rf_class_balanced | up_to_5 | 52,844 | 93.74% | 91.05% | 92.73% | 90.83% | 93.60% |

## Prespecified selection score

Unweighted mean macro F1 across the four sparse settings
on the primary comparison cohort.

| Model | Mean sparse macro F1 |
| --- | ---: |
| frozen_v2_harmonized | 64.46% |
| rf_group_weighted | 69.78% |
| rf_class_balanced | 67.05% |

Candidate selected for further investigation: **rf_group_weighted**.

Selection does not authorize deployment or establish improvement
for every condition.

## Interpretation

- Macro metrics use the fixed set of 21 ground-truth conditions.
- Undefined contributions to macro averages use zero.
- Per-condition CSV values are blank where a denominator is unavailable.
- V2 allergy probabilities are combined before canonical ranking.
- V2's COVID-19 score is retained; an incorrect COVID-19 prediction
  remains an error.
- COVID-19 sensitivity cannot be measured without positive cases.
- Full and sparse views are separate settings, not independent patients.
- Full-validation and legacy-overlap results are in summary.csv.
- No calibration-reserved or final-test observations were evaluated.
