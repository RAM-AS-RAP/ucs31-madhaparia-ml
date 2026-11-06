# COVID / sensory / duration experiment — research only, not activated

## What was actually completed

A new fixed-parameter logistic research classifier was fitted using recorded GCCR002 survey measurements and reported laboratory COVID outcomes, rather than invented COVID symptoms or disease-derived durations. The source is pinned to commit `d717674e3b86c3c7c314af05ea44497abc88a622`.

- Cohort: **4,694** complete compatible source rows — 4,148 reported-lab positive and 546 reported-lab negative.
- Split: **2,816 train / 939 validation / 939 test**. No reported participant ID or identical complete predictor profile crosses partitions. This does not establish clinical independence beyond the supplied source identifiers or eliminate near-duplicate/source bias.
- Same cohort and partitions for every feature ablation; train-only standardisation; fixed regularisation; selection by validation log loss before final test evaluation.
- Selected feature set: **all 19 features**, including actual reported onset interval, taste/smell-change checkboxes, and separate numeric taste/smell ratings before/during illness.
- Exported portable JSON scores agree with the fitted pipeline on **all 939 test rows**.
- No change to the active V3 model, application database, patient UI, supported clinical classes or patient result policy.

## Final reserved test results

| Inputs | ROC AUC | Log loss | Brier score |
|---|---:|---:|---:|
| Symptoms + age | 0.6229 | 0.3543 | 0.1022 |
| Symptoms + age + onset interval | 0.6218 | 0.3546 | 0.1022 |
| Symptoms + age + sensory measurements | 0.7381 | 0.3220 | 0.0936 |
| All features | **0.7407** | **0.3215** | **0.0936** |
| Constant training-cohort prevalence baseline | 0.5000 | 0.3634 | 0.1042 |

The selected model's source-cohort ROC AUC bootstrap interval is **0.6964–0.7880** (300 paired complete-profile-cluster resamples, percentile 95% interval). This is **not diagnostic accuracy, population transportability, external validation, or a patient-specific disease probability**.

The time-only addition did not improve the held-out results: its paired ROC AUC difference interval versus symptoms alone is **−0.0148 to +0.0134**, which includes no improvement. The combined sensory addition improved discrimination on this source split; it does not prove taste alone is useful independently of smell, or that taste change identifies COVID.

### Why it is not deployed

At the **descriptive, non-approved 0.5 cutoff**, the selected model's confusion matrix is:

| Actual reported laboratory outcome | Predicted negative | Predicted positive |
|---|---:|---:|
| Negative | 10 | **101** |
| Positive | **8** | 820 |

That cutoff misclassifies 101/111 negative test respondents. A positive-majority study means even a constant-positive decision looks superficially “accurate”; therefore a high accuracy/sensitivity number alone would be misleading. No patient-facing threshold was selected, and no post-test threshold tuning was used to make this look deployable.

## What this model does NOT establish

- COVID-negative respondents are not confirmed influenza, URTI, healthy or TB-negative cases. This is **not COVID versus flu/TB or a 22-class differential**.
- The 2020 crowd-sourced, chemosensory-ascertained sample is not current local prevalence. Outcomes are reported laboratory results, not independently linked contemporary assays.
- `Days_since_onset` is overall respiratory-illness onset time, not automatically cough duration. Taste/smell measurements are explicit 0–100 survey ratings; vague free text cannot be assigned invented values.
- The export requires every selected study-compatible input. Sparse reports and unknown ratings fail validation rather than being silently imputed.
- These are within-source random/group-held-out results, not an independently collected clinical test cohort. Model/feature selection and final comparisons are recorded transparently; do not reuse this test set to tune the next candidate and call it untouched.
- No clinical/language/governance reviewer has approved the classifier for patient use, and the working app still correctly reports that it does not predict COVID.

## Reproduce

From the ML component directory, using the project's backend virtual environment:

```powershell
& '..\..\ucs31-madhaparia-backend-main\ucs31-madhaparia-backend-main\.venv\Scripts\python.exe' scripts/train_covid_research.py --download --output reports/covid_research_repeat
& '..\..\ucs31-madhaparia-backend-main\ucs31-madhaparia-backend-main\.venv\Scripts\python.exe' -m pytest tests/test_covid_research.py -q
```

The script refuses to overwrite saved evaluation evidence. Raw processed respondent files and upstream attribution/licence are preserved under ignored `data/external/gccr002/`; hashes, source identity, software versions and partition-set fingerprints are in `evaluation.json`. No raw individual records are copied into this report.

Files: `evaluation.json`, `covid_survey_research_v1.json`. An initial fixed-parameter run before uncertainty reporting is retained separately in `reports/covid_research/`; parameters, splits and feature choices were not retuned using that test output.

## Attribution and next decision

Source: [GCCR002](https://github.com/GCCR/GCCR002). Gerkin et al., *Recent Smell Loss Is the Best Predictor of COVID-19 Among Individuals With Recent Respiratory Symptoms*, [doi:10.1093/chemse/bjaa081](https://doi.org/10.1093/chemse/bjaa081). Preserve upstream licence/attribution; published data and code are research resources, not clinical deployment approval.

Before any integration, obtain suitable comparator-diagnosis data, a compatible prospective input protocol, agreed validation/selection criteria, independent evaluation and appropriate clinical review. Do not splice this binary score into the original disease rankings or turn its negatives into flu labels.
