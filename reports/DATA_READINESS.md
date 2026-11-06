# Remaining COVID/duration/taste work: source audit and research boundary

Checked 4 November 2026. This is a source-readiness review, **not clinical approval or an exhaustive search of every available dataset**. The search integration returned HTTP 402; the primary repositories, dictionaries and papers below were inspected directly instead. No private or access-restricted records were accessed.

## What can now be done honestly

A separate, reproducible **GCCR002 binary research experiment** can use genuinely recorded symptom, taste/smell-rating and onset measurements with respondents' reported laboratory COVID outcomes. This is materially different from inventing COVID profiles or assigning durations from disease labels.

Its target is **reported-lab COVID-positive versus COVID-negative among surveyed people with recent respiratory illness**. It is not a model of confirmed COVID versus confirmed influenza, TB, and all 21 original classes. A negative COVID test does not identify the alternative illness.

The active application's V3 model remains unchanged; the research candidate is not silently deployed or combined with its scores.

## Sources reviewed

| Source | Relevant evidence | Why it does not complete the entire requested classifier |
|---|---|---|
| [Zoabi et al. COVID prediction](https://github.com/nshomron/covidpred), [paper](https://www.nature.com/articles/s41746-020-00372-6) | Public tested-individual symptom records; the paper describes RT-PCR-confirmed results and eight predictors | Paper explicitly says smell/taste were not recorded; no symptom-duration feature among the eight predictors; negative tests are not labelled flu/TB cases. Reporting bias is acknowledged |
| [Coswara](https://github.com/iiscleap/Coswara-Data), [dictionary](https://raw.githubusercontent.com/iiscleap/Coswara-Data/master/csv_labels_legend.json), [paper](https://www.nature.com/articles/s41597-023-02266-0) | Audio and symptom metadata; 2,635 subjects described in the paper; reported test status/date/type | Its `loss_of_smell` field is documented as combined “Loss of Smell & Taste”, not a separately measured taste feature. Test/record dates are not symptom-onset duration. No complete confirmed differential across the original conditions |
| [COUGHVID paper](https://www.nature.com/articles/s41597-021-00937-4), [correct dataset DOI](https://doi.org/10.5281/zenodo.4048311) | Crowdsourced cough audio and self-reported status, with expert impressions for a subset | An audio task is outside this text app's scope; expert cough impressions are not laboratory-confirmed differential diagnoses. The paper reports low diagnostic inter-rater agreement. No audio subsystem was added |
| [GCCR001](https://github.com/GCCR/GCCR001) | Published taste/smell survey study and explicit data reuse information | Useful context, not proof that a single symptom uniquely identifies COVID or a full 22-class labelled cohort |
| [GCCR002](https://github.com/GCCR/GCCR002), [paper](https://doi.org/10.1093/chemse/bjaa081) | Public processed measurements and target memberships: 4,148 reported-lab positive and 546 reported-lab negative respondents; separate taste/smell ratings and `Days_since_onset` | Supports a limited binary within-study experiment. No lab-confirmed influenza/TB target membership; crowdsourcing/chemosensory ascertainment, class imbalance and 2020 transportability limits. Does not establish current Kenyan clinical performance |
| [Carbon Health/Braid Health](https://github.com/mdcollab/covidclinicaldata), pinned source audit below | Separate smell/taste, overall symptom onset, COVID results and rapid-flu columns across 93,995 rows | Only one positive rapid-flu result, no complete measurement row for that flu-positive case, and no TB labels; cannot support the requested joint classifier |

The source dictionaries/headers and outcome-group counts were inspected directly; the study's published performance is **not** claimed as performance of this app or our newly trained candidate.

## Reproducible GCCR002 experiment

Local script: `ucs31-madhaparia-ml-main/ucs31-madhaparia-ml-main/scripts/train_covid_research.py`.

- Source pinned to commit `d717674e3b86c3c7c314af05ea44497abc88a622` (3 January 2021).
- Preserve upstream `LICENSE` and `README.md`; root repository supplies an MIT licence. Preserve attribution and study citation. Research reuse here is not legal/regulatory approval for clinical deployment.
- Download only processed symptom measurements, target definitions and metadata, not respiratory audio or a whole repository. Verify recorded SHA-256 hashes on repeat use.
- Raw respondent-level files stay under ignored `data/external/gccr002/`; do not publish them with an application release or treat the public source as permission to collect identifiable patient data.
- Use only the reported-laboratory positive/negative target groups. Exclude other/unknown/clinical-only diagnosis groups from this binary evaluation; do not relabel them as flu or negatives.
- Use a matched complete-valid measurement cohort for all four ablations. No duration fabrication, no model-imputed negative symptoms, and no arbitrary conversion of vague text to sensory ratings.
- Compare symptoms alone, symptoms plus duration, symptoms plus sensory information, and all features. Use the same fixed train/validation/test partitions for each.
- Group identical complete predictor profiles independently of labels to avoid exact-profile train/test overlap. Validate participant identifiers; reject duplicate source IDs. This cannot prove that different source IDs always represent independent clinical patients or eliminate near-duplicate/source bias.
- Fit a fixed regularised logistic model with training-only standardisation. Choose the feature set by validation log loss before examining final test metrics. Do not choose a model because an appealing prompt gets a desired answer.
- Report ROC AUC, average precision, log loss, Brier score, a descriptive 0.5 confusion matrix, and a constant training-prevalence baseline. The descriptive threshold is not a patient-facing safety gate.
- Add paired profile-cluster bootstrap intervals to disclose within-source statistical uncertainty and feature-ablation differences. These do not measure transportability or clinical validity.
- Export a portable JSON candidate and verify score parity against the fitted pipeline. Require complete, explicit, finite, study-compatible inputs at inference. Do not silently fill sparse user reports with invented values.

Final experiment outputs are under the ML root's `reports/covid_research_final/`. An initial run before adding uncertainty reporting is retained separately under `reports/covid_research/`; model choices/parameters/splits were not tuned using its test scores.

## Important semantic boundaries

1. `Days_since_onset` means days since respiratory illness onset in this survey. It is **not automatically the duration of a particular cough**, the interval since exposure, or days since a COVID test.
2. The study uses explicit 0–100 ratings before and during illness. “I do not taste well” must not be automatically converted into a zero score or a normal pre-illness rating of 100.
3. A food-flavour-change checkbox is broader than clinically established pure taste loss. Smell and taste ratings are separate features; do not merge them to make an input fit.
4. Both COVID-positive and COVID-negative respondents can have taste/smell changes. Feature associations do not establish that a symptom causes or confirms COVID.
5. Scores from a binary COVID-test-outcome model cannot be inserted as a twenty-second mutually exclusive class in the V3 model or renormalised together with its disease rankings. The targets/populations differ, and co-infection is possible.
6. No-source-data assumption, an old published AUC, or a successful script run establishes modern clinical validity. The patient app still honestly reports that it does not predict COVID.

## What is genuinely still blocked

- **Joint clinical disease discrimination:** needs an appropriate, licensed, deidentified cohort with trustworthy COVID and comparator diagnosis labels, aligned measurements, timing and missingness definitions. The inspected sources do not supply all original classes plus COVID as a validated joint target.
- **Clinical acceptance and external evaluation:** needs an appropriate clinical reviewer, intended-use protocol and independently collected/adjudicated evaluation data. Source-cohort splits and bootstrap intervals are not a substitute.
- **Language and patient-guidance approval:** needs a fluent Swahili reviewer and appropriate clinician review. Automated tests do not supply human sign-off.
- **Deployment approval:** requires applicable governance/privacy/security/regulatory work. It cannot be produced or signed on a user's behalf by this coding session.

These are release gates, not checkboxes that can be truthfully marked complete by adding more code. Until met, retain the working awareness app, educational COVID information and separate research evaluation rather than asserting a clinically complete diagnostic final version.

## Additional comparator audit — 5 November 2026

Executed `release/audit_comparator_source.py` against all 29 public CSV batches at Carbon Health/Braid Health commit `91a46906f13faf215df7f7828b7f087ee504b6a4`. The audit verifies each downloaded Git blob/size, records SHA-256 hashes and retains **aggregate counts only**, not respondent rows or chest-image links. Source terms: CC BY-NC-SA 4.0. Attribution: Carbon Health and Braid Health, *Coronavirus Disease 2019 (COVID-19) Clinical Data Repository*, 2020, version 10-20-2020.

| Measurement | Count |
|---|---:|
| Source rows, not proven independent patients | 93,995 |
| Missing rapid-flu result | 93,741 |
| Rapid-flu negative | 253 |
| Rapid-flu positive | 1 |
| COVID-positive / flu-negative rows | 23 |
| COVID-negative / flu-positive rows | 1 |
| Both tests negative, **not** an alternative-disease label | 230 |
| COVID-positive / flu-negative with all eleven explicit binary symptoms and valid overall-onset days | 7 |
| COVID-negative / flu-positive with those complete measurements | 0 |

Onset is missing in 78,130 rows. This is overall illness onset, not per-symptom duration. A rapid-flu result does not identify TB or all other respiratory conditions. The source provides no patient grouping identifier, excludes severe cases and describes its sampling as clinically driven and nonsystematic. Taste/smell/onset column names alone do not make this a suitable clinical training cohort.

**Decision:** do not train or deploy a COVID/flu/TB classifier from these records, do not turn COVID-negative records into flu/TB labels, and do not fill absent observations with invented symptoms. This source does not close the learned duration/taste or joint-differential gaps. The existing GCCR002 binary experiment remains offline. General web search again returned HTTP 402; this is a direct-source audit, not an exhaustive claim that no suitable data exists anywhere.

Evidence: [aggregate audit](additional-source-audit.json), `additional-source-audit.log`, and the executable audit's built-in missingness/label checks. No model was retrained, activated or given new clinical claims.
