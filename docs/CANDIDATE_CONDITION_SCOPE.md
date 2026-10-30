# Candidate Respiratory Condition Scope

Issue: #46
Status: scope decision for candidate experiments; not deployed

## Purpose

Improve the existing respiratory research model for the intended Kenyan
user population without increasing the disease count simply to change
undesirable predictions.

The scope includes upper-airway conditions, lower-airway conditions,
respiratory infections and pulmonary conditions. It does not provide
general diagnosis of every possible cause of the entered symptoms.

Kenyan representativeness has not been established by the current data.

## Scope decision

Retain the original 21 DDXPlus conditions for candidate experiments,
with the proposed canonical label correction described below.

Keep COVID-19 in the intended scope as a conditional extension requiring
a documented data and evaluation decision.

This produces:

- 21 canonical conditions using DDXPlus alone.
- 22 canonical conditions if COVID-19 is included.
- No separate synthetic Allergic rhinitis class competing with the
  original source label Allergic sinusitis.

The class count is a consequence of the supported condition definitions,
not a target that must be reached.

## Condition inventory and decisions

| Original source label | Candidate label | Decision |
| --- | --- | --- |
| Acute COPD exacerbation / infection | Same as source | Retain |
| Acute laryngitis | Same as source | Retain |
| Acute pulmonary edema | Same as source | Retain within pulmonary scope |
| Acute rhinosinusitis | Same as source | Retain |
| Allergic sinusitis | Allergic rhinitis | Apply documented canonical mapping during candidate preparation |
| Bronchiectasis | Same as source | Retain |
| Bronchiolitis | Same as source | Retain; flag limited data |
| Bronchitis | Same as source | Retain |
| Bronchospasm / acute asthma exacerbation | Same as source | Retain |
| Chronic rhinosinusitis | Same as source | Retain |
| Croup | Same as source | Retain |
| Epiglottitis | Same as source | Retain |
| Influenza | Same as source | Retain |
| Larygospasm | Same as source | Preserve exact source spelling for traceability |
| Pneumonia | Same as source | Retain |
| Pulmonary neoplasm | Same as source | Retain; do not remove to hide sparse-input failures |
| Spontaneous pneumothorax | Same as source | Retain |
| Tuberculosis | Same as source | Retain |
| URTI | Same as source | Retain |
| Viral pharyngitis | Same as source | Retain |
| Whooping cough | Same as source | Retain |
| External COVID-19 records | COVID-19 | Conditional extension; data approval pending |

Retaining a condition for experiments does not establish that the available
features distinguish it reliably.

Keep source definitions intact. For example, the asthma and COPD source
labels must not silently become claims about all forms of asthma or COPD.

## Allergy label decision

The DDXPlus entry Allergic sinusitis has:

- French name: Rhinite allergique.
- ICD-10 category: J30.
- The same four symptom codes as the added allergic-rhinitis generator.

Together, these support a documented candidate mapping to Allergic rhinitis.
Symptom overlap alone would not justify merging conditions.

The source has 26,203 training and 2,136 validation records before
candidate filtering. These remain simulated DDXPlus records.

Preserve source labels in provenance and apply the mapping consistently
to candidate training and evaluation labels.

Exclude the 400 hand-generated allergy records from the planned candidate
dataset. They are not needed to create a second allergy class.

Do not rewrite raw files, archived reports or existing V2 model classes.

## COVID-19 data decision

COVID-19 remains relevant to the intended application, but the current
400 rule-generated examples do not establish reliable classification.

The Carbon Health/Braid Health repository is a candidate source for audit,
not an approved training source:

https://github.com/mdcollab/covidclinicaldata

Its documentation describes positive and negative test results,
privacy-adjusted ages, exclusion of severe cases and a
CC BY-NC-SA 4.0 licence.

Before using it, verify its version, terms, symptom definitions,
missingness, demographics, repeated records and evaluation suitability.

Do not turn COVID-negative records into other disease labels without
supporting diagnoses.

Do not invent missing sex or other demographic values to satisfy the
current feature schema.

If adequate data cannot be established, proceed with the 21-condition
candidate experiment and document COVID-19 as an unsupported extension.
Do not substitute additional hand-generated rows as evidence of accuracy.

## Additional conditions

Defer additions beyond the scope above during this milestone.

Pulmonary embolism and sarcoidosis occur in the raw source dataset,
but adding them requires a separate review of condition definitions,
feature coverage, training design and evaluation.

Their deferral does not imply that they are unrelated to respiratory
presentations or that the application can exclude them.

Other non-respiratory source labels are not automatically added because
they share symptoms such as tiredness, headache or breathlessness.

## Data and population limitations

Bronchiolitis has only 261 raw training and 28 validation records.
Report its evaluation support explicitly. Generated variants do not
increase the number of independent source patients.

Review age coverage by condition during dataset preparation. Do not
assume that random ages or model use of AGE establish age suitability.

DDXPlus is simulated data. The additional current V2 examples are generated
from hand-written rules. Neither establishes performance in Kenyan users.

Raw disease counts and fitted class counts are not Kenyan prevalence
estimates.

## Evaluation requirements

- Keep the existing 23-class V2 model as the frozen baseline.
- Preserve the original baseline reports and label names.
- Evaluate candidates on consistently mapped ground-truth labels.
- Document how both V2 allergy outputs are handled in any comparison
  using canonical labels.
- Report harmonized comparisons separately from original-label results.
- Report per-condition support and performance, including sparse inputs.
- Evaluate COVID-19 detection only when labelled COVID-19 evaluation
  examples are available.
- Check whether a model learns dataset-specific missingness or collection
  patterns instead of condition-related symptom differences.
- Keep related records and generated variants together when splitting data.
- Do not use final test results to choose the scope or tune the model.

## Inference requirements for later issues

Insufficient information is an inference outcome, not a disease class.

The system must be able to avoid presenting a disease conclusion when
the supported evidence is inadequate.

Safety handling remains separate from disease ranking. Inclusion of serious
conditions does not make the classifier an emergency screening tool.

## Implementation sequence

- Issue #46: record the scope and data limitations.
- Issue #47: implement source provenance, approved label mapping,
  feature derivation and controlled partial-symptom generation.
- Issue #48: train and compare candidates with frozen V2.
- Issue #49: evaluate calibration and insufficient-information handling.
- Issue #50: complete final evaluation and the integration contract.

No backend or frontend changes are included in this scope decision.

## Supporting evidence

- reports/condition_scope_audit/scope_audit.json
- reports/condition_scope_audit/condition_inventory.csv
- docs/CONDITION_EXTENSION_DATA_REVIEW.md
- docs/SYMPTOM_MAPPING_REVIEW.md

ICD-10 reference for the J30 category:
https://classbrowser.nhs.uk/ICD-10-5TH-Edition/vol1/block-j30-j39.htm