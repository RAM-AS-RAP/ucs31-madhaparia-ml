# COVID-19 and Allergic Rhinitis Data Review

Issue: #46
Status: source review completed; replacement data assessment pending

## Scope decision

Retain COVID-19 and allergic rhinitis in the intended respiratory
application scope for Kenyan users.

Their inclusion in a replacement model remains conditional on suitable
data, feature representation and evaluation. The current V2 model remains
the frozen baseline.

The existing 21 DDXPlus conditions provide the initial comparison scope.
DDXPlus contains simulated patients; its validation results do not establish
clinical performance or representativeness for Kenyan users.

## Current implementation reviewed

- docs/v2_candidate_condition_profiles.json
- scripts/generate_v2_candidate_scenarios.py
- scripts/encode_v2_candidate_scenarios.py
- scripts/assemble_v2_experiment.py
- scripts/train_v2_23class_prototype.py
- models/rf_v2_23class_prototype_manifest.json

The saved manifest records 400 generated training examples for COVID-19
and 400 for allergic rhinitis.

Neither exact label, COVID-19 or Allergic rhinitis, occurs in the inspected
raw DDXPlus training or validation splits.

However, the DDXPlus entry named Allergic sinusitis has the French name
Rhinite allergique, ICD-10 category J30, and the same four symptom codes
as the added allergic-rhinitis generator. Therefore, absence of the exact
English label does not establish absence of allergic-rhinitis coverage.
See the label consistency review below.

## Generation findings

### COVID-19

The generator uses nine symptom features with manually specified sampling
probabilities. Examples include cough at 0.75 and tiredness at 0.65.

These probabilities are design assumptions, not measured patient symptom
frequencies or Kenyan prevalence estimates.

Symptoms are sampled individually before the generator rejects empty
patterns. This does not establish realistic symptom relationships.

### Allergic rhinitis

The generator uses four features:

- E_181: nasal congestion or clear runny nose
- E_169: itchy nose
- E_170: severe itching in one or both eyes
- E_201: cough

It requires at least one of the three nasal or eye features.

Consequently, there are at most 14 allowed symptom combinations:
seven non-empty combinations of the nasal/eye features, each with cough
present or unreported.

The 400 rows therefore do not represent 400 distinct symptom patterns.
Sneezing is recorded as NLP-only and is not encoded for model training.

### Shared limitations

- Ages are randomly sampled from 5 to 85, except the fixed first example.
- Sex is randomly sampled, except the fixed first example.
- Uniqueness is checked using symptoms, age and sex together.
- Both conditions receive the same deliberately ambiguous first example.
- No explicit absent symptoms are generated.
- All generated examples are included in prototype training.
- No independent evaluation of these two conditions is established.
- These generation rules do not establish suitability for Kenyan users.

The existing WHO and NHS references support symptom descriptions.
They do not validate the generator's numerical probabilities or demographic
sampling rules.

## External data status

Older untracked scripts refer to external COVID records:

- scripts/audit_v2_covid_records.py
- scripts/prepare_v2_covid_candidates.py

During this review, the external_covid and external_allergic_rhinitis
folders under the temporary v2_23class_experiment directory contained
no files.

Expected record counts embedded in older scripts are not evidence that
the source records are currently available or have been verified.

Those scripts remain outside the issue #46 commit.

## Requirements before using replacement data

For each proposed source, document:

- Original publisher, download location, version and permitted use.
- Population, country, collection dates and recruitment setting.
- How condition labels were established.
- Symptom definitions, demographics and missing-value meanings.
- Patient identifiers or grouping information for preventing leakage.
- Whether repeated records and duplicate publications are present.
- Whether suitable comparison conditions are included.
- A training and evaluation split that separates related records.

A negative COVID test must not be relabelled as a specific alternative
respiratory condition without supporting labels.

When combining sources, check whether different symptom questionnaires
or missing-data patterns reveal the dataset source and therefore the label.

Missing symptoms must remain unknown unless absence was explicitly recorded.
Missing demographics must not be invented.

## Feature implications

Review sneezing and allergy-related context against available labelled
data before extending the candidate schema.

Adding a feature column alone does not teach the model its relationship
with a condition.

Preserve the symptom-definition qualifications recorded in
SYMPTOM_MAPPING_REVIEW.md.

## Evaluation and release decision

Use the original 21 conditions for comparisons supported by existing
DDXPlus validation labels.

Evaluate the two extensions separately when suitable labelled data becomes
available. Report their effect on predictions for the original conditions.

Additional samples from the same hand-written generator may support
software checks or exploratory experiments, but cannot establish
independent clinical accuracy.

The current V2 baseline has 23 output labels. Following the allergy label
review, the candidate scope is 21 canonical DDXPlus conditions, with
COVID-19 as a conditional 22nd condition. This scope decision does not
approve a replacement model for deployment.


If adequate data cannot be obtained, document the limitation and keep
the extensions experimental.

## Label consistency review

The source entry named Allergic sinusitis contains:

- English name: Allergic sinusitis
- French name: Rhinite allergique
- ICD-10 category: J30
- Symptom codes: E_201, E_181, E_170, E_169

These are exactly the four symptom codes used by the additional
Allergic rhinitis generator.

The raw inventory contains 26,203 training and 2,136 validation records
under Allergic sinusitis.

This combination of metadata strongly suggests that the source label
represents allergic rhinitis despite its inconsistent English name.
Matching symptom codes alone would not establish equivalence.

### Proposed candidate treatment

Resolve this inconsistency before approving separate allergy classes.

The preferred proposal is to map the original source label
Allergic sinusitis to the canonical candidate label Allergic rhinitis,
with the source name preserved in provenance.

Do not automatically combine the 400 hand-generated allergy examples
with the DDXPlus records. Their inclusion requires a separate justification.

This proposal preserves allergy coverage while avoiding an unsupported
distinction between two training labels.

If this mapping is adopted and all other original conditions are retained,
the candidate would have 21 canonical DDXPlus conditions, including
allergic rhinitis. Adding COVID-19 would produce 22 conditions.

The final class count remains subject to the remaining scope review.

### Implementation boundaries

This issue documents the proposed mapping only.

Preserve raw source files, current V2 model classes and frozen baseline
reports. Apply an approved mapping consistently during candidate dataset
preparation and evaluation, with an explicit comparison label mapping.

No effect on model performance has yet been established.

## References

Reviewed on 2026-10-30.

- WHO COVID-19 fact sheet:
  https://www.who.int/news-room/fact-sheets/detail/coronavirus-disease-(covid-19)
- NHS allergic rhinitis overview:
  https://www.nhs.uk/conditions/allergic-rhinitis/