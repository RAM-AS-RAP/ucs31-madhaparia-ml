# Candidate Dataset Preparation Policy

Issue: #47
Status: implemented and verified for the initial 21-condition dataset

## Initial dataset scope

Build the initial candidate dataset from the original 21 DDXPlus
conditions.

Apply the documented candidate mapping:
Allergic sinusitis -> Allergic rhinitis.

Preserve original source labels in provenance.

Do not include the 400 hand-generated allergy examples.
COVID-19 remains conditional on a separate source-data decision.

## Source eligibility

Require valid source evidence and supported demographics.

Derive regional pain concepts from original evidence before masking.

Exclude records with no supported positive concepts from classifier
training. Record their counts and source references in the build audit.

Do not invent symptoms, demographic values or explicit negative evidence.

The source audit found:
- 443,075 in-scope training records.
- 8 training records without supported positive concepts.
- 443,067 provisionally eligible training records.
- 55,808 provisionally eligible validation records.
- No evidence-derivation errors or invalid demographics.

No source records contained both selected pain regions. The representation
allows this combination, but its predictive performance is not established
by the current data.

## Normalized source-input grouping

Define a source-input group using:
- Numeric age, without truncating fractional years.
- Recorded sex.
- Sorted unique original evidence codes.

Exclude pathology and differential diagnosis from the group signature.

These groups identify identical normalized inputs, not verified patient
identity. Different people can have identical inputs, and related records
with different inputs may remain undetected.

The inspected source groups had no conflicting condition labels.
If a future build finds conflicting labels within a source-input group,
stop for review instead of silently choosing a label.

## Repeated source-input handling

Keep one deterministic representative per normalized source-input group
within each split: the first eligible source row.

Record the group size so the effect of removing repeated inputs remains
visible.

Preserve the official training and validation partition assignments.

Give validation precedence for groups shared between the two splits:
exclude every matching training group before generating scenarios.

Never move validation records into training.

Expected counts from the current audit:

- Unique eligible training groups before overlap removal: 421,059.
- Unique eligible validation groups: 55,680.
- Shared source-input groups: 2,836.
- Training groups after overlap removal: 418,223.
- Validation groups retained: 55,680.

These are source-group counts, not generated scenario counts.

This policy changes repeated-input weighting. Report it explicitly when
comparing the candidate with the frozen V2 baseline.

## Candidate-input collisions

Do not deduplicate records merely because their reduced candidate features
match.

Do not delete conflicting disease labels or assign a majority label to
a candidate-input group.

Different full source records may legitimately become indistinguishable
after feature reduction.

The training audit found 4,384 candidate-input groups with multiple
condition labels, covering 79,159 rows before source-group filtering.

Recompute these counts after filtering and report them as a limitation.

Exact candidate-input overlap does not establish patient leakage.
Conversely, eliminating full source-input overlap does not establish
complete independence or clinical generalization.

## Partial-symptom generation

Generate scenarios only after source grouping and split filtering.

Keep every variant of a source group in that group's original split.
Never randomly split generated variants across training and validation.

Use deterministic seeds based on a fixed seed, source-group signature
and variant identifier. Do not use the disease label to select symptoms.

Sample only positive concepts derived from that source record.

Treat a regional pain concept and its required general-pain parent as
one reporting unit. Retaining the region must retain E_53.

Distinguish the number of reporting units from the number of positive
feature columns. One reported regional-pain unit activates two columns.

Allow both pain regions when supported by the source, and preserve their
parent relationship.

Keep hidden or unreported concepts unknown.
Do not convert them to confirmed absence.

Specify scenario counts and sampling proportions before model evaluation.
Record requested and actual reporting-unit counts and remove duplicate
variants within a source group.

## Evaluation and provenance

Preserve the original baseline reports unchanged.

Evaluate frozen V2 and the candidate on clearly documented comparable
source cohorts. Report label harmonization and cohort filtering explicitly.

Record:
- Source file fingerprints and source row numbers.
- Original and canonical labels.
- Source-input group signatures and group sizes.
- Schema and preparation-code fingerprints.
- Exclusion reasons and per-condition counts.
- Split, seed, variant identifier and reporting-unit counts.

Patient grouping cannot be guaranteed because the inspected source
columns do not provide a verified patient identifier.

The final test split remains outside this preparation and tuning process.
Check final-test overlap with training during final evaluation and report
any limitation without using test performance to tune the model.

## Evidence

- reports/candidate_source_audit/source_audit.json
- reports/candidate_source_audit/overlap_audit.json
- docs/CANDIDATE_CONDITION_SCOPE.md

## Initial scenario-generation configuration

Use seed 42.

For each retained training source group:
- Generate one full candidate observation.
- If there are at least two reporting units, also generate one partial
  observation.
- Select the partial reporting-unit count uniformly from 1 through
  min(5, available_units - 1).
- Sample that many reporting units without replacement.
- Preserve required pain parents.
- Assign each variant weight 1 / number_of_variants_for_that_group.

Each source group therefore contributes total training weight 1.
The model-training implementation must consume these weights.

"Full" means all positive candidate concepts available from the source,
not complete clinical information. Other concepts remain unknown.

For validation, initially store one full candidate observation per retained
source group. Issue #48 will evaluate predetermined sparse views retaining
up to 1, 2, 3 and 5 reporting units from the same validation groups.

Sparse validation views are separate evaluation settings, not additional
independent patients. If a group has fewer than the requested units,
retain all available units and record the actual count.

Do not change this configuration in response to final-test performance.