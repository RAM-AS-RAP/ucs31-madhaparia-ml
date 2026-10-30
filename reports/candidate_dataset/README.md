# Candidate Dataset Build and Verification

Issue: #47

## Result

Built and verified an initial 21-condition DDXPlus candidate dataset.

The candidate uses 70 numerical features: age, encoded sex, and
present/absent columns for 34 symptom concepts.

This is dataset preparation, not model training or evidence of improved
prediction accuracy.

## Condition scope

Map the original source label Allergic sinusitis to the candidate label
Allergic rhinitis, preserving the source label in provenance.

Exclude the additional hand-generated allergy records.

COVID-19 remains a conditional extension and is not included in this
initial dataset.

See docs/CANDIDATE_CONDITION_SCOPE.md for the scope decision.

## Retained data

| Split | Source-input groups | Full observations | Partial observations | Total rows |
| --- | ---: | ---: | ---: | ---: |
| Training | 418,223 | 418,223 | 409,332 | 827,555 |
| Validation | 55,680 | 55,680 | 0 | 55,680 |

A source-input group is defined by numeric age, recorded sex and sorted
unique original evidence codes. It is not a verified patient identifier.

Training exclusions:
- 3,780 rows belonging to 2,836 groups reserved for validation.
- 21,064 repeated source-input rows among the remaining training groups.
- 8 rows without supported positive concepts.

Validation exclusions:
- 128 repeated source-input rows.

Retained source-group overlap between training and validation is zero.

## Feature derivation and masking

Derive head-region and chest-region pain from original location evidence
before selecting reported symptoms.

A reported pain region retains its required general-pain parent.
A region and its parent count as one reporting unit.

Each training group receives one full observation and, where possible,
one strictly partial observation containing 1 to 5 reporting units.

Each group's variant weights sum to 1.

Unreported concepts remain unknown. No explicit negative evidence is
invented. The absent-feature columns are therefore all zero.

"Full" refers to all available positive candidate concepts, not complete
clinical information.

## Remaining ambiguity

After source-group filtering:

| Split | Full candidate-input groups with multiple labels | Rows in those groups |
| --- | ---: | ---: |
| Training | 4,383 | 75,719 |
| Validation | 1,300 | 5,853 |

There are 49,654 validation observations whose full candidate input
matches at least one retained training candidate input.

These reduced-feature matches do not establish duplicate patients.
Different original records can become identical after feature reduction.
They remain a limitation of the representation and evaluation.

No source records contained both selected pain regions. Tests verify that
the code can represent that combination; model performance on it is
not established.

Bronchiolitis has 242 retained training groups and 28 validation groups.
Report its limited evaluation support.

## Generated files

Generated data is stored locally in data/candidate_v1/ and ignored by Git:

- train.csv
- validation.csv
- source_groups.csv
- excluded_rows.csv
- manifest.json

The build manifest is copied here as build_manifest.json.
verification.json records the saved-output checks and collision audit.

## Training contract

Select only the columns listed in manifest.schema.feature_names as
model inputs, in exactly that order.

Use canonical_label as the target and source_group_weight as sample weight.

Never use source_group_id, variant_id, canonical_label,
source_group_weight or reported_unit_count as predictive features.

Do not randomly split generated variants. Preserve the prepared source
group separation.

The 70-feature candidate schema is incompatible with the existing
66-feature V2 model.

## Reproduction

Run from the ML repository root with the original source files available:

    python -B scripts/audit_candidate_source_records.py
    python -B scripts/audit_candidate_overlap.py
    python -B scripts/build_candidate_dataset.py
    python -B scripts/verify_candidate_dataset.py
    python -B -m pytest tests/ -q

The builder requires a fresh output directory and refuses to overwrite
an existing directory.

For an additional build, provide a new path with --output-dir.
The verification script currently targets data/candidate_v1/.

Small integration tests verify byte-for-byte reproducibility across
two builds of the same fixture data. The full dataset was built once
and its saved outputs were verified.

## Verification performed

- Input and generated-output fingerprints match.
- Feature order and numerical values satisfy the schema.
- Regional pain retains its parent.
- Partial variants are strict subsets of their full observation.
- Unreported concepts remain unknown.
- Labels match source-group provenance.
- Group weights total 1.
- Source-group IDs do not overlap between retained splits.
- Full validation observations and manifest counts agree.

## Limits and follow-up

DDXPlus is simulated data; Kenyan representativeness is unverified.

Absence columns do not establish learned negative-evidence handling.
Source-input grouping does not detect every possible patient relationship.

No final-test records were inspected. No model was trained or replaced.

Issue #48 will compare trained candidates with frozen V2, including
documented sparse validation views. Issues #49 and #50 address uncertainty
handling and final evaluation.