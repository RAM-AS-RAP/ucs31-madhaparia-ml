# Symptom Mapping Audit and Candidate Schema

Related issue: #45

## Purpose

Audit the 37 canonical symptom mappings against DDXPlus definitions
and specify a separate candidate feature representation.

This issue produces an audit, semantic review, proposed schema and
specification tests. It does not retrain or replace V2.

## Structural findings

| Check | Result |
|---|---:|
| Canonical symptoms | 37 |
| Current V2 symptom features | 32 |
| Symptoms with full V2 coverage | 31 |
| Symptoms with partial V2 coverage | 2 |
| Safety-only entries | 3 |
| NLP-only entries | 1 |
| Source-definition issues detected | 0 |
| Conflicting normalized aliases detected | 0 |
| Unknown canonical codes in alias dictionaries | 0 |

Full coverage means that mapped features are included in V2.
It does not establish that an alias accurately represents the
source evidence definition.

## Confirmed representation problem

Headache and chest pain both reduce to E_53 in V2.

Their discarded location features represent forehead pain and
upper-chest pain respectively. Simply restoring those two features
would still assign overly specific locations to broad symptom reports.

## Semantic review

All 37 canonical symptoms were reviewed against their source
definitions.

The review identifies aliases that omit qualifiers such as severity,
persistence, timing or relationships between symptoms.

Examples include:

- Generic tiredness mapped to constant fatigue/non-restful sleep.
- Chest tightness mapped to expiratory wheezing.
- Generic mucus mapped to coloured or unusually abundant sputum.
- Generic weight loss mapped to involuntary loss over three months.
- Improvement with rest mapped to both exertional worsening and
  improvement with rest.

These are source-alignment findings, not clinical validation.

The complete review and proposed decisions are documented in
[SYMPTOM_MAPPING_REVIEW.md](../../docs/SYMPTOM_MAPPING_REVIEW.md).

The generated mapping report retains `semantic_review: pending`
because its automated checks do not perform semantic review.
The separate review document records the human-readable findings.

## Proposed candidate schema

The separate design artifact is
[candidate_feature_schema_v1.json](../../docs/candidate_feature_schema_v1.json).

Its status is `proposed_not_trained`.

It contains:

- The existing 32 binary evidence concepts.
- A proposed head-region pain concept.
- A proposed chest-region pain concept.
- Age and the existing sex encoding.
- Present and absent indicators for each of the 34 symptom concepts.

Total numerical features: 70.

Unknown is represented by both indicators being zero.
Both indicators being one is invalid.

The schema can represent explicit absence, but usable negative
training evidence has not been established.

## Regional pain representation

The proposed head-region grouping includes forehead, temples,
top/back of the head and occiput.

The proposed chest-region grouping includes upper/lower chest,
sides of the chest and posterior chest wall.

Training derivation must use the original location evidence before
partial-symptom masking. Broad user reports must not be converted
into invented exact anatomical locations.

Regional pain requires the general pain feature to be present.
Both regions may be present together.

The groups are proposed source-based aggregations. Their coverage
and predictive usefulness still require evaluation.

## Scope and limitations

- Existing V2 mappings and artifacts remain unchanged.
- No candidate model has been trained.
- No candidate inference encoder is implemented by this issue.
- Specification tests do not establish model accuracy.
- Existing aliases are not approved merely because their evidence
  concepts remain in the candidate schema.
- Duration remains contextual information, outside the numerical schema.
- Safety-only entries remain separate from model features.
- Sneezing remains unsupported by the numerical schema.
- Backend and frontend implementation are outside this issue.

## Reproduction

Run from the ML repository root:

    python -B scripts/audit_symptom_feature_mapping.py
    python -B scripts/build_candidate_feature_schema.py
    python -B -m pytest tests/ -q

Generated artifacts:

- mapping_audit.json
- mapping_audit.csv
- ../../docs/candidate_feature_schema_v1.json

The audit includes a timestamp, so repeated JSON reports need not
be byte-identical. Source fingerprints and substantive findings
should agree when the inputs are unchanged.

Original DDXPlus evidence metadata is required for generation.
The source-location test skips when that untracked metadata is absent.

## Verification

Local test result:

- 109 tests passed.
- 5 subtests passed.
- No tests were skipped in the reported run.

The new tests verify feature ordering, distinct regional mappings,
state definitions, source-location meanings, unsupported-feature
boundaries and the recorded V2 schema fingerprint.

## Follow-up work

- #46: Decide candidate condition coverage.
- #47: Implement and validate candidate data transformations,
  including regional derivation and consistent symptom masking.
- #48: Train and compare candidate models.
- #49: Implement and evaluate candidate inference and uncertainty
  handling, respecting the reviewed source definitions.
- #50: Complete final evaluation and document integration requirements.

The original prediction problems remain open in the milestone.
Issue #45 completes the mapping audit and proposed schema specification.