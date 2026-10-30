# Symptom Mapping Semantic Review

Related issue: #45

## Purpose

Review all 37 canonical symptoms against their original DDXPlus
definitions and identify requirements for the candidate representation.

This document specifies proposed changes. Existing V2 mappings,
model artifacts and safety flags remain unchanged.

## Evidence reviewed

- nlp/canonical_symptoms.json
- nlp/english_aliases.json
- nlp/swahili_aliases.json
- data/raw/release_evidences.json
- models/v2_feature_schema.json
- reports/symptom_feature_audit/mapping_audit.json
- reports/v2_baseline_audit/README.md

## Structural audit

- Canonical symptoms: 37
- V2 binary symptom features: 32
- Full schema coverage: 31 symptoms
- Partial schema coverage: 2 symptoms
- Safety-only entries: 3
- NLP-only entry: 1
- Missing/invalid source definitions detected: 0
- Conflicting normalized aliases detected: 0
- Alias entries referring to unknown canonical codes: 0

Full schema coverage does not establish semantic correctness.

Headache and chest pain both reduce to E_53 in V2 because their
location features are excluded.

## Review terminology

- Retain explicit mapping: the named symptom has a corresponding
  source concept. This is not approval of every possible alias.
- Clarify wording: some aliases omit a required qualifier or express
  a different concept. Preserve the report and request confirmation
  rather than automatically asserting the more specific evidence.
- Redesign location mapping: preserve the reported region without
  inventing an exact anatomical location.
- Safety-only: preserve the separate safety path.
- Unsupported by current schema: preserve recognised text without
  inventing a trained feature.

## Symptom-by-symptom decisions

| Canonical code | Proposed decision | Reason or required clarification |
|---|---|---|
| fever | Clarify wording | "Temperature" alone does not assert fever. Keep explicit fever descriptions; review broad aliases. |
| chills | Clarify wording | Preserve chills/shivering. Review context-sensitive aliases such as "baridi" rather than assuming every use denotes chills. |
| cough | Retain explicit mapping | E_201 represents cough. Duration implied by "persistent" is separate information. |
| intense_coughing_fits | Retain explicit mapping | E_203 represents intense coughing fits. |
| vomiting_after_coughing | Retain explicit mapping | E_166 requires vomiting after coughing; preserve that relationship. |
| productive_cough | Clarify wording | E_77 requires coloured or more abundant sputum than usual. "Mucus" or "wet cough" alone does not establish those qualifiers. |
| coughing_blood | Retain explicit mapping | E_45 represents coughing up blood. Preserve independent safety handling. |
| shortness_of_breath | Clarify wording | E_66 specifies significant breathing difficulty. Preserve the reported symptom and safety review while clarifying severity for model encoding. |
| night_breathlessness | Clarify wording | E_67 specifies episodes that wake the patient. Breathlessness occurring at night does not necessarily establish waking from sleep. |
| wheezing | Clarify wording | E_214 specifies an expiratory wheezing sound. Do not automatically equate chest tightness with wheezing. |
| wheeze_after_coughing | Clarify wording | E_112 covers inspiratory wheeze or noisy breathing after coughing spells. Preserve the distinction from E_214. |
| stridor | Clarify wording | E_194 specifies a high-pitched sound during inhalation. Generic high-pitched breathing wording needs clarification; retain safety review. |
| drooling | Clarify wording | E_190 concerns increased saliva production. Review whether each drooling alias establishes the source concept; preserve existing safety handling. |
| difficulty_swallowing | Retain explicit mapping | E_65 covers difficulty or discomfort/blockage during swallowing. |
| sore_throat | Retain explicit mapping | E_97 represents sore throat. Review translated aliases for equivalent meaning before vocabulary changes. |
| hoarse_voice | Clarify wording | E_212 specifies a deeper, softer or hoarse voice. Generic "changed voice" is broader. |
| runny_blocked_nose | Clarify wording | E_181 covers nasal congestion or a clear runny nose. Preserve these alternatives without assuming discharge characteristics or inferring symptoms from a disease name. |
| nasal_discharge | Retain explicit mapping | E_182 represents greenish or yellowish nasal discharge. |
| loss_of_smell | Retain explicit mapping | E_103 represents loss of smell. |
| itchy_nose_throat | Retain explicit mapping | E_169 represents an itchy nose or back of the throat. |
| itchy_eyes | Clarify wording | E_170 specifies severe itching. Generic itchy eyes does not establish severity. |
| headache | Redesign location mapping | Generic headache currently asserts forehead pain, then V2 drops that location. Preserve head-region information without inventing forehead location. |
| chest_pain | Redesign location mapping | Generic chest pain currently asserts upper-chest pain, then V2 drops that location. Preserve chest-region information without inventing a precise location. |
| chest_pain_at_rest | Retain explicit mapping | E_14 represents chest pain even at rest. Preserve independent safety handling. |
| pain_deep_breath | Clarify wording | E_220 specifies pain increased by deep inspiration. Preserve the aggravating relationship. |
| worse_lying_down | Clarify wording | E_217 requires worsening when lying down and improvement when sitting up. One half alone does not establish both. |
| worse_with_exertion | Clarify wording | E_218 requires worsening with exertion and improvement with rest. One half alone does not establish both. |
| tiredness | Clarify wording | E_89 represents constant fatigue or non-restful sleep. Generic tiredness does not establish either qualifier. |
| activity_fatigue | Clarify wording | E_88 concerns inability to perform usual activities or being confined to bed by fatigue. Being tired while performing activities is not necessarily equivalent. |
| muscle_aches | Clarify wording | E_144 specifies diffuse muscle pain. Generic muscle pain may be localised. |
| heavy_sweating | Clarify wording | E_50 represents significantly increased sweating. Timing alone, such as night sweating, does not establish the increase. |
| loss_of_appetite | Clarify wording | E_161 describes recent appetite loss or earlier-than-usual fullness. "Not hungry" at one moment is insufficient. |
| weight_loss | Clarify wording | E_162 specifies involuntary weight loss over three months. Generic weight loss does not establish intent or time period. |
| blue_lips | Safety-only | No current model feature. Preserve the separate safety path. |
| confusion | Safety-only | No current model feature. Preserve the separate safety path. |
| fainting | Safety-only | No current model feature. Preserve the separate safety path. |
| sneezing | Unsupported by current schema | Recognise and retain the report, but do not invent a DDXPlus feature or substitute wheezing. |

## Candidate representation requirements

1. Preserve the difference between a user's broad symptom report and
   a more specific DDXPlus evidence definition.

2. Do not encode an unreported qualifier as confirmed present.

3. Keep present, explicitly absent and unknown states distinct.
   Unknown must not be silently interpreted as absent.

4. Distinguish head-region pain from chest-region pain. Any proposed
   grouping of source location values must be explicitly documented
   and applied consistently during training and inference.

5. Do not assign all possible anatomical locations when the patient
   reports only a broad region.

6. Keep duration as reported information unless an appropriate,
   documented training feature supports its use.

7. Keep safety handling independent of model feature availability.
   Missing model qualifiers must not silently discard safety concerns.

8. Retain unsupported and ambiguous reports for review. Do not
   convert them into invented model evidence.

9. Define a separate versioned candidate schema. The saved V2 model
   must continue to use its original schema.

10. Evaluate whether the source data can support each proposed feature
    before implementing candidate training.

## NLP observations carried forward from issue #44

The tested mapper correctly identified negation and symptom duration,
but still returned wording such as "no" and "for three weeks" as
unmatched text.

Future cleanup should distinguish already-interpreted context from
genuinely unsupported wording. This must preserve negation, duration
and patient-context information.

## Limits of this review

This is a source-definition and software representation review.
It is not independent clinical validation.

English and Kiswahili aliases require contextual testing. Ambiguous
translations should receive language review before being accepted
as equivalent medical expressions.

Changing aliases or feature definitions may require later changes
to symptom confirmation. Those integration requirements will be
documented; backend and frontend implementation is outside issue #45.

## Next step

Specify a candidate feature schema and validation checks that preserve
supported symptom distinctions without inventing information.

Training-data construction is tracked in #47. Model retraining and
comparison are tracked in #48.