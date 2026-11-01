# Portable Candidate Runtime Bundle Specification

Issue: #58

Status: implemented; bundle, wheel and isolated runtime verification passed.
Release evidence: reports/candidate_runtime_bundle/release_manifest.json.
Backend integration and activation remain separate work.

Backend dependency:
RAM-AS-RAP/ucs31-madhaparia-backend#52

## 1. Objective

Export the evaluated respiratory candidate for use outside the ML
repository and its experiment directory.

Preserve the selected model, feature semantics, calibration,
uncertainty decisions and public-response behaviour.

This work does not retrain the model, refit temperature, change policy
thresholds, extend condition scope or activate the backend candidate.

## 2. Frozen reference

The reference implementation is the existing candidate predictor and
its linked assessment, feature and probability helpers.

The exported candidate must preserve:

| Property | Value |
|---|---|
| Candidate | rf_group_weighted |
| Model version | candidate_v1_research |
| Schema ID | respiratory_candidate_v1 |
| Numerical features | 70 |
| Symptom concepts | 34 |
| Classes | 21 |
| Policy ID | candidate_v1_research_uncertainty_v1 |
| Probability representation | temperature_scaled |
| Temperature | 0.6408387217626598 |
| Minimum reporting units | 2 |
| Minimum leading score | 0.78 |
| Minimum top-two margin | 0.20 |

Exact values and class/feature order must come from the verified
artifacts. The export must not reconstruct or approximate model data.

Preserve the original model, schema and policy bytes.

Existing evaluated files and historical reports remain unchanged.

## 3. Distribution design

Deliver two versioned components:

1. An installable Python runtime package.
2. A data bundle containing the frozen model and supporting artifacts.

The Python package will be named respiratory_candidate_runtime.

The data bundle will be supplied through an explicit directory argument.
The runtime must not discover the ML repository by walking parent
directories or depend on the process working directory.

The installed runtime must not import the ML repository's inference,
scripts or nlp packages.

The runtime package must provide inference only. Dataset preparation,
training and temperature fitting are outside its public interface.

Record the tested Python and dependency versions in the release.
Do not assume the ML repository's unpinned requirements are a deployment
environment specification.

## 4. Data bundle contents

The bundle must contain:

- model.joblib: the unchanged selected model.
- feature_schema.json: the unchanged evaluated candidate schema.
- policy.json: the unchanged frozen policy.
- classes.json: ordered labels matching the model and policy.
- provenance/: selected training and evaluation records.
- reference_cases.json: parity inputs and expected outcomes.
- manifest.json: bundle identity, file inventory and integrity metadata.

Provenance must link the release to:

- Candidate training and selection.
- Temperature fitting and policy selection.
- Reserved uncertainty evaluation.
- Final official-test-split evaluation and verification.
- Original inference source fingerprints.

Include only the provenance needed to explain and verify the release.

Raw patient records, prepared dataset CSVs, full prediction archives,
credentials and local absolute paths are not runtime requirements and
must not be copied into the bundle.

Historical provenance may refer to files not shipped in the bundle.
Such references are audit information, not hidden runtime dependencies.

## 5. Manifest and release identity

The manifest must record:

- Bundle format version.
- Bundle ID.
- Model, schema and policy identities.
- Ordered features and classes, or verified references to their files.
- Required runtime-package version.
- Tested dependency versions.
- SHA-256 and byte size for every distributed payload file.
- Source provenance and evaluation status.
- Packaging limitations.

The manifest cannot include a hash of its own final bytes.

Produce a separate release record containing:

- The bundle manifest SHA-256.
- The runtime wheel filename and SHA-256.
- The bundle archive SHA-256 if an archive is distributed.
- Verification and parity results.

The expected manifest digest must be supplied from a trusted release
record or deployment configuration, not taken only from the same
untrusted bundle being checked.

Integrity checks detect changes. They do not make an arbitrary model
file safe to deserialize.

## 6. Export behaviour

The exporter must:

1. Verify the frozen model, schema, policy and linked evaluation records.
2. Confirm that the selected model and policy match the evaluated pair.
3. Record source fingerprints before export.
4. Copy the original artifact bytes.
5. Generate the class list, reference cases and manifest.
6. Verify exported bytes and reference identities.
7. Confirm source fingerprints are unchanged.
8. Mark the export complete only after verification succeeds.

Refuse to overwrite an existing completed bundle.

A failed or interrupted export must not leave an output that can be
mistaken for a completed release.

Generating parity predictions is allowed. Training, calibration fitting,
policy search and final-test metric regeneration are not part of export.

## 7. Runtime loading

The loader must accept:

- An explicit bundle directory.
- An expected manifest SHA-256 from the trusted release record.

Before deserializing the model, it must validate:

- Manifest identity and supported format version.
- Required payload files.
- File sizes and SHA-256 values.
- Safe bundle-relative paths.
- Schema dimensions and feature order.
- Policy references, thresholds and score representation.
- Ordered class-label consistency.
- Runtime compatibility.

Reject absolute payload paths, parent-directory traversal and links
that resolve outside the bundle.

After loading, validate model feature count, feature names when
available, class order and prediction capability.

Use immutable bundle contents during operation. A controlled artifact
update requires a fresh runtime instance or process and revalidation.

No automatic fallback to V1, V2 or another bundle is permitted.

## 8. Prediction interface

The portable predictor accepts:

- A dictionary containing exactly the 70 candidate features.
- Safety flags.
- Unsupported symptom codes.
- An optional research-diagnostics flag, defaulting to false.

Preserve the reference predictor's handling of:

- Demographics and binary indicators.
- Present, absent and unknown states.
- Contradictory symptoms.
- Regional-pain parent requirements.
- Reporting-unit counts.
- Safety-only symptoms.
- Unsupported confirmed symptoms.
- Explicit negative evidence.
- Temperature scaling and score-margin calculations.

Preserve these assessment statuses:

- prediction_available.
- insufficient_information.
- unsupported_input.
- safety_review_required.

Withheld results must have no public condition or score.

Preserve prediction_performed semantics: scoring can occur even when
the resulting prediction is withheld.

The backend will perform API-envelope normalization separately.
The portable runtime must not silently change the ML response contract
to match an HTTP response format.

Bundle and runtime provenance may be exposed separately from the
reference assessment response.

## 9. Error boundaries

Distinguish:

- Invalid caller input.
- Invalid or unavailable bundle resources.
- Unexpected inference failures.

This distinction allows the backend to separate request errors from
service failures.

Do not report an unavailable model as insufficient_information.

Do not expose diagnostic rankings as a fallback after withholding.

## 10. Parity verification

Compare the portable runtime with the frozen reference implementation.

Required cases include:

- Tiredness and confirmed head-region pain.
- Tiredness and confirmed chest-region pain.
- Fever and cough.
- One supported reporting unit.
- No confirmed symptoms.
- Unsupported symptoms only.
- Supported and unsupported symptoms together.
- Safety-only symptoms.
- Supplied safety flags.
- Explicit negative evidence.
- Contradictory states.
- A region without its required parent.

Also test probability assessment around the score and margin thresholds
using controlled probability inputs.

Require exact agreement for:

- Status.
- Leading condition when returned.
- Withholding reasons.
- Reporting-unit count.
- Prediction-performed flag.
- Safety and unsupported-evidence handling.
- Acceptance decisions.

Compare finite numerical scores using a documented absolute tolerance
of 1e-12 and zero relative tolerance.

Small floating-point differences must not be used to excuse a changed
status, class or acceptance decision.

Document exception-type adaptations required by the portable runtime;
invalid cases must remain invalid.

The unlabelled examples are behavioural probes, not diagnostic accuracy
measurements.

## 11. Portability and failure tests

Verify the installed runtime and copied bundle in an isolated environment
outside the ML checkout.

The test must succeed without access to the original experiment files
or imports from the original scripts and inference packages.

Test failures for:

- Missing model, schema, policy or manifest.
- Modified payload bytes.
- An incorrect expected manifest digest.
- Invalid file paths.
- Incompatible schema or class order.
- Unsupported bundle format or runtime compatibility.
- Invalid feature input.

Integrity failures must occur before model deserialization.

Test early withholding without invoking model.predict_proba.

Retain the evaluated research implementation as the reference.
Do not rewrite it to make parity tests pass.

## 12. Storage and delivery

Commit:

- Runtime source and package configuration.
- Export and verification tools.
- Tests.
- Bundle documentation.
- Compact release and verification reports.

Keep generated model bundles, wheels, build directories and temporary
environments out of ordinary Git tracking.

Document how the exact verified release is transferred to the backend.
Do not assume that cloning the ML repository provides the ignored model
artifacts.

The transfer process must verify the trusted release digests before use.

## 13. Limitations

The bundle preserves the evaluated candidate's limitations.

It does not add:

- Clinical validation for Kenyan users.
- COVID-19 as a candidate output class.
- Evaluated explicit negative-evidence support.
- Adaptive follow-up questions.
- Complete NLP or clinical safety assessment.
- Approval to activate the model.

A successful parity check establishes preservation of evaluated
behaviour, not improved clinical performance.

## 14. Completion criteria

Issue #58 is complete when:

- A portable runtime package and verified bundle are produced.
- Source artifacts remain unchanged.
- Unit, integrity, parity and isolation tests pass.
- Installation and transfer instructions are documented.
- Backend #52 can consume the release without the ML experiment folder.