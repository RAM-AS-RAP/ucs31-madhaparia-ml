# Verified portable respiratory candidate

Issue: #58

Runtime version: 0.1.0

Status: bundle, wheel and isolated installed-runtime verification passed.
Backend integration and activation remain separate work.

## Release contents

The release consists of:

- The respiratory_candidate_runtime Python wheel.
- A separate bundle directory containing the model, feature schema,
  frozen policy, classes, reference cases and provenance.
- Trusted release and verification records committed in this folder.

Generated artifacts are stored locally under:

data/candidate_experiment_v1/portable_runtime_v1/

The wheel is in its wheels subfolder.
The model artifacts are in its bundle subfolder.

These generated artifacts are ignored by Git. Cloning this repository
does not provide the model bundle or wheel.

## Trusted identities

Bundle manifest SHA-256:

52ee494aedd887210a0c4ca991d94321321af0e3766ee7b63f0742867730858b

Wheel filename:

respiratory_candidate_runtime-0.1.0-py3-none-any.whl

Wheel SHA-256:

08d31afb8934fe8ed37a573a7af032fd4f8156753f274e9cf17df7b48371d05e

Use release_manifest.json from the reviewed repository as the trusted
release record. Do not derive trust solely from files supplied alongside
an unknown model bundle.

## Verified environment

Python: 3.11.9

Direct dependencies:

- numpy 2.4.6
- pandas 3.0.6
- scipy 1.17.1
- scikit-learn 1.9.1
- joblib 1.6.0

runtime_constraints.txt also records the transitive runtime dependency
versions installed during the successful isolated check.

The runtime currently requires an exact match for Python and its five
direct dependency versions.

## Installation and transfer

For backend issue #52:

1. Transfer the verified wheel and entire bundle directory through a
   trusted local or artifact-delivery process.
2. Obtain release_manifest.json and runtime_constraints.txt from the
   reviewed ML repository.
3. Check the wheel SHA-256 against the trusted release record before
   installation.
4. Install into an environment using Python 3.11.9:

   python -m pip install --only-binary=:all: --constraint runtime_constraints.txt respiratory_candidate_runtime-0.1.0-py3-none-any.whl

5. Run python -m pip check.
6. Supply the bundle directory explicitly when loading the runtime.
7. Supply the expected manifest digest from trusted configuration.

Do not copy the ML experiment directory, training scripts or raw datasets.
Do not replace or activate the backend model as part of this release.

## Python interface

```python
from respiratory_candidate_runtime.loader import load_runtime

runtime = load_runtime(
    bundle_directory,
    expected_manifest_sha256=trusted_manifest_digest,
)

result = runtime.predict(
    feature_values,
    safety_flags=[],
    unsupported_symptoms=[],
)
```

feature_values must contain exactly the 70 schema features.
The backend must map confirmed symptoms into those features.

Research diagnostics are disabled by default.
The backend must preserve withheld results without exposing diagnostic
rankings as replacement predictions.

Errors are separated into CandidateInputError, BundleValidationError
and CandidateInferenceError.

## Verification evidence

- All 10 installed runtime modules matched the verified wheel.
- The wheel modules matched the source fingerprints captured at export.
- The frozen reference and portable runtime matched 13 saved cases.
- The separately installed runtime passed 26 public/diagnostic calls.
- Invalid-input errors, safety handling and withholding were preserved.
- Tests used isolated Python mode from a directory outside the ML repository.
- Imports from scripts, inference and nlp were blocked.
- Bundle and wheel fingerprints remained unchanged.
- pip check reported no broken requirements.

Floating-point score comparisons used absolute tolerance 1e-12 and
relative tolerance zero. Statuses, conditions and decisions were exact.

The earlier export and wheel records retain their historical pending
statuses. release_manifest.json links the subsequent successful checks.

## Reproduction

The packaging workflow is:

1. Run the runtime tests.
2. Run export_candidate_runtime_bundle.py --check-only.
3. Run export_candidate_runtime_bundle.py.
4. Build the wheel using pip wheel --no-deps.
5. Run verify_candidate_runtime_wheel.py.
6. Install the wheel in a fresh external virtual environment.
7. Copy the bundle and verification inputs outside the repository.
8. Run verify_installed_candidate_runtime.py using Python -I -B and
   independently supplied expected manifest and wheel digests.

Export tools refuse to overwrite existing completed outputs.
A rebuilt wheel can have a different digest and requires fresh verification.
Do not replace artifacts while continuing to use this release's digests.

## Limitations

This release preserves the evaluated 21-condition research candidate.
It does not establish clinical validation for Kenyan users.

Scores are not clinically validated individual disease risks.
Historical exposure to the official test split cannot be ruled out.

The release does not add COVID-19, evaluated explicit negative-evidence
support, adaptive follow-up questions, NLP or a complete safety assessment.

Successful portability verification does not approve deployment.