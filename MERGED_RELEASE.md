# Merged ML/research release

The webapp serves the verified **V3 partial-evidence artifact in the backend**, not a newly trained model inside this research folder. Serving inference does not require the large raw datasets. The original ML ZIP source/model binaries were retained; two recorded schema byte identities were restored through line endings only.

## What was included

- `scripts/evaluate_release_models.py` and tests: matched sparse V1/V3 comparison, real serving-score parity, label-independent profile exclusions and validation-frozen research selection.
- `scripts/restore_research_sources.py` and tests: self-contained, source-checked official DDXPlus restoration and exact-byte identity repair. Evaluation no longer depends on an external old workspace module.
- `reports/release_model_evaluation/`: preserved matched evaluation/policy evidence; the script refuses to overwrite that evidence. Do not retune on the frozen test set to obtain preferred labels.
- Separate `scripts/train_covid_research.py`, tests and `reports/covid_research*`: retained **offline research only**. It is not integrated into patient predictions and must not be blended into V3 or treated as a COVID-vs-flu/TB differential.
- Copied source-readiness/restoration/optional-runtime reports under `reports/` as historical research context.

The actual serving chain is the backend mapper/context/confirmation -> V3 predictor -> naming policy -> stored assessment -> frontend. Copying this ML folder alone does not transfer the observed outputs. Keep the sibling backend and frontend integration.

## Artifact identity

Backend-relative files:

```text
app/ml/artifacts/v3_partial_evidence.json
SHA-256 ae709ef5ddf6dd7a101fe426b610c3e83ef58aa93cb895ca13e7500220dc889c

app/ml/artifacts/research_selection_policy.json
SHA-256 db89087fd44261b158131407c0a67eb3531294cae73b4d1ed587a56aabf77045
```

34 supported features, 21 classes, age-based groups `[1, 5, 12, 18, 40, 65]`, temperature 0.75. Unmentioned evidence stays unknown; explicit negatives are supported. Duration and taste do not alter raw disease scores. COVID is not a trained class. No retraining or decision-threshold tuning was done during the merge.

## Research verification and local inputs

The ML integration branch passed **650 tests plus 5 subtests**, including five new backend-layout regression cases, using the current working backend Python environment. The explicitly scoped ML-only run passed **642 tests plus 5 subtests** with the serving-parity module excluded. No new evaluation/train run was claimed; saved evidence was retained. Raw official files were available locally through ignored hard links to the already-restored research inputs for reproducibility tests. Raw files, secrets and environments are not part of this branch and are unnecessary for app inference.

For research reproduction, create a compatible environment and install the existing research requirements/test dependencies. The serving-parity tests require the **merged** backend and its V3 artifact as a sibling; pushing this ML branch alone does not install that backend. The evaluator accepts both normal repository names (`ucs31-madhaparia-ml` beside `ucs31-madhaparia-backend`) and the older single/double-nested `*-main` archive layout. An incomplete sibling fails explicitly instead of silently selecting another backend copy. Run unit tests from this ML root:

```powershell
python -m pytest tests -q
```

To run the ML-only checks before the merged backend is available, explicitly exclude the serving-parity module (including its backend-layout tests): `python -m pytest tests -q --ignore=tests/test_release_model_evaluation.py`. This is not a full serving-parity pass. Raw source metadata is still required by some original research checks; see restoration below. Never commit that raw data or your Python environment.

Some historical tests require the original official source data. Restore those from the licensed sources rather than substituting truncated fixtures or changing trusted hashes. The pinned source helper performs downloads only when explicitly executed:

```powershell
python scripts/restore_research_sources.py
```

This can download/extract hundreds of MB. Keep DDXPlus/GCCR attribution/licences, do not ship respondent-level data casually, and do not mistake synthetic-source metrics for independent clinical validation. Do not install/activate the old optional runtime merely because its source exists here; its frozen backend bundle/wheel remains unavailable.

Known limitation: tiredness/headache can still yield a lung-tumour research label. Preserving that output is regression parity, not a medically sound inference. The separate COVID survey experiment also remains unsuitable for patient deployment. Full merge evidence/limits are in `../../MERGE_RESULT.md` in the current layout.
