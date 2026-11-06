# Older optional candidate — external artifact dependency

This is **not** the active V3 model or the new GCCR002 research experiment. Its original generated artifacts were excluded from the downloaded source archives and repository. The current application does not activate or depend on it.

The retained legacy backend tests expect:

- `app/ml/artifacts/candidate_v1/bundle/`, matching `docs/candidate_runtime_release/bundle_manifest.json`.
- Original `model.joblib`: **36,327,661 bytes**, SHA-256 `7edc3dc42570c79c67bf14f6f8c7d651d0f61324161e3836341d20f49a6f3759`.
- Trusted bundle-manifest SHA-256 `52ee494aedd887210a0c4ca991d94321321af0e3766ee7b63f0742867730858b` and all manifest-listed class/schema/policy/provenance/reference-case files.
- Original `respiratory_candidate_runtime-0.1.0-py3-none-any.whl`: **15,323 bytes**, SHA-256 `08d31afb8934fe8ed37a573a7af032fd4f8156753f274e9cf17df7b48371d05e`.
- Recorded runtime environment: Python **3.11.9**, NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, scikit-learn 1.9.1, joblib 1.6.0. The current application uses a different environment; do not claim identity by bypassing those checks.

To complete that optional path, obtain the original bundle/wheel from the project artifact owner, verify every size/hash, then test it in an isolated matching environment. Do not overwrite the active application environment, substitute V1/V2/V3 models, regenerate a different model and reuse the old identity, edit golden hashes or disable runtime guards.

The separate source-JSON CRLF restoration is legitimate because it reproduces existing recorded hashes exactly with identical JSON values. That cannot recover an absent trained-model binary or make an arbitrary rebuilt wheel match the original artifact.

Supplying source code or synthetic training data is not the same as supplying this frozen artifact release. The selected V3 application can remain a tested research-awareness demo without shipping this older optional model; full legacy-candidate tests remain explicitly blocked until these inputs are available.

## Recovery recheck — 5 November 2026

- Searched the workspace for wheel/joblib files and inspected the artifact entries in all three supplied component ZIPs. They contain the original V1/V2 models, not the recorded candidate model or runtime wheel.
- Inspected complete, non-truncated public Git trees for the supplied `Eeshan-Vaghjiani/codex-version` repository at `main` (`c04aca7ae5551af72046ef2ee3b852013f100c18`), `codex-respiracare-improvements-2026-10-29` (`3186a6d7d55ca60c7aff465fb0c7ba9abfb9c5fa`) and `muse/respiracare-improvements-2026-10-29` (`24612e9cf3e56f6bb49362aa6eb68a8d08c98f7b`). No matching candidate bundle/wheel paths or model-sized blobs were present.
- Public [releases](https://api.github.com/repos/Eeshan-Vaghjiani/codex-version/releases) returned an empty list; [workflow artifacts](https://api.github.com/repos/Eeshan-Vaghjiani/codex-version/actions/artifacts) reported zero artifacts. This does not claim that the owner has no private/offline copy.
- No package was installed, environment replaced, model substituted, test skipped or trusted digest edited. The **1 failure and 76 setup errors remain unresolved**, rather than being relabelled as passing tests.

To unblock this path, the artifact owner must provide the exact separately delivered release described above. Rebuilding another model from source is a new candidate, not recovery of this frozen release.
