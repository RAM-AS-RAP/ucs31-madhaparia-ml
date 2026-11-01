"""Load the candidate only after bundle and metadata verification."""

import copy
from importlib import metadata
import io
import platform
import warnings

import joblib
from sklearn.exceptions import InconsistentVersionWarning

from . import __version__
from .errors import BundleValidationError
from .integrity import decode_json_bytes, read_verified_bundle
from .metadata import validate_policy, validate_schema
from .predictor import _predict_with_resources


RUNTIME_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
)


def current_environment():
    """Return versions required for this release's compatibility check."""

    return {
        "python": platform.python_version(),
        "packages": {
            name: metadata.version(name)
            for name in RUNTIME_DEPENDENCIES
        },
    }


class _CandidateRuntime:
    """Loaded resources; construct through load_runtime."""

    def __init__(self, model, schema, policy, classes, provenance):
        self._model = model
        self._schema = copy.deepcopy(schema)
        self._policy = copy.deepcopy(policy)
        self._classes = tuple(classes)
        self._provenance = copy.deepcopy(provenance)

    @property
    def schema(self):
        return copy.deepcopy(self._schema)

    @property
    def classes(self):
        return self._classes

    @property
    def provenance(self):
        return copy.deepcopy(self._provenance)

    def predict(
        self,
        feature_values,
        *,
        safety_flags=(),
        unsupported_symptoms=(),
        include_diagnostics=False,
    ):
        return _predict_with_resources(
            feature_values,
            model=self._model,
            schema=self._schema,
            policy=self._policy,
            safety_flags=safety_flags,
            unsupported_symptoms=unsupported_symptoms,
            include_diagnostics=include_diagnostics,
        )


def _validate_model(model, schema, classes):
    if getattr(model, "n_features_in_", None) != 70:
        raise BundleValidationError(
            "Loaded candidate must expect 70 features."
        )

    if list(getattr(model, "classes_", [])) != classes:
        raise BundleValidationError(
            "Loaded model class order differs from the bundle."
        )

    if (
        hasattr(model, "feature_names_in_")
        and list(model.feature_names_in_) != schema["feature_names"]
    ):
        raise BundleValidationError(
            "Loaded model feature order differs from the schema."
        )

    if not callable(getattr(model, "predict_proba", None)):
        raise BundleValidationError(
            "Loaded model does not support probability prediction."
        )


def load_runtime(bundle_directory, *, expected_manifest_sha256):
    """Verify a trusted release and return its loaded prediction runtime.

    Uses verified model bytes rather than reopening the model path.
    Loading does not perform predictions.
    """

    manifest, payloads = read_verified_bundle(
        bundle_directory,
        expected_manifest_sha256,
    )

    try:
        if manifest.get("environment") != current_environment():
            raise BundleValidationError(
                "Python or dependency versions differ from the "
                "verified release environment."
            )

        schema = decode_json_bytes(
            payloads["feature_schema.json"],
            "feature_schema.json",
        )
        policy = decode_json_bytes(
            payloads["policy.json"],
            "policy.json",
        )
        classes = decode_json_bytes(
            payloads["classes.json"],
            "classes.json",
        )

        validate_schema(schema)

        files = manifest["files"]

        validate_policy(
            policy,
            schema,
            classes,
            model_sha256=files["model.joblib"]["sha256"],
            schema_sha256=files["feature_schema.json"]["sha256"],
        )

        expected_identities = {
            "model_version": "candidate_v1_research",
            "schema_id": schema["schema_id"],
            "policy_id": policy["policy_id"],
        }

        for name, value in expected_identities.items():
            if manifest.get(name) != value:
                raise BundleValidationError(
                    f"Manifest identity mismatch: {name}"
                )

        # All payload fingerprints, environment requirements and
        # relevant metadata have been checked before this point.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                InconsistentVersionWarning,
            )
            model = joblib.load(
                io.BytesIO(payloads["model.joblib"])
            )

        _validate_model(model, schema, classes)

        provenance = {
            "bundle_id": manifest["bundle_id"],
            "bundle_format_version": manifest["bundle_format_version"],
            "manifest_sha256": expected_manifest_sha256,
            "runtime_version": __version__,
            **expected_identities,
            "model_sha256": files["model.joblib"]["sha256"],
            "schema_sha256": files["feature_schema.json"]["sha256"],
            "policy_sha256": files["policy.json"]["sha256"],
            "environment": copy.deepcopy(manifest["environment"]),
        }

        return _CandidateRuntime(
            model,
            schema,
            policy,
            classes,
            provenance,
        )

    except BundleValidationError:
        raise
    except Exception as error:
        raise BundleValidationError(
            "Unable to load the verified candidate resources."
        ) from error