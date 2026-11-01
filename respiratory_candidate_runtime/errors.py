"""Distinct errors for callers and runtime resources."""


class CandidateInputError(ValueError):
    """The supplied observation or context is invalid."""


class BundleValidationError(RuntimeError):
    """Required runtime artifacts are unavailable or incompatible."""


class CandidateInferenceError(RuntimeError):
    """Inference failed after input and resource validation."""