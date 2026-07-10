"""Activation validation for positive-homogeneous scale normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUPPORTED_ACTIVATIONS = {
    "relu": "positive_homogeneous",
    "leaky_relu": "positive_homogeneous",
    # TLU(y) = max(y, tau) is jointly positively homogeneous in (y, tau); the
    # threshold is a bound channel parameter scaled with its group, so
    # FRN/TLU stacks admit the exact post-norm affine scale symmetry.
    "tlu": "positive_homogeneous",
    # GELU admits no hidden-unit scale symmetry; it is accepted so that
    # architectures whose scale symmetries are activation-independent
    # (attention circuits) can declare their true activation, and rejected by
    # plans that require positive homogeneity (dense chains).
    "gelu": "non_homogeneous",
}


def is_positive_homogeneous(name: str) -> bool:
    """Whether the (validated) activation admits the ReLU scale symmetry."""

    return SUPPORTED_ACTIVATIONS[name] == "positive_homogeneous"


def validate_activation(
    name: str,
    activation_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Validate and normalize an activation name for scale normalization."""

    key = str(name).lower()
    if key not in SUPPORTED_ACTIVATIONS:
        available = ", ".join(sorted(SUPPORTED_ACTIVATIONS))
        raise ValueError(f"Unknown activation '{name}'. Available: {available}")
    if activation_kwargs:
        raise ValueError(
            "scale_normalize.activation_kwargs is currently unsupported; "
            f"{key} uses the shared positive-homogeneous norm formula."
        )
    return key


__all__ = ["SUPPORTED_ACTIVATIONS", "is_positive_homogeneous", "validate_activation"]
