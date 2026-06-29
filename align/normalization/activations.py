"""Activation validation for positive-homogeneous scale normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUPPORTED_ACTIVATIONS = {
    "relu": "positive_homogeneous",
    "leaky_relu": "positive_homogeneous",
}


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


__all__ = ["SUPPORTED_ACTIVATIONS", "validate_activation"]
