"""Dense-layer data structures used by dense MLP utilities.

This module intentionally centralizes dense-layer representations so that
architecture recipes and canonicalization do not depend on `align.matching` for
dense-specific helpers.
"""

from dataclasses import dataclass

import numpy as np

_VALID_DEGENERATE_HANDLING = frozenset(
    {"preserve", "zero_outgoing", "canonical_vector"}
)


@dataclass
class DenseLayer:
    """Dense layer parameters.

    Conventions:
    - kernel has shape (in_features, out_features)
    - bias has shape (out_features,)
    """

    kernel: np.ndarray
    bias: np.ndarray


def _validate_degenerate_channels(value: str) -> str:
    if value not in _VALID_DEGENERATE_HANDLING:
        valid = ", ".join(sorted(_VALID_DEGENERATE_HANDLING))
        raise ValueError(f"Unknown degenerate_channels {value!r}. Available: {valid}")
    return value


def _require_bool(name: str, value: bool) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")
    return value


def compute_degenerate_mask(
    layer: DenseLayer, *, epsilon: float = 1e-8, include_bias_in_norm: bool = True
) -> np.ndarray:
    """Return channels whose incoming weight-and-bias energy is degenerate."""

    kernel_sq = np.sum(layer.kernel**2, axis=0)
    bias_sq = layer.bias**2 if include_bias_in_norm else 0.0
    return (kernel_sq + bias_sq) <= epsilon


def _canonicalize_degenerate(
    kernel: np.ndarray, bias: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Replace masked columns with the first basis vector and zero bias."""

    canonical = np.zeros_like(kernel)
    if canonical.shape[0] > 0:
        canonical[0] = np.where(mask, 1.0, 0.0).astype(kernel.dtype)
    return (
        np.where(mask[None, :], canonical, kernel),
        np.where(mask, np.zeros_like(bias), bias),
    )


def compute_incoming_norms(
    layer: DenseLayer,
    epsilon: float = 1e-8,
    include_bias_in_norm: bool = True,
) -> np.ndarray:
    """Compute the per-channel incoming weight-and-bias norm."""

    include_bias_in_norm = _require_bool("include_bias_in_norm", include_bias_in_norm)
    kernel_sq = np.sum(layer.kernel**2, axis=0)
    bias_sq = layer.bias**2 if include_bias_in_norm else 0.0
    return np.sqrt(kernel_sq + bias_sq + epsilon)


def _outgoing_scales(
    norms: np.ndarray, degenerate_mask: np.ndarray, *, degenerate_channels: str
) -> np.ndarray:
    degenerate_channels = _validate_degenerate_channels(degenerate_channels)
    if degenerate_channels in ("zero_outgoing", "canonical_vector"):
        return np.where(degenerate_mask, 0.0, norms)
    return np.where(degenerate_mask, 1.0, norms)


__all__ = ["DenseLayer", "compute_degenerate_mask", "compute_incoming_norms"]
