"""Scale-normalization math kernel for dense ReLU-style MLPs."""

import jax
import jax.numpy as jnp

from .dense import DenseLayer

_VALID_DEGENERATE_HANDLING = frozenset(
    {"preserve", "zero_outgoing", "canonical_vector"}
)


def _validate_degenerate_handling(value: str) -> str:
    if value not in _VALID_DEGENERATE_HANDLING:
        valid = ", ".join(sorted(_VALID_DEGENERATE_HANDLING))
        raise ValueError(f"Unknown degenerate_handling '{value}'. Available: {valid}")
    return value


def _require_bool(name: str, value: bool) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")
    return value


def compute_degenerate_mask(
    layer: DenseLayer, *, epsilon: float = 1e-8, normalize_biases: bool = True
) -> jnp.ndarray:
    """Boolean mask of neurons whose incoming ``(w, b)`` energy is ``<= epsilon``."""
    kernel_sq = jnp.sum(layer.kernel**2, axis=0)
    bias_sq = layer.bias**2 if normalize_biases else 0.0
    return (kernel_sq + bias_sq) <= epsilon


def _canonicalize_degenerate(
    kernel: jax.Array, bias: jax.Array, mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Replace masked columns with the canonical basis vector ``e_0`` and zero bias."""
    canonical = jnp.zeros_like(kernel)
    if canonical.shape[0] > 0:
        canonical = canonical.at[0].set(jnp.where(mask, 1.0, 0.0).astype(kernel.dtype))
    return (
        jnp.where(mask[None, :], canonical, kernel),
        jnp.where(mask, jnp.zeros_like(bias), bias),
    )


def compute_incoming_norms(
    layer: DenseLayer, epsilon: float = 1e-8, normalize_biases: bool = True
) -> jnp.ndarray:
    """Compute per-neuron ||(w, b)||_2 for a dense layer."""
    normalize_biases = _require_bool("normalize_biases", normalize_biases)
    kernel_sq = jnp.sum(layer.kernel**2, axis=0)
    bias_sq = layer.bias**2 if normalize_biases else 0.0
    return jnp.sqrt(kernel_sq + bias_sq + epsilon)


def _outgoing_scale_factors(
    norms: jnp.ndarray, degenerate_mask: jnp.ndarray, *, degenerate_handling: str
) -> jnp.ndarray:
    degenerate_handling = _validate_degenerate_handling(degenerate_handling)
    if degenerate_handling in ("zero_outgoing", "canonical_vector"):
        return jnp.where(degenerate_mask, 0.0, norms)
    return jnp.where(degenerate_mask, 1.0, norms)


def normalize_last_layer_classification(
    layer: DenseLayer, num_classes: int, epsilon: float = 1e-8
) -> tuple[DenseLayer, float]:
    """Rescale the final layer to lie on a sphere of radius sqrt(C)."""
    gamma = jnp.sqrt(jnp.sum(layer.kernel**2) + jnp.sum(layer.bias**2) + epsilon)
    scale = jnp.sqrt(num_classes) / gamma
    return DenseLayer(kernel=layer.kernel * scale, bias=layer.bias * scale), gamma


__all__ = [
    "compute_degenerate_mask",
    "compute_incoming_norms",
    "normalize_last_layer_classification",
]
