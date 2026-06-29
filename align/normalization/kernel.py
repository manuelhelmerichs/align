"""Scale-normalization math kernel for dense ReLU-style MLPs."""

import functools
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from .activations import validate_activation
from .dense_layers import DenseLayer

_VALID_DEGENERATE_HANDLING = frozenset(
    {"preserve", "zero_outgoing", "canonical_vector"}
)


@functools.partial(jax.jit, static_argnums=(2, 3))
def _compute_incoming_norms_jit(
    kernel: jax.Array, bias: jax.Array, epsilon: float, normalize_biases: bool
) -> jax.Array:
    """JIT-compiled norm computation for a single layer.

    static_argnums: epsilon (2), normalize_biases (3) are compile-time constants.
    """
    kernel_sq = jnp.sum(kernel**2, axis=0)
    bias_sq = bias**2 if normalize_biases else 0.0
    return jnp.sqrt(kernel_sq + bias_sq + epsilon)


def _validate_degenerate_handling(value: str) -> str:
    if value not in _VALID_DEGENERATE_HANDLING:
        valid = ", ".join(sorted(_VALID_DEGENERATE_HANDLING))
        raise ValueError(f"Unknown degenerate_handling '{value}'. Available: {valid}")
    return value


def _require_bool(name: str, value: bool) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")
    return value


@functools.partial(jax.jit, static_argnums=(3, 5))
def _normalize_hidden_layer_jit(
    kernel: jax.Array,
    bias: jax.Array,
    norms: jax.Array,
    degenerate_handling: str,
    degenerate_mask: jax.Array,
    epsilon: float,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled hidden layer normalization.

    static_argnums: degenerate_handling (3), epsilon (5).
    """
    safe_norms = jnp.where(degenerate_mask, 1.0, norms)
    normalized_kernel = kernel / safe_norms[None, :]
    normalized_bias = bias / safe_norms

    if degenerate_handling == "canonical_vector":
        canonical = jnp.zeros_like(normalized_kernel)
        if canonical.shape[0] > 0:
            mask_values = jnp.where(degenerate_mask, 1.0, 0.0)
            canonical = canonical.at[0].set(mask_values)
        normalized_kernel = jnp.where(
            degenerate_mask[None, :], canonical, normalized_kernel
        )
        normalized_bias = jnp.where(
            degenerate_mask, jnp.zeros_like(normalized_bias), normalized_bias
        )

    return normalized_kernel, normalized_bias


@functools.partial(jax.jit, static_argnums=(2, 3))
def _normalize_last_layer_classification_jit(
    kernel: jax.Array, bias: jax.Array, num_classes: int, epsilon: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JIT-compiled optional classification-head rescaling.

    static_argnums: num_classes (2), epsilon (3).
    """
    gamma = jnp.sqrt(jnp.sum(kernel**2) + jnp.sum(bias**2) + epsilon)
    scale = jnp.sqrt(num_classes) / gamma
    return kernel * scale, bias * scale, gamma


def compute_incoming_norms(
    layer: DenseLayer, epsilon: float = 1e-8, normalize_biases: bool = True
) -> jnp.ndarray:
    """Compute per-neuron ||(w, b)||_2 for a dense layer.

    Uses JIT-compiled implementation for performance.
    """
    normalize_biases = _require_bool("normalize_biases", normalize_biases)
    return _compute_incoming_norms_jit(
        layer.kernel, layer.bias, epsilon, normalize_biases
    )


def _outgoing_scale_factors(
    norms: jnp.ndarray, degenerate_mask: jnp.ndarray, *, degenerate_handling: str
) -> jnp.ndarray:
    degenerate_handling = _validate_degenerate_handling(degenerate_handling)
    if degenerate_handling in ("zero_outgoing", "canonical_vector"):
        return jnp.where(degenerate_mask, 0.0, norms)
    return jnp.where(degenerate_mask, 1.0, norms)


def normalize_hidden_layer(
    layer: DenseLayer,
    norms: jnp.ndarray,
    *,
    epsilon: float = 1e-8,
    degenerate_handling: str = "preserve",
    degenerate_mask: jnp.ndarray | None = None,
    normalize_biases: bool = True,
) -> DenseLayer:
    """Normalize incoming weights and bias for a hidden layer.

    Uses JIT-compiled implementation for performance.
    """
    degenerate_handling = _validate_degenerate_handling(degenerate_handling)
    normalize_biases = _require_bool("normalize_biases", normalize_biases)
    if degenerate_mask is None:
        kernel_sq = jnp.sum(layer.kernel**2, axis=0)
        bias_sq = layer.bias**2 if normalize_biases else 0.0
        raw_norm_sq = kernel_sq + bias_sq
        degenerate_mask = raw_norm_sq <= epsilon

    kernel, bias = _normalize_hidden_layer_jit(
        layer.kernel,
        layer.bias,
        norms,
        degenerate_handling,
        degenerate_mask,
        epsilon,
    )
    return DenseLayer(kernel=kernel, bias=bias)


def normalize_last_layer_classification(
    layer: DenseLayer, num_classes: int, epsilon: float = 1e-8
) -> tuple[DenseLayer, float]:
    """Rescale the final layer to lie on a sphere of radius sqrt(C).

    Uses JIT-compiled implementation for performance.
    """
    kernel, bias, gamma = _normalize_last_layer_classification_jit(
        layer.kernel, layer.bias, num_classes, epsilon
    )
    return DenseLayer(kernel=kernel, bias=bias), gamma


def normalize_layers(
    layers: Sequence[DenseLayer],
    *,
    task_type: str = "regression",
    num_classes: int | None = None,
    epsilon: float = 1e-8,
    degenerate_handling: str = "preserve",
    normalize_biases: bool = True,
    activation: str = "relu",
    activation_kwargs: dict[str, Any] | None = None,
    classification_head_rescale: bool = False,
) -> tuple[list[DenseLayer], list[jnp.ndarray], dict[str, Any]]:
    """Apply scale normalization to a sequence of dense layers."""

    if len(layers) < 2:
        raise ValueError("Need at least 2 layers (1 hidden + 1 output)")
    degenerate_handling = _validate_degenerate_handling(degenerate_handling)
    normalize_biases = _require_bool("normalize_biases", normalize_biases)
    classification_head_rescale = _require_bool(
        "classification_head_rescale", classification_head_rescale
    )

    task = (task_type or "regression").lower()
    if task not in ("regression", "classification"):
        raise ValueError("task_type must be 'regression' or 'classification'")
    if task == "classification" and classification_head_rescale and num_classes is None:
        raise ValueError(
            "num_classes required when classification_head_rescale is enabled"
        )

    activation = validate_activation(activation, activation_kwargs)

    normalized_layers: list[DenseLayer] = []
    scale_factors: list[jnp.ndarray] = []

    outgoing_scale: jnp.ndarray | None = None
    for idx, layer in enumerate(layers):
        kernel = layer.kernel
        bias = layer.bias

        if idx > 0 and outgoing_scale is not None:
            kernel = outgoing_scale[:, None] * kernel

        if idx < len(layers) - 1:
            working = DenseLayer(kernel=kernel, bias=bias)
            norms = compute_incoming_norms(
                working, epsilon=epsilon, normalize_biases=normalize_biases
            )
            raw_norm_sq = jnp.sum(working.kernel**2, axis=0)
            if normalize_biases:
                raw_norm_sq = raw_norm_sq + working.bias**2
            degenerate_mask = raw_norm_sq <= epsilon
            scale_factors.append(norms)
            normalized_layer = normalize_hidden_layer(
                working,
                norms,
                epsilon=epsilon,
                degenerate_handling=degenerate_handling,
                degenerate_mask=degenerate_mask,
                normalize_biases=normalize_biases,
            )
            normalized_layers.append(normalized_layer)
            outgoing_scale = _outgoing_scale_factors(
                norms, degenerate_mask, degenerate_handling=degenerate_handling
            )
        else:
            normalized_layers.append(DenseLayer(kernel=kernel, bias=bias))

    aux: dict[str, Any] = {
        "task_type": task,
        "epsilon": epsilon,
        "degenerate_handling": degenerate_handling,
        "normalize_biases": normalize_biases,
        "activation": activation,
        "classification_head_rescale": classification_head_rescale,
    }
    if task == "classification" and classification_head_rescale:
        normalized_output, gamma = normalize_last_layer_classification(
            normalized_layers[-1],
            int(num_classes),
            epsilon=epsilon,
        )
        normalized_layers[-1] = normalized_output
        aux["last_layer_gamma"] = gamma
        aux["num_classes"] = int(num_classes)

    return normalized_layers, scale_factors, aux


__all__ = [
    "compute_incoming_norms",
    "normalize_layers",
]
