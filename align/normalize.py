"""Scale normalization for ReLU MLPs."""

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .activation_normalizers import get_activation_normalizer
from .dense_layers import DenseLayer
from .rebasin import ParamTree


@dataclass
class NormalizationResult:
    """Result of normalizing a single sample."""

    normalized_params: ParamTree
    scale_factors: list[jnp.ndarray]
    aux: dict[str, Any] | None = None


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


@functools.partial(jax.jit, static_argnums=(3, 5, 6))
def _normalize_hidden_layer_jit(
    kernel: jax.Array,
    bias: jax.Array,
    norms: jax.Array,
    degenerate_handling: str,
    degenerate_mask: jax.Array,
    epsilon: float,
    normalize_biases: bool,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled hidden layer normalization.

    static_argnums: degenerate_handling (3), epsilon (5), normalize_biases (6).
    """
    safe_norms = jnp.where(degenerate_mask, 1.0, norms)
    normalized_kernel = kernel / safe_norms[None, :]
    normalized_bias = bias / safe_norms if normalize_biases else bias

    if degenerate_handling == "canonical_vector":
        canonical = jnp.zeros_like(normalized_kernel)
        if canonical.shape[0] > 0:
            mask_values = jnp.where(degenerate_mask, 1.0, 0.0)
            canonical = canonical.at[0].set(mask_values)
        normalized_kernel = jnp.where(
            degenerate_mask[None, :], canonical, normalized_kernel
        )
        if normalize_biases:
            normalized_bias = jnp.where(
                degenerate_mask, jnp.zeros_like(normalized_bias), normalized_bias
            )

    return normalized_kernel, normalized_bias


@jax.jit
def _scale_outgoing_jit(kernel: jax.Array, scale: jax.Array) -> jax.Array:
    """JIT-compiled outgoing weight scaling."""
    return kernel * scale[:, None]


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
    return _compute_incoming_norms_jit(
        layer.kernel, layer.bias, epsilon, normalize_biases
    )


def _outgoing_scale_factors(
    norms: jnp.ndarray, degenerate_mask: jnp.ndarray, *, degenerate_handling: str
) -> jnp.ndarray:
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
        normalize_biases,
    )
    return DenseLayer(kernel=kernel, bias=bias)


def scale_outgoing_weights(layer: DenseLayer, scale: jnp.ndarray) -> DenseLayer:
    """Scale outgoing weights (rows) by the provided factors.

    Uses JIT-compiled implementation for performance.
    """
    scaled_kernel = _scale_outgoing_jit(layer.kernel, scale)
    return DenseLayer(kernel=scaled_kernel, bias=layer.bias)


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

    task = (task_type or "regression").lower()
    if task not in ("regression", "classification"):
        raise ValueError("task_type must be 'regression' or 'classification'")
    if task == "classification" and classification_head_rescale and num_classes is None:
        raise ValueError(
            "num_classes required when classification_head_rescale is enabled"
        )

    normalizer = get_activation_normalizer(activation, **(activation_kwargs or {}))

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
            norms = normalizer.compute_norms(
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


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _batch_normalize_hidden_layer_jit(
    kernels: jax.Array,  # (batch, in_dim, out_dim)
    biases: jax.Array,  # (batch, out_dim)
    degenerate_handling: str,
    epsilon: float,
    normalize_biases: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JIT-compiled batched hidden layer normalization using vmap.

    Returns: (normalized_kernels, normalized_biases, scale_factors)
    """
    kernel_sq = jnp.sum(kernels**2, axis=1)
    bias_sq = biases**2 if normalize_biases else jnp.zeros_like(biases)
    raw_norm_sq = kernel_sq + bias_sq
    norms = jnp.sqrt(raw_norm_sq + epsilon)

    degenerate_mask = raw_norm_sq <= epsilon
    safe_norms = jnp.where(degenerate_mask, 1.0, norms)

    # Normalize: kernel / norms[None, :] -> (batch, in_dim, out_dim) / (batch, 1, out_dim)
    normalized_kernels = kernels / safe_norms[:, None, :]
    normalized_biases = biases / safe_norms if normalize_biases else biases

    if degenerate_handling == "canonical_vector":
        canonical = jnp.zeros_like(normalized_kernels)
        mask_values = jnp.where(degenerate_mask, 1.0, 0.0)
        canonical = canonical.at[:, 0, :].set(mask_values)
        normalized_kernels = jnp.where(
            degenerate_mask[:, None, :], canonical, normalized_kernels
        )
        if normalize_biases:
            normalized_biases = jnp.where(
                degenerate_mask, jnp.zeros_like(normalized_biases), normalized_biases
            )

    return normalized_kernels, normalized_biases, norms


def batch_normalize_layers(
    layers_batch: Sequence[Sequence[DenseLayer]],
    *,
    task_type: str = "regression",
    num_classes: int | None = None,
    epsilon: float = 1e-8,
    degenerate_handling: str = "preserve",
    normalize_biases: bool = True,
    activation: str = "relu",
    activation_kwargs: dict[str, Any] | None = None,
    classification_head_rescale: bool = False,
) -> tuple[list[list[DenseLayer]], list[list[jnp.ndarray]], list[dict[str, Any]]]:
    """Apply scale normalization to a batch of layer sequences using vmap.

    Args:
        layers_batch: Sequence of layer sequences, each sample has the same architecture.
        task_type: "regression" or "classification"
        num_classes: Required only when classification_head_rescale is enabled.
        epsilon: Small constant for numerical stability.
        degenerate_handling: How to handle degenerate neurons.
        normalize_biases: Whether to include biases in norm computation.
        activation: Activation function name.
        activation_kwargs: Additional activation kwargs.
        classification_head_rescale: Whether to apply the non-symmetric final-layer
            rescaling heuristic for classification.

    Returns:
        Tuple of (normalized_layers_batch, scale_factors_batch, aux_batch)
    """
    if not layers_batch:
        return [], [], []

    batch_size = len(layers_batch)
    num_layers = len(layers_batch[0])

    if num_layers < 2:
        raise ValueError("Need at least 2 layers (1 hidden + 1 output)")

    task = (task_type or "regression").lower()
    if task not in ("regression", "classification"):
        raise ValueError("task_type must be 'regression' or 'classification'")
    if task == "classification" and classification_head_rescale and num_classes is None:
        raise ValueError(
            "num_classes required when classification_head_rescale is enabled"
        )

    stacked_kernels = [
        jnp.stack(
            [layers_batch[b][layer_idx].kernel for b in range(batch_size)], axis=0
        )
        for layer_idx in range(num_layers)
    ]
    stacked_biases = [
        jnp.stack([layers_batch[b][layer_idx].bias for b in range(batch_size)], axis=0)
        for layer_idx in range(num_layers)
    ]

    normalized_kernels = [None] * num_layers
    normalized_biases = [None] * num_layers
    scale_factors_stacked = []

    for layer_idx in range(num_layers - 1):
        kernels = stacked_kernels[layer_idx]
        biases = stacked_biases[layer_idx]

        if layer_idx > 0:
            outgoing = scale_factors_stacked[layer_idx - 1]
            raw_outgoing_sq = jnp.square(outgoing) - epsilon
            if degenerate_handling in ("zero_outgoing", "canonical_vector"):
                degenerate_mask = raw_outgoing_sq <= epsilon
                outgoing = jnp.where(degenerate_mask, 0.0, outgoing)
            else:
                degenerate_mask = raw_outgoing_sq <= epsilon
                outgoing = jnp.where(degenerate_mask, 1.0, outgoing)
            # Scale: (batch, in_dim, out_dim) * (batch, in_dim, 1)
            kernels = kernels * outgoing[:, :, None]

        norm_k, norm_b, norms = _batch_normalize_hidden_layer_jit(
            kernels, biases, degenerate_handling, epsilon, normalize_biases
        )
        normalized_kernels[layer_idx] = norm_k
        normalized_biases[layer_idx] = norm_b
        scale_factors_stacked.append(norms)

    last_kernels = stacked_kernels[-1]
    if num_layers > 1:
        outgoing = scale_factors_stacked[-1]
        raw_outgoing_sq = jnp.square(outgoing) - epsilon
        if degenerate_handling in ("zero_outgoing", "canonical_vector"):
            degenerate_mask = raw_outgoing_sq <= epsilon
            outgoing = jnp.where(degenerate_mask, 0.0, outgoing)
        else:
            degenerate_mask = raw_outgoing_sq <= epsilon
            outgoing = jnp.where(degenerate_mask, 1.0, outgoing)
        last_kernels = last_kernels * outgoing[:, :, None]

    normalized_kernels[-1] = last_kernels
    normalized_biases[-1] = stacked_biases[-1]

    aux_template: dict[str, Any] = {
        "task_type": task,
        "epsilon": epsilon,
        "degenerate_handling": degenerate_handling,
        "normalize_biases": normalize_biases,
        "activation": activation,
        "classification_head_rescale": classification_head_rescale,
    }

    gammas = None
    if task == "classification" and classification_head_rescale:
        last_k = normalized_kernels[-1]
        last_b = normalized_biases[-1]
        gamma_sq = (
            jnp.sum(last_k**2, axis=(1, 2)) + jnp.sum(last_b**2, axis=1) + epsilon
        )
        gammas = jnp.sqrt(gamma_sq)
        scale = jnp.sqrt(num_classes) / gammas
        normalized_kernels[-1] = last_k * scale[:, None, None]
        normalized_biases[-1] = last_b * scale[:, None]

    normalized_batch: list[list[DenseLayer]] = []
    scale_batch: list[list[jnp.ndarray]] = []
    aux_batch: list[dict[str, Any]] = []

    for b in range(batch_size):
        sample_layers = [
            DenseLayer(
                kernel=normalized_kernels[layer_idx][b],
                bias=normalized_biases[layer_idx][b],
            )
            for layer_idx in range(num_layers)
        ]
        sample_scales = [sf[b] for sf in scale_factors_stacked]
        sample_aux = dict(aux_template)
        if task == "classification" and gammas is not None:
            sample_aux["last_layer_gamma"] = float(gammas[b])
            sample_aux["num_classes"] = int(num_classes)

        normalized_batch.append(sample_layers)
        scale_batch.append(sample_scales)
        aux_batch.append(sample_aux)

    return normalized_batch, scale_batch, aux_batch


__all__ = [
    "NormalizationResult",
    "compute_incoming_norms",
    "normalize_hidden_layer",
    "scale_outgoing_weights",
    "normalize_last_layer_classification",
    "normalize_layers",
    "batch_normalize_layers",
]
