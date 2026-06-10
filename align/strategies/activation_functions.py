"""JAX activation function lookup for strategies and costs.

This module provides a simple registry for obtaining JAX activation callables
by name. Used by activation-matching strategies and cost functions that need
to forward-pass through networks during alignment.

**Important**: This is NOT the same as activation normalizers. For norm computation
during scale normalization, see ``align.activation_normalizers``.

Usage::

    from align.strategies.activation_functions import get_activation_fn

    relu = get_activation_fn("relu")
    activations = relu(pre_activations)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp


def get_activation_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a JAX activation callable by name."""
    normalized = name.lower()
    if normalized == "relu":
        return jax.nn.relu
    if normalized == "tanh":
        return jnp.tanh
    if normalized == "gelu":
        return jax.nn.gelu
    if normalized == "identity":
        return lambda x: x
    raise ValueError(f"Unknown activation function '{name}'")


__all__ = ["get_activation_fn"]
