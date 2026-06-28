"""Activation normalizer registry for scale normalization.

This module provides activation-specific norm computation for scale normalization.
Each normalizer computes per-neuron norms based on the activation function's
properties (e.g., positive homogeneity for ReLU/LeakyReLU).

Usage::

    from align.normalization import get_activation_normalizer

    normalizer = get_activation_normalizer("relu")
    norms = normalizer.compute_norms(layer, epsilon=1e-8, normalize_biases=True)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from .dense_layers import DenseLayer


class ActivationNormalizer(ABC):
    name: str

    @abstractmethod
    def compute_norms(
        self, layer: DenseLayer, epsilon: float, normalize_biases: bool
    ) -> jnp.ndarray: ...


_NORMALIZERS: dict[str, type[ActivationNormalizer]] = {}


def register_activation_normalizer(name: str):
    def decorator(cls: type[ActivationNormalizer]) -> type[ActivationNormalizer]:
        _NORMALIZERS[name] = cls
        return cls

    return decorator


def get_activation_normalizer(name: str, **kwargs) -> ActivationNormalizer:
    key = name.lower()
    if key not in _NORMALIZERS:
        raise ValueError(
            f"Unknown activation normalizer '{name}'. Available: {list(_NORMALIZERS)}"
        )
    return _NORMALIZERS[key](**kwargs)


@register_activation_normalizer("relu")
class ReLUNormalizer(ActivationNormalizer):
    name = "relu"

    def compute_norms(
        self, layer: DenseLayer, epsilon: float, normalize_biases: bool
    ) -> jnp.ndarray:
        from .kernel import compute_incoming_norms

        return compute_incoming_norms(
            layer, epsilon=epsilon, normalize_biases=normalize_biases
        )


@register_activation_normalizer("leaky_relu")
class LeakyReLUNormalizer(ActivationNormalizer):
    name = "leaky_relu"

    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope

    def compute_norms(
        self, layer: DenseLayer, epsilon: float, normalize_biases: bool
    ) -> jnp.ndarray:
        from .kernel import compute_incoming_norms

        return compute_incoming_norms(
            layer, epsilon=epsilon, normalize_biases=normalize_biases
        )


__all__ = [
    "ActivationNormalizer",
    "get_activation_normalizer",
    "register_activation_normalizer",
    "ReLUNormalizer",
    "LeakyReLUNormalizer",
]
