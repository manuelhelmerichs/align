"""Default scale normalization strategy for ReLU MLPs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from . import register_normalization_strategy
from .base import NormalizationStrategy

if TYPE_CHECKING:
    from ..dense_layers import DenseLayer


@register_normalization_strategy("scale_normalize")
@dataclass
class ScaleNormalizeStrategy(NormalizationStrategy):
    """Standard per-neuron scale normalization (Pittorino et al.)."""

    name: str = "scale_normalize"
    epsilon: float = 1e-8
    degenerate_handling: str = "preserve"
    normalize_biases: bool = True
    activation: str = "relu"
    activation_kwargs: dict[str, Any] = field(default_factory=dict)

    def supports_batching(self) -> bool:
        """ScaleNormalizeStrategy supports efficient batched processing."""
        return True

    def normalize_layers(
        self,
        layers: Sequence[DenseLayer],
        *,
        task_type: str = "regression",
        num_classes: int | None = None,
        **kwargs: Any,
    ) -> tuple[list[DenseLayer], list[jnp.ndarray], dict[str, Any] | None]:
        from ..normalize import normalize_layers

        return normalize_layers(
            layers,
            task_type=task_type,
            num_classes=num_classes,
            epsilon=self.epsilon,
            degenerate_handling=self.degenerate_handling,
            normalize_biases=self.normalize_biases,
            activation=self.activation,
            activation_kwargs=self.activation_kwargs,
        )

    def batch_normalize_layers(
        self,
        layers_batch: Sequence[Sequence[DenseLayer]],
        *,
        task_type: str = "regression",
        num_classes: int | None = None,
        **kwargs: Any,
    ) -> tuple[
        list[list[DenseLayer]], list[list[jnp.ndarray]], list[dict[str, Any] | None]
    ]:
        """Batched normalization using vectorized implementation."""
        from ..normalize import batch_normalize_layers

        return batch_normalize_layers(
            layers_batch,
            task_type=task_type,
            num_classes=num_classes,
            epsilon=self.epsilon,
            degenerate_handling=self.degenerate_handling,
            normalize_biases=self.normalize_biases,
            activation=self.activation,
            activation_kwargs=self.activation_kwargs,
        )
