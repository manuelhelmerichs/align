"""Abstract base class for re-basin strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import jax

    from ..dense_layers import DenseLayer


@runtime_checkable
class StrategyConfig(Protocol):
    """Lightweight protocol for strategy configuration dataclasses."""

    method: str

    @property
    def method_kwargs(self) -> Mapping[str, Any]: ...


class RebasinStrategy(ABC):
    """Base class for all re-basin strategies.

    Strategies now operate on architecture-agnostic alignment views built
    by an :class:`align.architecture.ArchitectureAdapter`.
    """

    name: str  # e.g., "weight_matching", "sinkhorn"

    @abstractmethod
    def match(
        self,
        spec,
        ref_views: Mapping[str, Sequence[Any]],
        target_views: Mapping[str, Sequence[Any]],
        *,
        rng_key: jax.Array | None = None,
    ) -> tuple[Mapping[str, Any], dict[str, Any] | None]:
        """Compute permutations aligning target to reference.

        Args:
            spec: Alignment spec describing permutation groups.
            ref_views: Flattened views for the reference model keyed by group id.
            target_views: Flattened views for the target model keyed by group id.
            rng_key: Optional JAX PRNG key for stochastic strategies.

        Returns:
            A tuple of (permutation_matrices, auxiliary_info).
            The auxiliary info dict may be None or contain strategy-specific data.
        """

    @abstractmethod
    def identity_permutations(
        self, spec, ref_views: Mapping[str, Sequence[Any]] | None = None
    ) -> Mapping[str, Any]:
        """Return identity permutations for the provided spec.

        Args:
            spec: Alignment spec.

        Returns:
            Mapping of group_id -> permutation matrix.
        """

    def supports_batching(self) -> bool:
        """Return True if this strategy supports batch processing.

        Override in subclasses that implement efficient batched matching.
        """
        return False

    def warmup(
        self, spec, ref_views: Mapping[str, Sequence[Any]], *, batch_size: int = 1
    ) -> None:
        """Optional hook for compiling or priming kernels before execution."""
        return None

    def estimate_memory(
        self,
        spec,
        *,
        batch_size: int = 1,
    ) -> int | None:
        """Return a rough byte estimate for strategy execution, if available."""
        return None

    def batch_match_layers(
        self,
        spec,
        target_batches,
        *,
        rng_keys: Sequence[jax.Array | None] | None = None,
    ) -> tuple[list[Mapping[str, Any]], list[dict[str, Any] | None]]:
        """Batch version of match for strategies that support it.

        Args:
            spec: Alignment spec for the model.
            target_batches: Sequence of target view mappings to align.
            rng_keys: Optional per-sample PRNG keys.

        Returns:
            Tuple of (list of permutation lists, list of auxiliary dicts).

        Raises:
            NotImplementedError: If the strategy doesn't support batching.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batched matching."
        )

    @property
    def permutation_dtype(self) -> np.dtype:
        """Return the numpy dtype for serializing permutation matrices.

        Override in subclasses if the default (float32) is not appropriate.
        """
        return np.dtype(np.float32)

    @property
    def requires_numpy_views(self) -> bool:
        """Return True if match expects NumPy-backed views."""
        return False


class NormalizationStrategy(ABC):
    """Base class for scale normalization strategies."""

    name: str

    @abstractmethod
    def normalize_layers(
        self,
        layers: Sequence[DenseLayer],
        *,
        task_type: str = "regression",
        num_classes: int | None = None,
        **kwargs: Any,
    ) -> tuple[list[DenseLayer], list[Any], dict[str, Any] | None]:
        """Normalize scale symmetries in the provided layers."""

    def supports_batching(self) -> bool:
        """Return True if this strategy supports batch_normalize_layers."""
        return False

    def batch_normalize_layers(
        self,
        layers_batch: Sequence[Sequence[DenseLayer]],
        *,
        task_type: str = "regression",
        num_classes: int | None = None,
        **kwargs: Any,
    ) -> tuple[list[list[DenseLayer]], list[list[Any]], list[dict[str, Any] | None]]:
        """Batch version of normalize_layers for strategies that support it.

        Default implementation falls back to sequential processing.
        Override in subclasses for efficient batched implementation.
        """
        results_layers = []
        results_scales = []
        results_aux = []
        for layers in layers_batch:
            norm_layers, scales, aux = self.normalize_layers(
                layers, task_type=task_type, num_classes=num_classes, **kwargs
            )
            results_layers.append(norm_layers)
            results_scales.append(scales)
            results_aux.append(aux)
        return results_layers, results_scales, results_aux

    def warmup(self, layers: Sequence[DenseLayer]) -> None:
        """Optional hook for ahead-of-time compilation in normalization paths."""
        return None


__all__ = ["RebasinStrategy", "NormalizationStrategy", "StrategyConfig"]
