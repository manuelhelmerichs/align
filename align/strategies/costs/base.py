"""Cost function registry for Sinkhorn rebasining."""

from abc import ABC, abstractmethod

import jax.numpy as jnp


class CostFunction(ABC):
    """Base class for Sinkhorn cost functions."""

    name: str

    @abstractmethod
    def compute_cost(
        self,
        spec,
        ref_views,
        target_views,
        soft_perms,
    ) -> jnp.ndarray:
        """Compute a scalar cost for a single set of alignment views."""

    @abstractmethod
    def compute_batched_cost(
        self,
        spec,
        ref_views,
        target_views_batched,
        soft_perms,
    ) -> jnp.ndarray:
        """Compute a vector of costs for batched alignment views."""


_COST_FUNCTIONS: dict[str, type[CostFunction]] = {}


def register_cost_function(name: str):
    def decorator(cls: type[CostFunction]) -> type[CostFunction]:
        _COST_FUNCTIONS[name] = cls
        return cls

    return decorator


def get_cost_function(name: str, **kwargs) -> CostFunction:
    try:
        cls = _COST_FUNCTIONS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown cost function '{name}'. Available: {list(_COST_FUNCTIONS)}"
        ) from exc
    return cls(**kwargs)


__all__ = ["CostFunction", "register_cost_function", "get_cost_function"]
