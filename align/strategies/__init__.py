"""Strategy pattern for re-basin algorithms.

This module provides a registry and factory for re-basin strategies, allowing
easy extension with new algorithms while maintaining a consistent interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import NormalizationStrategy, RebasinStrategy

STRATEGIES: dict[str, type[RebasinStrategy]] = {}
NORMALIZATION_STRATEGIES: dict[str, type[NormalizationStrategy]] = {}


def register_strategy(name: str):
    """Decorator to register a strategy class under the given name."""

    def decorator(cls: type[RebasinStrategy]) -> type[RebasinStrategy]:
        STRATEGIES[name] = cls
        return cls

    return decorator


def register_normalization_strategy(name: str):
    """Decorator to register a normalization strategy."""

    def decorator(cls: type[NormalizationStrategy]) -> type[NormalizationStrategy]:
        NORMALIZATION_STRATEGIES[name] = cls
        return cls

    return decorator


def get_strategy(name: str, **kwargs: Any) -> RebasinStrategy:
    """Factory function to instantiate a strategy by name.

    Args:
        name: The registered name of the strategy (e.g., "weight_matching", "sinkhorn").
        **kwargs: Strategy-specific configuration options.

    Returns:
        An instance of the requested strategy.

    Raises:
        ValueError: If the strategy name is not registered.
    """
    name_lower = name.lower()
    if name_lower not in STRATEGIES:
        available = ", ".join(sorted(STRATEGIES.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGIES[name_lower](**kwargs)


def get_normalization_strategy(name: str, **kwargs: Any) -> NormalizationStrategy:
    """Factory function to instantiate a normalization strategy by name."""

    name_lower = name.lower()
    if name_lower not in NORMALIZATION_STRATEGIES:
        available = ", ".join(sorted(NORMALIZATION_STRATEGIES.keys()))
        raise ValueError(
            f"Unknown normalization strategy '{name}'. Available: {available}"
        )
    return NORMALIZATION_STRATEGIES[name_lower](**kwargs)


def available_strategies() -> list[str]:
    """Return a sorted list of registered strategy names."""
    return sorted(STRATEGIES.keys())


def available_normalization_strategies() -> list[str]:
    """Return a sorted list of registered normalization strategies."""

    return sorted(NORMALIZATION_STRATEGIES.keys())


from . import scale_normalize as _scale_normalize  # noqa: E402, F401
from . import sinkhorn as _sinkhorn  # noqa: E402, F401
from . import weight_matching as _weight_matching  # noqa: E402, F401

__all__ = [
    "STRATEGIES",
    "NORMALIZATION_STRATEGIES",
    "register_strategy",
    "register_normalization_strategy",
    "get_strategy",
    "get_normalization_strategy",
    "available_strategies",
    "available_normalization_strategies",
]
