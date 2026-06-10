"""Lightweight dataclasses shared between strategy modules and config."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WeightMatchingOptions:
    """Configuration options for the weight matching strategy."""

    max_iters: int = 25
    tol: float = 0.0


@dataclass
class SinkhornOptions:
    """Configuration options for the Sinkhorn strategy."""

    tau: float = 0.1
    n_sinkhorn_iters: int = 50
    lr: float = 1e-2
    max_steps: int = 200
    tol: float = 1e-5
    init_scale: float = 1e-2
    record_loss_history: bool = False
    cost_function: str = "l2_weight"
    cost_function_kwargs: dict[str, Any] = field(default_factory=dict)


__all__ = ["WeightMatchingOptions", "SinkhornOptions"]
