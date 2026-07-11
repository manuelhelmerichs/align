"""Stage configuration for the canonicalize and match pipeline stages.

These dataclasses are JAX-free: the CLI imports them before configuring the
JAX platform, so nothing here may import the matching/canonicalization runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._utils import _maybe_int, _require_bool, _validate_fields

_SOLVER_FIELDS = frozenset(
    {
        "solver",
        "max_sweeps",
        "max_steps",
        "tolerance",
        "tau",
        "learning_rate",
        "sinkhorn_iterations",
        "init_scale",
        "record_loss_history",
        "record_ambiguity",
        "groups",
        "components",
    }
)
_CANONICALIZE_FIELDS = frozenset(
    {
        "strategy",
        "epsilon",
        "degenerate_channels",
        "include_bias_in_norm",
        "activation",
    }
)
_CENTER_SOFTMAX_HEAD_FIELDS = frozenset({"head"})
_MATCH_FIELDS = frozenset({"objective", "seed", "barycenter_passes", "solvers"})
_OBJECTIVE_FIELDS = frozenset(
    {"type", "weights_path", "tensor_weights", "metrics_path", "tensor_metrics"}
)
_VALID_STRATEGIES = frozenset({"balanced", "unit_norm"})
_VALID_DEGENERATE_CHANNELS = frozenset(
    {"preserve", "zero_outgoing", "canonical_vector"}
)
_VALID_ACTIVATIONS = frozenset({"relu", "leaky_relu", "tlu", "gelu"})
_VALID_OBJECTIVES = frozenset({"euclidean", "diagonal_fisher", "relative_fisher"})
_VALID_SOLVERS = frozenset({"lap", "procrustes", "sinkhorn"})


def _validate_choice(name: str, value: str, valid: frozenset[str]) -> str:
    if value not in valid:
        raise ValueError(
            f"Unknown {name} '{value}'. Available: {', '.join(sorted(valid))}"
        )
    return value


@dataclass(frozen=True)
class SolverStep:
    """JAX-free config representation of one match solver step."""

    solver: str
    max_sweeps: int = 25
    max_steps: int = 200
    tolerance: float = 0.0
    tau: float = 0.1
    learning_rate: float = 1e-2
    sinkhorn_iterations: int = 50
    init_scale: float = 1e-2
    record_loss_history: bool = False
    record_ambiguity: bool = False
    groups: tuple[str, ...] | None = None
    components: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.groups is not None and self.components is not None:
            raise ValueError(
                "A match solver step may set 'groups' or 'components', not both."
            )
        for name, value in (
            ("max_sweeps", self.max_sweeps),
            ("max_steps", self.max_steps),
            ("sinkhorn_iterations", self.sinkhorn_iterations),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"match.solvers.{name} must be a positive integer.")
        for name, value, positive in (
            ("tolerance", self.tolerance, False),
            ("tau", self.tau, True),
            ("learning_rate", self.learning_rate, True),
            ("init_scale", self.init_scale, False),
        ):
            value = float(value)
            if not math.isfinite(value) or (value <= 0.0 if positive else value < 0.0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(
                    f"match.solvers.{name} must be finite and {qualifier}."
                )
        for name, values in (("groups", self.groups), ("components", self.components)):
            if values is not None and (not values or len(set(values)) != len(values)):
                raise ValueError(
                    f"match.solvers.{name} must be a non-empty list of unique names."
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SolverStep:
        _validate_fields("match solver", payload, _SOLVER_FIELDS)
        if "solver" not in payload:
            raise ValueError("Each match solver entry must define 'solver'.")
        groups = payload.get("groups")
        components = payload.get("components")
        return cls(
            solver=_validate_choice(
                "match solver", str(payload["solver"]).lower(), _VALID_SOLVERS
            ),
            max_sweeps=int(payload.get("max_sweeps", 25)),
            max_steps=int(payload.get("max_steps", 200)),
            tolerance=float(payload.get("tolerance", 0.0)),
            tau=float(payload.get("tau", 0.1)),
            learning_rate=float(payload.get("learning_rate", 1e-2)),
            sinkhorn_iterations=int(payload.get("sinkhorn_iterations", 50)),
            init_scale=float(payload.get("init_scale", 1e-2)),
            record_loss_history=_require_bool(
                "match.solvers.record_loss_history",
                payload.get("record_loss_history", False),
            ),
            record_ambiguity=_require_bool(
                "match.solvers.record_ambiguity",
                payload.get("record_ambiguity", False),
            ),
            groups=tuple(str(group) for group in groups)
            if groups is not None
            else None,
            components=tuple(str(component) for component in components)
            if components is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"solver": self.solver, "tolerance": self.tolerance}
        if self.solver in ("lap", "procrustes"):
            payload["max_sweeps"] = self.max_sweeps
        elif self.solver == "sinkhorn":
            payload.update(
                {
                    "max_steps": self.max_steps,
                    "tau": self.tau,
                    "learning_rate": self.learning_rate,
                    "sinkhorn_iterations": self.sinkhorn_iterations,
                    "init_scale": self.init_scale,
                    "record_loss_history": self.record_loss_history,
                    "record_ambiguity": self.record_ambiguity,
                }
            )
        elif self.record_ambiguity:
            payload["record_ambiguity"] = True
        if self.groups is not None:
            payload["groups"] = list(self.groups)
        if self.components is not None:
            payload["components"] = list(self.components)
        return payload


@dataclass
class CanonicalizeConfig:
    """Configuration for the scale-canonicalization stage.

    ``strategy`` chooses the dense/conv orbit representative (``balanced`` or
    ``unit_norm``); leave it unset to let each architecture plan pick its own
    canonical form (attention/RMSNorm circuits have a single fixed form).
    """

    strategy: str | None = None
    epsilon: float = 1e-8
    degenerate_channels: str = "preserve"
    include_bias_in_norm: bool = True
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.strategy is not None:
            self.strategy = _validate_choice(
                "canonicalize.strategy", str(self.strategy).lower(), _VALID_STRATEGIES
            )
        self.epsilon = float(self.epsilon)
        self.degenerate_channels = _validate_choice(
            "canonicalize.degenerate_channels",
            str(self.degenerate_channels),
            _VALID_DEGENERATE_CHANNELS,
        )
        self.include_bias_in_norm = _require_bool(
            "canonicalize.include_bias_in_norm", self.include_bias_in_norm
        )
        self.activation = _validate_choice(
            "canonicalize.activation", str(self.activation).lower(), _VALID_ACTIVATIONS
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CanonicalizeConfig:
        if not isinstance(payload, Mapping):
            raise ValueError("canonicalize section must be a mapping.")
        _validate_fields("canonicalize", payload, _CANONICALIZE_FIELDS)
        return cls(
            strategy=payload.get("strategy"),
            epsilon=payload.get("epsilon", 1e-8),
            degenerate_channels=payload.get("degenerate_channels", "preserve"),
            include_bias_in_norm=_require_bool(
                "canonicalize.include_bias_in_norm",
                payload.get("include_bias_in_norm", True),
            ),
            activation=payload.get("activation", "relu"),
        )

    @property
    def canonicalizer_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :meth:`ScaleCanonicalizer.canonicalize`."""

        return {
            "epsilon": self.epsilon,
            "strategy": self.strategy,
            "degenerate_channels": self.degenerate_channels,
            "include_bias_in_norm": self.include_bias_in_norm,
            "activation": self.activation,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "epsilon": self.epsilon,
            "degenerate_channels": self.degenerate_channels,
            "include_bias_in_norm": self.include_bias_in_norm,
            "activation": self.activation,
        }


@dataclass
class CenterSoftmaxHeadConfig:
    """Configuration for the opt-in softmax-head translation stage.

    Centers the classification head to zero-mean class columns — exactly
    probability-preserving but *not* logit-preserving, which is why it is a
    separate stage rather than part of ``canonicalize``. ``head`` names the
    head module (dot-separated path containing ``kernel`` and optionally
    ``bias``); leave it unset to auto-detect the unique structural head.
    """

    head: str | None = None

    def __post_init__(self) -> None:
        if self.head is not None:
            self.head = str(self.head)
            if not self.head:
                raise ValueError("center_softmax_head.head must be a non-empty path.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CenterSoftmaxHeadConfig:
        if not isinstance(payload, Mapping):
            raise ValueError("center_softmax_head section must be a mapping.")
        _validate_fields("center_softmax_head", payload, _CENTER_SOFTMAX_HEAD_FIELDS)
        return cls(head=payload.get("head"))

    @property
    def head_path(self) -> tuple[str, ...] | None:
        return tuple(self.head.split(".")) if self.head is not None else None

    def as_dict(self) -> dict[str, Any]:
        return {"head": self.head}


@dataclass
class ObjectiveConfig:
    """Discriminated match objective configuration."""

    type: str = "euclidean"
    weights_path: str | None = None
    tensor_weights: Any = None
    metrics_path: str | None = None
    tensor_metrics: Any = None

    def __post_init__(self) -> None:
        self.type = _validate_choice(
            "match.objective.type", str(self.type).lower(), _VALID_OBJECTIVES
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | str) -> ObjectiveConfig:
        if isinstance(payload, str):
            payload = {"type": payload}
        if not isinstance(payload, Mapping):
            raise ValueError("match.objective must be a mapping.")
        _validate_fields("match.objective", payload, _OBJECTIVE_FIELDS)
        return cls(
            type=payload.get("type", "euclidean"),
            weights_path=payload.get("weights_path"),
            tensor_weights=payload.get("tensor_weights"),
            metrics_path=payload.get("metrics_path"),
            tensor_metrics=payload.get("tensor_metrics"),
        )

    @property
    def kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.weights_path is not None:
            kwargs["weights_path"] = self.weights_path
        if self.tensor_weights is not None:
            kwargs["tensor_weights"] = self.tensor_weights
        if self.metrics_path is not None:
            kwargs["metrics_path"] = self.metrics_path
        if self.tensor_metrics is not None:
            kwargs["tensor_metrics"] = self.tensor_metrics
        return kwargs

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.weights_path is not None:
            payload["weights_path"] = self.weights_path
        if self.metrics_path is not None:
            payload["metrics_path"] = self.metrics_path
        return payload


@dataclass
class MatchConfig:
    """Configuration for the matrix-transform matching stage."""

    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    seed: int | None = None
    barycenter_passes: int = 2
    solvers: tuple[SolverStep, ...] = field(
        default_factory=lambda: (SolverStep(solver="lap"),)
    )

    def __post_init__(self) -> None:
        if isinstance(self.barycenter_passes, bool):
            raise ValueError("match.barycenter_passes must be an integer at least 1.")
        try:
            passes = int(self.barycenter_passes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "match.barycenter_passes must be an integer at least 1."
            ) from exc
        if passes != self.barycenter_passes or passes < 1:
            raise ValueError("match.barycenter_passes must be an integer at least 1.")
        self.barycenter_passes = passes

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MatchConfig:
        if not isinstance(payload, Mapping):
            raise ValueError("match section must be a mapping.")
        _validate_fields("match", payload, _MATCH_FIELDS)
        raw_solvers = payload.get("solvers")
        if raw_solvers is None:
            solvers = (SolverStep(solver="lap"),)
        else:
            if not isinstance(raw_solvers, list) or not raw_solvers:
                raise ValueError("match.solvers must be a non-empty list.")
            solvers = tuple(SolverStep.from_mapping(dict(item)) for item in raw_solvers)
        return cls(
            objective=ObjectiveConfig.from_mapping(payload.get("objective", {})),
            seed=_maybe_int(payload.get("seed")),
            barycenter_passes=payload.get("barycenter_passes", 2),
            solvers=solvers,
        )

    def solvers_payload(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.solvers]

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective.as_dict(),
            "seed": self.seed,
            "barycenter_passes": self.barycenter_passes,
            "solvers": self.solvers_payload(),
        }


__all__ = [
    "CanonicalizeConfig",
    "CenterSoftmaxHeadConfig",
    "MatchConfig",
    "ObjectiveConfig",
    "SolverStep",
]
