"""Stage-specific configuration for normalization and rebasin."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..options import SolverStep
from ._utils import _maybe_int, _require_bool, _validate_fields

_REBASIN_FIELDS = frozenset(
    {
        "enabled",
        "objective",
        "objective_kwargs",
        "schedule",
        "parameter_root",
        "seed",
        "barycenter_passes",
    }
)
_NORMALIZE_FIELDS = frozenset(
    {
        "enabled",
        "method",
        "parameter_root",
        "task_type",
        "num_classes",
        "scale_normalize",
    }
)
_SCALE_NORMALIZE_FIELDS = frozenset(
    {
        "epsilon",
        "mode",
        "degenerate_handling",
        "normalize_biases",
        "activation",
        "activation_kwargs",
        "classification_head_rescale",
    }
)
_VALID_SCALE_MODES = frozenset({"balanced", "unit_norm"})
_VALID_DEGENERATE_HANDLING = frozenset(
    {"preserve", "zero_outgoing", "canonical_vector"}
)
_VALID_NORMALIZATION_METHODS = frozenset({"scale_normalize"})
_VALID_TASK_TYPES = frozenset({"regression", "classification"})
_VALID_ACTIVATIONS = frozenset({"relu", "leaky_relu", "tlu", "gelu"})


def _validate_choice(name: str, value: str, valid: frozenset[str]) -> str:
    if value not in valid:
        raise ValueError(
            f"Unknown {name} '{value}'. Available: {', '.join(sorted(valid))}"
        )
    return value


@dataclass
class ScaleCanonicalizationConfig:
    """Configuration options for scale normalization."""

    epsilon: float = 1e-8
    mode: str | None = None
    degenerate_handling: str = "preserve"
    normalize_biases: bool = True
    activation: str = "relu"
    activation_kwargs: dict[str, Any] = field(default_factory=dict)
    classification_head_rescale: bool = False

    def __post_init__(self):
        self.epsilon = float(self.epsilon)
        if self.mode is not None:
            self.mode = _validate_choice(
                "scale_normalize.mode", str(self.mode).lower(), _VALID_SCALE_MODES
            )
        self.degenerate_handling = _validate_choice(
            "degenerate_handling",
            str(self.degenerate_handling),
            _VALID_DEGENERATE_HANDLING,
        )
        self.normalize_biases = _require_bool(
            "scale_normalize.normalize_biases", self.normalize_biases
        )
        self.activation = _validate_choice(
            "scale_normalize.activation",
            str(self.activation).lower(),
            _VALID_ACTIVATIONS,
        )
        if not isinstance(self.activation_kwargs, Mapping):
            raise ValueError("scale_normalize.activation_kwargs must be a mapping.")
        self.activation_kwargs = dict(self.activation_kwargs)
        if self.activation_kwargs:
            raise ValueError(
                "scale_normalize.activation_kwargs is currently unsupported."
            )
        self.classification_head_rescale = _require_bool(
            "scale_normalize.classification_head_rescale",
            self.classification_head_rescale,
        )


@dataclass
class CanonicalizeConfig:
    """Configuration for the normalization stage."""

    enabled: bool = True
    method: str = "scale_normalize"
    parameter_root: str | None = None
    task_type: str = "regression"
    num_classes: int | None = None
    scale_normalize: ScaleCanonicalizationConfig = field(
        default_factory=ScaleCanonicalizationConfig
    )

    def __post_init__(self):
        self.enabled = _require_bool("normalize.enabled", self.enabled)
        self.method = _validate_choice(
            "normalization method",
            str(self.method).lower(),
            _VALID_NORMALIZATION_METHODS,
        )
        self.task_type = _validate_choice(
            "normalize.task_type",
            str(self.task_type).lower(),
            _VALID_TASK_TYPES,
        )
        if self.parameter_root is not None:
            self.parameter_root = str(self.parameter_root)
        self.num_classes = _maybe_int(self.num_classes)
        if not isinstance(self.scale_normalize, ScaleCanonicalizationConfig):
            raise ValueError(
                "normalize.scale_normalize must define normalization options."
            )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | bool | None
    ) -> CanonicalizeConfig | None:
        if payload is None:
            return None
        if payload is False:
            return cls(enabled=False)
        if payload is True:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError("normalize section must be a mapping, boolean, or null.")

        _validate_fields("normalize", payload, _NORMALIZE_FIELDS)
        scale_opts = payload.get("scale_normalize", {})
        if not isinstance(scale_opts, Mapping):
            raise ValueError("normalize.scale_normalize must be a mapping.")
        _validate_fields(
            "normalize.scale_normalize", scale_opts, _SCALE_NORMALIZE_FIELDS
        )
        return cls(
            enabled=_require_bool("normalize.enabled", payload.get("enabled", True)),
            method=payload.get("method", "scale_normalize"),
            parameter_root=payload.get("parameter_root"),
            task_type=payload.get("task_type", "regression"),
            num_classes=_maybe_int(payload.get("num_classes")),
            scale_normalize=ScaleCanonicalizationConfig(**scale_opts),
        )

    @property
    def method_kwargs(self) -> dict[str, Any]:
        return {
            "epsilon": self.scale_normalize.epsilon,
            "mode": self.scale_normalize.mode,
            "degenerate_handling": self.scale_normalize.degenerate_handling,
            "normalize_biases": self.scale_normalize.normalize_biases,
            "activation": self.scale_normalize.activation,
            "activation_kwargs": self.scale_normalize.activation_kwargs,
            "classification_head_rescale": self.scale_normalize.classification_head_rescale,
        }

    def validate_method(self) -> None:
        _validate_choice(
            "normalization method", self.method, _VALID_NORMALIZATION_METHODS
        )
        if (
            self.enabled
            and self.method == "scale_normalize"
            and self.task_type == "classification"
            and self.scale_normalize.classification_head_rescale
            and self.num_classes is None
        ):
            raise ValueError(
                "num_classes must be specified when classification_head_rescale is enabled."
            )


@dataclass
class MatchConfig:
    """Configuration for the rebasin stage."""

    enabled: bool = True
    objective: str = "l2_weight"
    objective_kwargs: dict[str, Any] = field(default_factory=dict)
    schedule: tuple[SolverStep, ...] = field(
        default_factory=lambda: (SolverStep(solver="lap"),)
    )
    parameter_root: str | None = None
    seed: int | None = None
    barycenter_passes: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.barycenter_passes, bool):
            raise ValueError("rebasin.barycenter_passes must be an integer at least 1.")
        try:
            barycenter_passes = int(self.barycenter_passes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rebasin.barycenter_passes must be an integer at least 1."
            ) from exc
        if barycenter_passes != self.barycenter_passes or barycenter_passes < 1:
            raise ValueError("rebasin.barycenter_passes must be an integer at least 1.")
        self.barycenter_passes = barycenter_passes

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | bool | None
    ) -> MatchConfig | None:
        if payload is None:
            return None
        if payload is False:
            return cls(enabled=False)
        if payload is True:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError("rebasin section must be a mapping, boolean, or null.")

        _validate_fields("rebasin", payload, _REBASIN_FIELDS)
        raw_schedule = payload.get("schedule")
        if raw_schedule is None:
            schedule = (SolverStep(solver="lap"),)
        else:
            if not isinstance(raw_schedule, list) or not raw_schedule:
                raise ValueError("rebasin.schedule must be a non-empty list.")
            schedule = tuple(
                SolverStep.from_mapping(dict(item)) for item in raw_schedule
            )
        return cls(
            enabled=_require_bool("rebasin.enabled", payload.get("enabled", True)),
            objective=str(payload.get("objective", "l2_weight")),
            objective_kwargs=dict(payload.get("objective_kwargs", {})),
            schedule=schedule,
            parameter_root=payload.get("parameter_root"),
            seed=_maybe_int(payload.get("seed")),
            barycenter_passes=payload.get("barycenter_passes", 2),
        )

    def schedule_payload(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.schedule]

    def validate_method(self) -> None:
        """Validate objective and solver names without importing JAX."""

        valid_objectives = {"l2_weight", "fisher_l2"}
        if self.objective.lower() not in valid_objectives:
            raise ValueError(
                f"Unknown rebasin objective '{self.objective}'. "
                f"Available: {', '.join(sorted(valid_objectives))}"
            )
        valid_solvers = {"lap", "procrustes", "sinkhorn"}
        for step in self.schedule:
            if step.solver.lower() not in valid_solvers:
                raise ValueError(
                    f"Unknown rebasin solver '{step.solver}'. "
                    f"Available: {', '.join(sorted(valid_solvers))}"
                )


__all__ = ["ScaleCanonicalizationConfig", "CanonicalizeConfig", "MatchConfig"]
