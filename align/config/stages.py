"""Stage-specific configuration for normalization and rebasin."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..options import RebasinScheduleStep
from ._utils import _maybe_int

_REBASIN_FIELDS = frozenset(
    {
        "enabled",
        "objective",
        "objective_kwargs",
        "schedule",
        "layer_root",
        "seed",
    }
)
_NORMALIZE_FIELDS = frozenset(
    {
        "enabled",
        "method",
        "layer_root",
        "task_type",
        "num_classes",
        "scale_normalize",
    }
)
_SCALE_NORMALIZE_FIELDS = frozenset(
    {
        "epsilon",
        "degenerate_handling",
        "normalize_biases",
        "activation",
        "activation_kwargs",
        "classification_head_rescale",
    }
)
_VALID_DEGENERATE_HANDLING = frozenset(
    {"preserve", "zero_outgoing", "canonical_vector"}
)
_VALID_NORMALIZATION_METHODS = frozenset({"scale_normalize"})
_VALID_TASK_TYPES = frozenset({"regression", "classification"})


def _validate_config_fields(
    section: str, payload: Mapping[str, Any], allowed: frozenset[str]
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section} config option(s): " + ", ".join(unknown))


def _require_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean, got {type(value).__name__}.")
    return value


def _validate_choice(name: str, value: str, valid: frozenset[str]) -> str:
    if value not in valid:
        raise ValueError(
            f"Unknown {name} '{value}'. Available: {', '.join(sorted(valid))}"
        )
    return value


@dataclass
class NormalizationOptions:
    """Configuration options for scale normalization."""

    epsilon: float = 1e-8
    degenerate_handling: str = "preserve"
    normalize_biases: bool = True
    activation: str = "relu"
    activation_kwargs: dict[str, Any] = field(default_factory=dict)
    classification_head_rescale: bool = False

    def __post_init__(self):
        self.epsilon = float(self.epsilon)
        self.degenerate_handling = _validate_choice(
            "degenerate_handling",
            str(self.degenerate_handling),
            _VALID_DEGENERATE_HANDLING,
        )
        self.normalize_biases = _require_bool(
            "scale_normalize.normalize_biases", self.normalize_biases
        )
        self.activation = str(self.activation).lower()
        if not isinstance(self.activation_kwargs, Mapping):
            raise ValueError("scale_normalize.activation_kwargs must be a mapping.")
        self.activation_kwargs = dict(self.activation_kwargs)
        self.classification_head_rescale = _require_bool(
            "scale_normalize.classification_head_rescale",
            self.classification_head_rescale,
        )


@dataclass
class NormalizeConfig:
    """Configuration for the normalization stage."""

    enabled: bool = True
    method: str = "scale_normalize"
    layer_root: str | None = None
    task_type: str = "regression"
    num_classes: int | None = None
    scale_normalize: NormalizationOptions = field(default_factory=NormalizationOptions)

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
        if self.layer_root is not None:
            self.layer_root = str(self.layer_root)
        self.num_classes = _maybe_int(self.num_classes)
        if not isinstance(self.scale_normalize, NormalizationOptions):
            raise ValueError(
                "normalize.scale_normalize must define normalization options."
            )

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | bool | None
    ) -> NormalizeConfig | None:
        if payload is None:
            return None
        if payload is False:
            return cls(enabled=False)
        if payload is True:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError("normalize section must be a mapping, boolean, or null.")

        _validate_config_fields("normalize", payload, _NORMALIZE_FIELDS)
        scale_opts = payload.get("scale_normalize", {})
        if not isinstance(scale_opts, Mapping):
            raise ValueError("normalize.scale_normalize must be a mapping.")
        _validate_config_fields(
            "normalize.scale_normalize", scale_opts, _SCALE_NORMALIZE_FIELDS
        )
        return cls(
            enabled=_require_bool("normalize.enabled", payload.get("enabled", True)),
            method=payload.get("method", "scale_normalize"),
            layer_root=payload.get("layer_root"),
            task_type=payload.get("task_type", "regression"),
            num_classes=_maybe_int(payload.get("num_classes")),
            scale_normalize=NormalizationOptions(**scale_opts),
        )

    @property
    def method_kwargs(self) -> dict[str, Any]:
        return {
            "epsilon": self.scale_normalize.epsilon,
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


@dataclass
class RebasinConfig:
    """Configuration for the rebasin stage."""

    enabled: bool = True
    objective: str = "l2_weight"
    objective_kwargs: dict[str, Any] = field(default_factory=dict)
    schedule: tuple[RebasinScheduleStep, ...] = field(
        default_factory=lambda: (RebasinScheduleStep(solver="lap"),)
    )
    layer_root: str | None = None
    seed: int | None = None

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | bool | None
    ) -> RebasinConfig | None:
        if payload is None:
            return None
        if payload is False:
            return cls(enabled=False)
        if payload is True:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError("rebasin section must be a mapping, boolean, or null.")

        _validate_config_fields("rebasin", payload, _REBASIN_FIELDS)
        raw_schedule = payload.get("schedule")
        if raw_schedule is None:
            schedule = (RebasinScheduleStep(solver="lap"),)
        else:
            if not isinstance(raw_schedule, list) or not raw_schedule:
                raise ValueError("rebasin.schedule must be a non-empty list.")
            schedule = tuple(
                RebasinScheduleStep.from_mapping(dict(item)) for item in raw_schedule
            )
        return cls(
            enabled=_require_bool("rebasin.enabled", payload.get("enabled", True)),
            objective=str(payload.get("objective", "l2_weight")),
            objective_kwargs=dict(payload.get("objective_kwargs", {})),
            schedule=schedule,
            layer_root=payload.get("layer_root"),
            seed=_maybe_int(payload.get("seed")),
        )

    def schedule_payload(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.schedule]

    def validate_method(self) -> None:
        """Validate objective and solver names without importing JAX."""

        valid_objectives = {"l2_weight"}
        if self.objective.lower() not in valid_objectives:
            raise ValueError(
                f"Unknown rebasin objective '{self.objective}'. "
                f"Available: {', '.join(sorted(valid_objectives))}"
            )
        valid_solvers = {"lap", "sinkhorn"}
        for step in self.schedule:
            if step.solver.lower() not in valid_solvers:
                raise ValueError(
                    f"Unknown rebasin solver '{step.solver}'. "
                    f"Available: {', '.join(sorted(valid_solvers))}"
                )


__all__ = ["NormalizationOptions", "NormalizeConfig", "RebasinConfig"]
