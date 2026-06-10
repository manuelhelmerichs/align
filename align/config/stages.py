"""Stage-specific configuration for normalization and rebasin."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ..options import SinkhornOptions, WeightMatchingOptions
from ._utils import _maybe_int


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
        self.normalize_biases = bool(self.normalize_biases)
        self.classification_head_rescale = bool(self.classification_head_rescale)


@dataclass
class NormalizeConfig:
    """Configuration for the normalization stage."""

    enabled: bool = True
    method: str = "scale_normalize"
    layer_root: str | None = None
    task_type: str = "regression"
    num_classes: int | None = None
    scale_normalize: NormalizationOptions = field(default_factory=NormalizationOptions)

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

        scale_opts = payload.get("scale_normalize", {})
        return cls(
            enabled=bool(payload.get("enabled", True)),
            method=str(payload.get("method", "scale_normalize")),
            layer_root=payload.get("layer_root"),
            task_type=str(payload.get("task_type", "regression")),
            num_classes=_maybe_int(payload.get("num_classes")),
            scale_normalize=NormalizationOptions(**scale_opts),
        )

    @property
    def method_kwargs(self) -> dict[str, Any]:
        if self.method.lower() == "scale_normalize":
            return {
                "epsilon": self.scale_normalize.epsilon,
                "degenerate_handling": self.scale_normalize.degenerate_handling,
                "normalize_biases": self.scale_normalize.normalize_biases,
                "activation": self.scale_normalize.activation,
                "activation_kwargs": self.scale_normalize.activation_kwargs,
                "classification_head_rescale": self.scale_normalize.classification_head_rescale,
            }
        return {}


@dataclass
class RebasinConfig:
    """Configuration for the rebasin stage."""

    enabled: bool = True
    method: str = "weight_matching"
    layer_root: str | None = None
    seed: int | None = None
    weight_matching: WeightMatchingOptions = field(
        default_factory=WeightMatchingOptions
    )
    sinkhorn: SinkhornOptions = field(default_factory=SinkhornOptions)

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

        weight_matching = payload.get("weight_matching", {})
        sinkhorn = payload.get("sinkhorn", {})
        return cls(
            enabled=bool(payload.get("enabled", True)),
            method=str(payload.get("method", "weight_matching")),
            layer_root=payload.get("layer_root"),
            seed=_maybe_int(payload.get("seed")),
            weight_matching=WeightMatchingOptions(**weight_matching),
            sinkhorn=SinkhornOptions(**sinkhorn),
        )

    @property
    def weight_matching_kwargs(self) -> dict[str, Any]:
        return asdict(self.weight_matching)

    @property
    def sinkhorn_kwargs(self) -> dict[str, Any]:
        return asdict(self.sinkhorn)

    @property
    def method_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the currently selected method."""
        method_name = self.method.lower()
        if method_name == "weight_matching":
            return self.weight_matching_kwargs
        if method_name == "sinkhorn":
            return self.sinkhorn_kwargs
        return {}

    def validate_method(self) -> None:
        """Validate that the method is a registered strategy."""
        from ..strategies import available_strategies

        method_name = self.method.lower()
        valid = available_strategies()
        if method_name not in valid:
            raise ValueError(
                f"Unknown method '{self.method}'. Available: {', '.join(valid)}"
            )


__all__ = ["NormalizationOptions", "NormalizeConfig", "RebasinConfig"]
