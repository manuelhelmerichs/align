"""Discriminated architecture-recipe selection configuration.

The ``architecture`` mapping names a recipe ``family`` and carries exactly the
discovery options that family accepts; there is no separate free-form options
mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._utils import _validate_fields

# Discovery options each recipe family accepts (besides ``family`` itself).
_FAMILY_OPTIONS: dict[str, frozenset[str]] = {
    "mlp": frozenset({"parameter_root"}),
    "convnet": frozenset({"parameter_root"}),
    "residual_convnet": frozenset(
        {
            "parameter_root",
            "batch_stats_root",
            "residual_topology",
            "residual_connections",
        }
    ),
    "layernorm_mha_transformer": frozenset({"parameter_root"}),
    "rmsnorm_gqa_rope_transformer": frozenset(
        {"parameter_root", "stream_transform_family"}
    ),
}


@dataclass
class ArchitectureConfig:
    """Recipe family plus its validated discovery options."""

    family: str = "mlp"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in _FAMILY_OPTIONS:
            raise ValueError(
                f"Unknown architecture family '{self.family}'. Available: "
                f"{', '.join(sorted(_FAMILY_OPTIONS))}"
            )
        self.options = dict(self.options)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ArchitectureConfig:
        if not isinstance(payload, Mapping):
            raise ValueError("architecture section must be a mapping.")
        family = str(payload.get("family", "mlp"))
        if family not in _FAMILY_OPTIONS:
            raise ValueError(
                f"Unknown architecture family '{family}'. Available: "
                f"{', '.join(sorted(_FAMILY_OPTIONS))}"
            )
        _validate_fields(
            f"architecture (family={family})",
            payload,
            _FAMILY_OPTIONS[family] | {"family"},
        )
        options = {key: value for key, value in payload.items() if key != "family"}
        return cls(family=family, options=options)

    @property
    def recipe_kwargs(self) -> dict[str, Any]:
        """Keyword arguments forwarded to ``get_recipe(family, ...)``."""

        return dict(self.options)

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, **self.options}


__all__ = ["ArchitectureConfig"]
