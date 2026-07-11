"""Architecture recipe framework: the recipe contract and registry.

An :class:`ArchitectureRecipe` is a thin *recipe*: it discovers where components
live in a parameter tree and composes
:class:`~align.architectures.rules.SymmetryRule` instances, which perform all
graph construction through a shared
:class:`~align.architectures.graph_builder.SymmetryGraphBuilder`. The resulting
:class:`~align.symmetry.SymmetryGraph` is consumed by both the matching
(matrix-transform) and canonicalization (scale) stages. Concrete recipes live
alongside this module under ``align.architectures``.
"""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from .schemas import (
    CONVNET_OPTIONS,
    LAYERNORM_MHA_TRANSFORMER_OPTIONS,
    MLP_OPTIONS,
    RESIDUAL_CONVNET_OPTIONS,
    RMSNORM_GQA_ROPE_TRANSFORMER_OPTIONS,
)


class ArchitectureRecipe(ABC):
    """Discovery + composition recipe for one architecture family.

    Implementations locate component instances (dense stacks, attention modules,
    conv stacks, the residual stream) in a parameter tree and delegate
    construction to component rules. Config and CLI select recipes by ``name``.
    """

    name: str
    config_options: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def build_graph(self, params: Mapping[str, Any]):
        """Return the architecture-derived symmetry graph for ``params``."""


_RECIPES: dict[str, type[ArchitectureRecipe]] = {}
_BUILTIN_RECIPES: dict[str, tuple[str, str, frozenset[str]]] = {
    "convnet": (
        "align.architectures.convnet",
        "ConvNetRecipe",
        CONVNET_OPTIONS,
    ),
    "mlp": (
        "align.architectures.mlp",
        "MLPRecipe",
        MLP_OPTIONS,
    ),
    "rmsnorm_gqa_rope_transformer": (
        "align.architectures.rmsnorm_gqa_rope_transformer",
        "RMSNormGQARoPETransformerRecipe",
        RMSNORM_GQA_ROPE_TRANSFORMER_OPTIONS,
    ),
    "residual_convnet": (
        "align.architectures.residual_convnet",
        "ResidualConvNetRecipe",
        RESIDUAL_CONVNET_OPTIONS,
    ),
    "layernorm_mha_transformer": (
        "align.architectures.layernorm_mha_transformer",
        "LayerNormMHATransformerRecipe",
        LAYERNORM_MHA_TRANSFORMER_OPTIONS,
    ),
}


def register_recipe(cls: type[ArchitectureRecipe]) -> type[ArchitectureRecipe]:
    """Decorator to register an ArchitectureRecipe implementation."""

    if not getattr(cls, "name", None):
        raise ValueError("Recipe classes must define a 'name' attribute.")
    builtin = _BUILTIN_RECIPES.get(cls.name)
    if builtin is not None and frozenset(cls.config_options) != builtin[2]:
        raise ValueError(
            f"Built-in recipe {cls.name!r} declares config options "
            f"{sorted(cls.config_options)}, expected {sorted(builtin[2])}."
        )
    _RECIPES[cls.name] = cls
    return cls


def _load_builtin_recipe(name: str) -> type[ArchitectureRecipe] | None:
    """Import and return the requested built-in recipe class, if known."""
    module_spec = _BUILTIN_RECIPES.get(name)
    if module_spec is None:
        return None

    module_name, class_name, _ = module_spec
    module = importlib.import_module(module_name)
    recipe_cls = getattr(module, class_name)
    if not isinstance(recipe_cls, type) or not issubclass(
        recipe_cls, ArchitectureRecipe
    ):
        raise TypeError(
            f"Built-in recipe '{name}' resolved to invalid class {class_name!r}."
        )
    _RECIPES.setdefault(name, recipe_cls)
    return recipe_cls


def get_recipe(name: str, **kwargs: Any) -> ArchitectureRecipe:
    recipe_cls = _RECIPES.get(name)
    if recipe_cls is None:
        recipe_cls = _load_builtin_recipe(name)
    if recipe_cls is None:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown architecture recipe '{name}'. Available: {available_recipes()}"
        )
    return recipe_cls(**kwargs)


def available_recipes() -> list[str]:
    return sorted(set(_BUILTIN_RECIPES) | set(_RECIPES))


def recipe_options(name: str) -> frozenset[str]:
    """Return a registered schema without importing built-in recipe modules."""

    builtin = _BUILTIN_RECIPES.get(name)
    if builtin is not None:
        return builtin[2]
    recipe_cls = _RECIPES.get(name)
    if recipe_cls is None:
        raise ValueError(
            f"Unknown architecture recipe '{name}'. Available: {available_recipes()}"
        )
    return frozenset(recipe_cls.config_options)


__all__ = [
    "ArchitectureRecipe",
    "available_recipes",
    "get_recipe",
    "register_recipe",
    "recipe_options",
]
