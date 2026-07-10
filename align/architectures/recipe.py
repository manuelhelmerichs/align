"""Architecture recipe framework: the recipe contract and registry.

An :class:`ArchitectureRecipe` is a thin *recipe*: it discovers where components
live in a parameter tree and composes
:class:`~align.architectures.rules.SymmetryRule` instances, which perform all
problem construction through a shared
:class:`~align.architectures.graph_builder.SymmetryGraphBuilder`. The resulting
:class:`~align.symmetry.SymmetryGraph` is consumed by both the rebasin
(permutation) and normalization (scale) stages. Concrete recipes live
alongside this module under ``align.architectures``.
"""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class ArchitectureRecipe(ABC):
    """Discovery + composition recipe for one architecture family.

    Implementations locate component instances (dense stacks, attention modules,
    conv stacks, the residual stream) in a parameter tree and delegate
    construction to component adapters. Config and CLI select recipes by ``name``.
    """

    name: str

    @abstractmethod
    def build_problem(self, params: Mapping[str, Any]):
        """Return the graph-native alignment problem for ``params``."""


_RECIPES: dict[str, type[ArchitectureRecipe]] = {}
_BUILTIN_RECIPES: dict[str, tuple[str, str]] = {
    "convnet": ("align.architectures.convnet", "ConvNetRecipe"),
    "mlp": ("align.architectures.mlp", "MLPRecipe"),
    "rmsnorm_gqa_rope_transformer": (
        "align.architectures.rmsnorm_gqa_rope_transformer",
        "RMSNormGQARoPETransformerRecipe",
    ),
    "residual_convnet": (
        "align.architectures.residual_convnet",
        "ResidualConvNetRecipe",
    ),
    "layernorm_mha_transformer": (
        "align.architectures.layernorm_mha_transformer",
        "LayerNormMHATransformerRecipe",
    ),
}


def register_recipe(cls: type[ArchitectureRecipe]) -> type[ArchitectureRecipe]:
    """Decorator to register an ArchitectureRecipe implementation."""

    if not getattr(cls, "name", None):
        raise ValueError("Adapter classes must define a 'name' attribute.")
    _RECIPES[cls.name] = cls
    return cls


def _load_builtin_adapter(name: str) -> type[ArchitectureRecipe] | None:
    """Import and return the requested built-in adapter class, if known."""
    module_spec = _BUILTIN_RECIPES.get(name)
    if module_spec is None:
        return None

    module_name, class_name = module_spec
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    if not isinstance(adapter_cls, type) or not issubclass(
        adapter_cls, ArchitectureRecipe
    ):
        raise TypeError(
            f"Built-in adapter '{name}' resolved to invalid class {class_name!r}."
        )
    _RECIPES.setdefault(name, adapter_cls)
    return adapter_cls


def get_recipe(name: str, **kwargs: Any) -> ArchitectureRecipe:
    adapter_cls = _RECIPES.get(name)
    if adapter_cls is None:
        adapter_cls = _load_builtin_adapter(name)
    if adapter_cls is None:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown architecture adapter '{name}'. Available: {available_recipes()}"
        )
    return adapter_cls(**kwargs)


def available_recipes() -> list[str]:
    return sorted(set(_BUILTIN_RECIPES) | set(_RECIPES))


__all__ = [
    "ArchitectureRecipe",
    "available_recipes",
    "get_recipe",
    "register_recipe",
]
