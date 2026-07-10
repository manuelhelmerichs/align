"""Architecture recipes and the component rules they compose.

Component rules (``align.architectures.rules``) construct alignment problems
through a shared ``SymmetryGraphBuilder``; architecture recipes discover component
locations in a parameter tree and compose them.
"""

import importlib
from typing import Any

from .recipe import (
    ArchitectureRecipe,
    available_recipes,
    get_recipe,
    register_recipe,
)

_RECIPE_EXPORTS = {
    "ConvNetRecipe": "align.architectures.convnet",
    "MLPRecipe": "align.architectures.mlp",
    "RMSNormGQARoPETransformerRecipe": "align.architectures.rmsnorm_gqa_rope_transformer",
    "ResidualConvNetRecipe": "align.architectures.residual_convnet",
    "ResidualConvNetRule": "align.architectures.residual_convnet",
    "LayerNormMHATransformerRecipe": "align.architectures.layernorm_mha_transformer",
    "SymmetryGraphBuilder": "align.architectures.graph_builder",
    "SymmetryRule": "align.architectures.rules",
    "DenseChainRule": "align.architectures.rules",
    "MHAAttentionRule": "align.architectures.rules",
    "GQARoPEAttentionRule": "align.architectures.rules",
    "ResidualStreamRule": "align.architectures.rules",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _RECIPE_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_RECIPE_EXPORTS))


__all__ = [
    "ArchitectureRecipe",
    "MHAAttentionRule",
    "SymmetryRule",
    "ConvNetRecipe",
    "ResidualConvNetRule",
    "MLPRecipe",
    "DenseChainRule",
    "GQARoPEAttentionRule",
    "RMSNormGQARoPETransformerRecipe",
    "SymmetryGraphBuilder",
    "ResidualConvNetRecipe",
    "ResidualStreamRule",
    "LayerNormMHATransformerRecipe",
    "available_recipes",
    "get_recipe",
    "register_recipe",
]
