"""Dense MLP architecture recipe: one dense-stack component."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..symmetry import SymmetryGraph
from .graph_builder import SymmetryGraphBuilder
from .recipe import ArchitectureRecipe, register_recipe
from .rules import DenseChainRule


@register_recipe
@dataclass
class MLPRecipe(ArchitectureRecipe):
    """Recipe for simple dense MLP parameter trees.

    Discovers the layer chain under ``parameter_root`` and delegates construction
    to :class:`~align.architectures.rules.DenseChainRule` (component
    ``fcn``, hidden groups ``fcn/h{j}``).
    """

    name: str = "dense_mlp"
    parameter_root: str = "params.fcn"

    def _tokens_from_path(self, path: str) -> tuple[str, ...]:
        return tuple(part for part in path.split(".") if part)

    def _natural_key(self, value: str) -> list[int | str]:
        parts = re.split(r"(\d+)", value)
        return [int(part) if part.isdigit() else part for part in parts if part]

    def _descend(self, mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        node: Any = mapping
        for key in path:
            node = node[key]
        return node

    def _infer_layer_paths(self, params: Mapping[str, Any]) -> list[tuple[str, ...]]:
        root = self._tokens_from_path(self.parameter_root)
        subtree = self._descend(params, root)
        if not isinstance(subtree, Mapping):
            raise ValueError(
                f"Expected mapping under '{self.parameter_root}', got {type(subtree)}"
            )

        candidates: list[str] = []
        for name, value in subtree.items():
            if not isinstance(value, Mapping):
                continue
            if "kernel" in value and "bias" in value:
                candidates.append(str(name))

        layer_names = sorted(candidates, key=self._natural_key)
        if not layer_names:
            raise ValueError(
                f"No dense layers discovered under '{self.parameter_root}'."
            )
        return [(root + (name,)) for name in layer_names]

    def build_problem(self, params: Mapping[str, Any]) -> SymmetryGraph:
        layer_paths = self._infer_layer_paths(params)
        builder = SymmetryGraphBuilder(params, architecture=self.name)
        DenseChainRule(component_id="fcn", layer_paths=tuple(layer_paths)).build(
            builder
        )
        builder.metadata["layer_paths"] = layer_paths
        return builder.finish()
