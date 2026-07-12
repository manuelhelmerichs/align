"""Dense MLP architecture recipe: one dense-stack component."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ..symmetry import SymmetryGraph
from ..symmetry.tensor_ops import _descend
from ._utils import natural_key, path_tokens
from .graph_builder import SymmetryGraphBuilder
from .recipe import ArchitectureRecipe, register_recipe
from .rules import DenseChainRule
from .schemas import MLP_OPTIONS


@register_recipe
@dataclass
class MLPRecipe(ArchitectureRecipe):
    """Recipe for simple dense MLP parameter trees.

    Discovers the layer chain under ``parameter_root`` and delegates construction
    to :class:`~align.architectures.rules.DenseChainRule` (component
    ``mlp``, hidden groups ``mlp/h{j}``).
    """

    name: str = "mlp"
    parameter_root: str = "params.fcn"
    config_options: ClassVar[frozenset[str]] = MLP_OPTIONS

    def _infer_layer_paths(self, params: Mapping[str, Any]) -> list[tuple[str, ...]]:
        root = path_tokens(self.parameter_root)
        subtree = _descend(params, root)
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

        layer_names = sorted(candidates, key=natural_key)
        if not layer_names:
            raise ValueError(
                f"No dense layers discovered under '{self.parameter_root}'."
            )
        return [(root + (name,)) for name in layer_names]

    def build_graph(self, params: Mapping[str, Any]) -> SymmetryGraph:
        layer_paths = self._infer_layer_paths(params)
        builder = SymmetryGraphBuilder(params, architecture=self.name)
        DenseChainRule(component_id="mlp", layer_paths=tuple(layer_paths)).add_to(
            builder
        )
        builder.metadata["layer_paths"] = layer_paths
        return builder.finish()
