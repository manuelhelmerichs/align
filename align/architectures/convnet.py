"""Plain ConvNet architecture recipe: conv stack -> flatten -> dense chain.

Targets LeNet-style trees (including the bundled ``LeNetti`` family): a sequence
of convolutions (optionally separated by parameter-free, channel-preserving
ops such as pooling), a flatten, and a dense classifier chain. Discovery is
structural: modules with 4-D kernels are convs, modules with 2-D kernels are
dense layers; each family is ordered by natural name sort and validated by
shape chaining.

The flatten boundary is the interesting part. Flax flattens ``(batch, h, w,
c)`` activations with ``reshape(batch, -1)``, so row ``(p, c)`` of the first
dense kernel is ``p * C + c`` (channel-fastest). Permuting the last conv's
``C`` output channels therefore permutes each spatial position's contiguous
row slice of the first dense kernel: the conv group binds that kernel's input
axis once per spatial position, on the half-open interval ``[p*C, (p+1)*C)``.
Same-axis multi-bindings keep the exact LAP linearization (each slice
contributes an independent linear term).

Scale symmetries: bare conv channels and dense hidden units carry the usual
positive-homogeneous scale symmetry, handled by the conv producer-energy plan
(the ``convnet`` component selects it); non-homogeneous activations (e.g. the
GELU lenetti) are rejected loudly by that plan, which is correct — they have
no scale symmetry.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..symmetry import SymmetryGraph
from .graph_builder import SymmetryGraphBuilder
from .recipe import ArchitectureRecipe, register_recipe
from .rules import DenseChainRule
from .schemas import CONVNET_OPTIONS


def _natural_key(value: str) -> list[int | str]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts if part]


def _shape(node: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.shape(node))


@register_recipe
@dataclass
class ConvNetRecipe(ArchitectureRecipe):
    """Recipe for plain conv->flatten->dense parameter trees.

    Emits one ``features`` component of kind ``convnet`` (channel groups named by module
    scope path) plus one ``classifier`` dense-chain component, tied together by the
    flatten-boundary interval bindings described in the module docstring.
    """

    name: str = "convnet"
    parameter_root: str = "core"
    config_options: ClassVar[frozenset[str]] = CONVNET_OPTIONS

    def _tokens_from_path(self, path: str) -> tuple[str, ...]:
        return tuple(part for part in path.split(".") if part)

    def _descend(self, mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        node: Any = mapping
        for key in path:
            node = node[key]
        return node

    def _discover(
        self, params: Mapping[str, Any]
    ) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
        root = self._tokens_from_path(self.parameter_root)
        subtree = self._descend(params, root)
        if not isinstance(subtree, Mapping):
            raise ValueError(
                f"Expected mapping under '{self.parameter_root}', got {type(subtree)}"
            )
        conv_names: list[str] = []
        dense_names: list[str] = []
        for name, value in subtree.items():
            if not isinstance(value, Mapping) or "kernel" not in value:
                continue
            rank = len(_shape(value["kernel"]))
            if rank == 4:
                conv_names.append(str(name))
            elif rank == 2:
                dense_names.append(str(name))
            else:
                raise ValueError(
                    f"convnet recipe found module {name!r} with a {rank}-D kernel "
                    "under the parameter root; only 4-D conv and 2-D dense kernels "
                    "are supported."
                )
        if not conv_names:
            raise ValueError(
                f"No conv layers (4-D kernels) discovered under "
                f"'{self.parameter_root}'; use the mlp recipe for pure "
                "dense trees."
            )
        if not dense_names:
            raise ValueError(
                f"No dense layers (2-D kernels) discovered under "
                f"'{self.parameter_root}'; the convnet recipe needs a dense chain "
                "after the flatten."
            )
        conv_names.sort(key=_natural_key)
        dense_names.sort(key=_natural_key)
        return (
            [(*root, name) for name in conv_names],
            [(*root, name) for name in dense_names],
        )

    def build_graph(self, params: Mapping[str, Any]) -> SymmetryGraph:
        conv_paths, dense_paths = self._discover(params)
        builder = SymmetryGraphBuilder(params, architecture=self.name)

        conv_groups: list[str] = []
        previous_out: int | None = None
        for path in conv_paths:
            module = self._descend(params, path)
            kernel_shape = _shape(module["kernel"])
            in_channels, out_channels = kernel_shape[2], kernel_shape[3]
            if previous_out is not None and in_channels != previous_out:
                raise ValueError(
                    f"ConvNet layers do not chain: {'/'.join(path)} expects "
                    f"{in_channels} input channels, previous conv produced "
                    f"{previous_out}. Channel-mixing ops between convs are "
                    "unsupported."
                )
            group_id = builder.add_group("/".join(path), out_channels)
            builder.bind((*path, "kernel"), 3, group_id, role="out")
            if "bias" in module:
                builder.bind((*path, "bias"), 0, group_id, role="out")
            if conv_groups:
                builder.bind((*path, "kernel"), 2, conv_groups[-1], role="in")
            conv_groups.append(group_id)
            previous_out = out_channels

        first_dense = self._descend(params, dense_paths[0])
        dense_in = _shape(first_dense["kernel"])[0]
        channels = previous_out
        if dense_in % channels != 0:
            raise ValueError(
                f"Flatten boundary mismatch: first dense layer "
                f"{'/'.join(dense_paths[0])} has {dense_in} input rows, which "
                f"is not a multiple of the last conv's {channels} output "
                "channels."
            )
        spatial_positions = dense_in // channels
        builder.bind(
            (*dense_paths[0], "kernel"),
            0,
            conv_groups[-1],
            start=0,
            stop=channels,
            repeat=spatial_positions,
            stride=channels,
            role="in",
        )

        builder.add_component(
            "features",
            "convnet",
            tuple(conv_groups),
            metadata={
                "conv_layers": ["/".join(path) for path in conv_paths],
                "spatial_positions": spatial_positions,
            },
        )
        DenseChainRule(
            component_id="classifier", layer_paths=tuple(dense_paths)
        ).add_to(builder)

        builder.metadata["conv_paths"] = conv_paths
        builder.metadata["dense_paths"] = dense_paths
        builder.metadata["spatial_positions"] = spatial_positions
        return builder.finish()
