"""Dense MLP architecture adapter."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from ..alignment import AlignmentProblem, AxisBinding, PermutationGroup, TensorSpec
from ..normalization import DenseLayer
from .base import ArchitectureAdapter, register_adapter


@register_adapter
@dataclass
class DenseMLPAdapter(ArchitectureAdapter):
    """Adapter for simple dense MLP parameter trees."""

    name: str = "dense_mlp"
    layer_root: str = "params.fcn"

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
        root = self._tokens_from_path(self.layer_root)
        subtree = self._descend(params, root)
        if not isinstance(subtree, Mapping):
            raise ValueError(
                f"Expected mapping under '{self.layer_root}', got {type(subtree)}"
            )

        candidates: list[str] = []
        for name, value in subtree.items():
            if not isinstance(value, Mapping):
                continue
            if "kernel" in value and "bias" in value:
                candidates.append(str(name))

        layer_names = sorted(candidates, key=self._natural_key)
        if not layer_names:
            raise ValueError(f"No dense layers discovered under '{self.layer_root}'.")
        return [(root + (name,)) for name in layer_names]

    def _extract_layers(
        self, params: Mapping[str, Any], layer_paths: list[tuple[str, ...]]
    ) -> list[DenseLayer]:
        layers: list[DenseLayer] = []
        for path in layer_paths:
            layer_dict = self._descend(params, path)
            if not isinstance(layer_dict, Mapping):
                raise TypeError(f"Layer at {path} is not a mapping")
            try:
                kernel = jnp.asarray(layer_dict["kernel"])
                bias = jnp.asarray(layer_dict["bias"])
            except KeyError as exc:
                raise KeyError(
                    f"Layer at {path} must contain 'kernel' and 'bias'."
                ) from exc
            layers.append(DenseLayer(kernel=kernel, bias=bias))
        return layers

    def build_problem(self, params: Mapping[str, Any]) -> AlignmentProblem:
        layer_paths = self._infer_layer_paths(params)
        layers = self._extract_layers(params, layer_paths)

        groups: dict[str, PermutationGroup] = {
            f"layer_{idx}": PermutationGroup(
                id=f"layer_{idx}", size=int(layer.kernel.shape[1])
            )
            for idx, layer in enumerate(layers[:-1])
        }
        tensors: dict[str, TensorSpec] = {}
        bindings: list[AxisBinding] = []

        def _tensor_id(path: tuple[str, ...]) -> str:
            return "/".join(path)

        for idx, (layer, layer_path) in enumerate(
            zip(layers, layer_paths, strict=True)
        ):
            kernel_path = (*layer_path, "kernel")
            bias_path = (*layer_path, "bias")
            kernel_id = _tensor_id(kernel_path)
            bias_id = _tensor_id(bias_path)
            tensors[kernel_id] = TensorSpec(
                id=kernel_id,
                path=kernel_path,
                shape=tuple(int(dim) for dim in layer.kernel.shape),
                metadata={"layer_index": idx, "kind": "kernel"},
            )
            tensors[bias_id] = TensorSpec(
                id=bias_id,
                path=bias_path,
                shape=tuple(int(dim) for dim in layer.bias.shape),
                metadata={"layer_index": idx, "kind": "bias"},
            )

            if idx > 0:
                bindings.append(
                    AxisBinding(
                        tensor_id=kernel_id,
                        axis=0,
                        group=f"layer_{idx - 1}",
                        role="in",
                    )
                )
            if idx < len(layers) - 1:
                group_id = f"layer_{idx}"
                bindings.append(
                    AxisBinding(tensor_id=kernel_id, axis=1, group=group_id, role="out")
                )
                bindings.append(
                    AxisBinding(tensor_id=bias_id, axis=0, group=group_id, role="out")
                )

        problem = AlignmentProblem(
            groups=groups,
            tensors=tensors,
            axis_bindings=tuple(bindings),
            metadata={
                "layer_paths": layer_paths,
                "group_order": [f"layer_{idx}" for idx in range(len(groups))],
                "architecture": self.name,
            },
        )
        problem.validate(params)
        return problem
