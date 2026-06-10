"""Dense MLP architecture adapter."""

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from flax.core import frozen_dict

from ..architecture import (
    AlignmentSite,
    AlignmentSpec,
    ArchitectureAdapter,
    PermutationGroup,
    PermutationRule,
    ViewTemplate,
    register_adapter,
)
from ..dense_layers import DenseLayer


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

    def _inject_layers(
        self,
        params: Mapping[str, Any],
        layer_paths: list[tuple[str, ...]],
        updated_layers: list[DenseLayer],
    ) -> Mapping[str, Any]:
        if len(layer_paths) != len(updated_layers):
            raise ValueError("Layer count mismatch while injecting updated layers.")
        is_frozen = isinstance(params, frozen_dict.FrozenDict)
        mutable = frozen_dict.unfreeze(params) if is_frozen else copy.deepcopy(params)
        for path, layer in zip(layer_paths, updated_layers, strict=True):
            node = self._descend(mutable, path)
            node["kernel"] = layer.kernel
            node["bias"] = layer.bias
        return frozen_dict.freeze(mutable) if is_frozen else mutable

    def build_spec(self, params: Mapping[str, Any]) -> AlignmentSpec:
        layer_paths = self._infer_layer_paths(params)
        layers = self._extract_layers(params, layer_paths)

        groups: dict[str, PermutationGroup] = {}
        view_templates: list[ViewTemplate] = []
        permutation_rules: list[PermutationRule] = []
        sites: list[AlignmentSite] = []

        for idx, layer in enumerate(layers[:-1]):
            group_id = f"layer_{idx}"
            groups[group_id] = PermutationGroup(id=group_id, size=layer.kernel.shape[1])
            layer_path = layer_paths[idx]
            view_templates.append(
                ViewTemplate(
                    name=f"{group_id}_incoming",
                    group=group_id,
                    weight_path=(*layer_path, "kernel"),
                    axis=1,
                    bias_path=(*layer_path, "bias"),
                    role="incoming",
                )
            )
            permutation_rules.extend(
                [
                    PermutationRule(
                        path=(*layer_path, "kernel"),
                        axis=1,
                        group=group_id,
                        direction="right",
                    ),
                    PermutationRule(
                        path=(*layer_path, "bias"),
                        axis=0,
                        group=group_id,
                        direction="left",
                    ),
                ]
            )

            next_path = layer_paths[idx + 1]
            sites.append(
                AlignmentSite(
                    id=group_id,
                    group=group_id,
                    params=(
                        (*layer_path, "kernel"),
                        (*layer_path, "bias"),
                        (*next_path, "kernel"),
                    ),
                    tags=("mlp_hidden",),
                    invariants=("feeds next layer incoming rows",),
                )
            )
            view_templates.append(
                ViewTemplate(
                    name=f"{group_id}_outgoing",
                    group=group_id,
                    weight_path=(*next_path, "kernel"),
                    axis=0,
                    role="outgoing",
                )
            )
            permutation_rules.append(
                PermutationRule(
                    path=(*next_path, "kernel"),
                    axis=0,
                    group=group_id,
                    direction="left",
                )
            )

        return AlignmentSpec(
            groups=groups,
            view_templates=view_templates,
            permutation_rules=permutation_rules,
            sites=sites,
            metadata={
                "layer_paths": layer_paths,
                "group_order": [f"layer_{idx}" for idx in range(len(groups))],
            },
        )

    def normalize(
        self, params: Mapping[str, Any], spec: AlignmentSpec, **kwargs: Any
    ) -> tuple[Mapping[str, Any], list[Any], dict[str, Any] | None]:
        from ..normalize import normalize_layers

        layer_paths = spec.metadata.get("layer_paths")
        if not isinstance(layer_paths, list):
            raise RuntimeError(
                "DenseMLPAdapter spec is missing 'layer_paths' metadata."
            )

        layers = self._extract_layers(params, layer_paths)
        normalized_layers, scale_factors, aux = normalize_layers(layers, **kwargs)
        normalized_params = self._inject_layers(params, layer_paths, normalized_layers)
        return normalized_params, scale_factors, aux
