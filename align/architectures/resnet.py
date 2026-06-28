"""ResNet-style architecture adapter (conv + FRN/BN)."""

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..alignment import (
    AlignmentProblem,
    AxisBinding,
    GraphConstraint,
    PermutationGroup,
    TensorSpec,
)
from ..alignment.tensor_ops import _canonical_axis, _descend, _maybe_descend
from .base import ArchitectureAdapter, register_adapter


def _natural_key(value: str) -> list[int | str]:
    """Split a string into textual and numerical chunks for natural sorting."""

    import re

    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts if part]


class _GroupUnionFind:
    """Union-find that validates consistent channel sizes."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, name: str, size: int) -> None:
        if name in self.parent:
            if self.size[name] != size:
                raise ValueError(
                    f"Group '{name}' has inconsistent sizes: {self.size[name]} vs {size}."
                )
            return
        self.parent[name] = name
        self.size[name] = size

    def find(self, name: str) -> str:
        if name not in self.parent:
            raise KeyError(name)
        parent = self.parent[name]
        if parent != name:
            self.parent[name] = self.find(parent)
        return self.parent[name]

    def union(self, left: str, right: str) -> str:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return root_left
        size_left = self.size[root_left]
        size_right = self.size[root_right]
        if size_left != size_right:
            raise ValueError(
                "Residual join requires matching channel sizes: "
                f"{root_left} has {size_left}, {root_right} has {size_right}."
            )
        self.parent[root_right] = root_left
        return root_left

    def roots(self) -> dict[str, int]:
        return {
            name: size for name, size in self.size.items() if self.find(name) == name
        }


def _validate_channel_size(
    path: tuple[str, ...], array: Any, expected: int, *, axis: int = -1
) -> None:
    """Ensure ``array`` has ``expected`` channels along ``axis``."""
    shape = np.shape(array)
    axis = _canonical_axis(len(shape), axis)
    actual = shape[axis]
    if actual != expected:
        raise ValueError(f"{path} has channel dimension {actual}, expected {expected}.")


def _load_graph_descriptor(descriptor: Any) -> Any:
    """Load a module-graph descriptor from a mapping, list, or file path."""

    if descriptor is None:
        return None
    if isinstance(descriptor, (str, Path)):
        path = Path(descriptor)
        if not path.exists():
            raise FileNotFoundError(f"module_graph path does not exist: {path}")
        text = path.read_text()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return yaml.safe_load(text)
    return descriptor


def _as_name_list(value: Any) -> list[str]:
    """Normalize a possibly nested value into a list of string names."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        for key in ("name", "id", "target"):
            if key in value and isinstance(value[key], str):
                return [value[key]]
        return []
    if isinstance(value, Sequence):
        names: list[str] = []
        for item in value:
            names.extend(_as_name_list(item))
        return names
    return []


def _filter_conv_names(
    names: Sequence[str], conv_set: set[str], *, allow_partial: bool = True
) -> list[str]:
    """Return known conv names, optionally allowing non-conv tokens."""

    def _last_segment(name: str) -> str:
        return name.split("/")[-1]

    unknown_conv_like = [
        name
        for name in names
        if isinstance(name, str)
        and _last_segment(name).lower().startswith("conv")
        and name not in conv_set
    ]
    if unknown_conv_like:
        joined = ", ".join(unknown_conv_like)
        raise ValueError(f"Residual descriptor references unknown convs: {joined}.")

    known = [name for name in names if name in conv_set]
    if known:
        return known
    if not allow_partial and names:
        raise ValueError(
            f"Residual descriptor references unknown convs: {', '.join(names)}."
        )
    return []


def _merge_overlapping(groups: list[set[str]]) -> list[set[str]]:
    """Merge overlapping residual groups inferred from descriptors."""

    merged: list[set[str]] = []
    for group in groups:
        merged_group = set(group)
        changed = True
        while changed:
            changed = False
            for existing in list(merged):
                if merged_group & existing:
                    merged.remove(existing)
                    merged_group |= existing
                    changed = True
        merged.append(merged_group)
    return merged


def _sort_groups(
    groups: list[set[str]], order_index: Mapping[str, int]
) -> list[set[str]]:
    """Sort groups deterministically by first occurrence in conv order."""

    def _key(group: set[str]) -> int:
        return min(order_index[name] for name in group)

    return sorted(groups, key=_key)


def _groups_from_nodes(
    nodes: Mapping[str, Any] | Sequence[Any], conv_names: Sequence[str]
) -> list[set[str]]:
    """Infer residual groups from a module graph with explicit add nodes."""

    if not nodes:
        return []
    conv_set = set(conv_names)
    order_index = {name: idx for idx, name in enumerate(conv_names)}
    node_iter: Sequence[Any]
    if isinstance(nodes, Mapping):
        node_iter = list(nodes.values())
    else:
        node_iter = list(nodes)

    groups: list[set[str]] = []
    for node in node_iter:
        if not isinstance(node, Mapping):
            continue
        op = str(
            node.get("type")
            or node.get("op")
            or node.get("kind")
            or node.get("node_type")
            or ""
        ).lower()
        if "add" not in op and op not in {"sum", "residual"}:
            continue
        raw_inputs = (
            node.get("inputs") or node.get("parents") or node.get("branches") or []
        )
        names = _as_name_list(raw_inputs)
        if "stream" in names:
            stream_id = node.get("stream_id")
            if not isinstance(stream_id, str):
                raise ValueError(
                    "Residual add node uses 'stream' but does not provide stream_id."
                )
            names.append(stream_id)
        conv_hits = _filter_conv_names(names, conv_set, allow_partial=True)
        if len(conv_hits) < 2:
            continue
        ordered = [name for name in conv_names if name in set(conv_hits)]
        groups.append(set(ordered))

    merged = _merge_overlapping(groups)
    return _sort_groups(merged, order_index)


def _infer_residual_groups(
    descriptor: Any, conv_names: Sequence[str]
) -> list[set[str]]:
    """Infer residual groups from a module graph descriptor."""

    data = _load_graph_descriptor(descriptor)
    if data is None:
        return []

    def _has_node_signature(value: Any) -> bool:
        return isinstance(value, Mapping) and any(
            key in value for key in ("type", "op", "kind", "node_type")
        )

    groups: list[set[str]] = []
    if isinstance(data, Mapping):
        nodes_payload: Mapping[str, Any] | Sequence[Any] | None = data.get("nodes")
        if nodes_payload is None:
            values = list(data.values())
            if values and all(isinstance(val, Mapping) for val in values):
                if any(_has_node_signature(val) for val in values):
                    nodes_payload = data
        if nodes_payload is None:
            raise ValueError("module_graph must define nodes.")
        groups.extend(_groups_from_nodes(nodes_payload, conv_names))

    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if not any(
            _has_node_signature(val) for val in data if isinstance(val, Mapping)
        ):
            raise ValueError("module_graph must define nodes.")
        groups.extend(_groups_from_nodes(data, conv_names))

    merged = _merge_overlapping(groups)
    order_index = {name: idx for idx, name in enumerate(conv_names)}
    return _sort_groups(merged, order_index)


@register_adapter
@dataclass
class ResNetAdapter(ArchitectureAdapter):
    """Adapter for ResNet-style conv nets (conv + FRN/BN)."""

    name: str = "resnet"
    layer_root: str = "core"
    batch_stats_root: str | None = "batch_stats"
    module_graph: Mapping[str, Any] | Sequence[Any] | str | None = None
    residual_joins: Sequence[tuple[str, str]] = dataclasses.field(default_factory=list)

    def _module_root(self, layer_root_path: tuple[str, ...]) -> tuple[str, ...]:
        if layer_root_path and layer_root_path[0] == "params":
            return layer_root_path[1:]
        return layer_root_path

    def _collect_modules(
        self, subtree: Mapping[str, Any]
    ) -> tuple[
        list[tuple[str, ...]],
        list[tuple[str, ...]],
        list[tuple[str, ...]],
        list[tuple[str, ...]],
    ]:
        conv_paths: list[tuple[str, ...]] = []
        frn_paths: list[tuple[str, ...]] = []
        bn_paths: list[tuple[str, ...]] = []
        dense_paths: list[tuple[str, ...]] = []

        def _walk(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
            for name, value in node.items():
                if not isinstance(value, Mapping):
                    continue
                path = prefix + (name,)
                if name.startswith("Conv_"):
                    conv_paths.append(path)
                elif name.startswith("FRN_"):
                    frn_paths.append(path)
                elif name.startswith("BatchNorm_"):
                    bn_paths.append(path)
                elif name.startswith("Dense_"):
                    dense_paths.append(path)
                else:
                    _walk(value, path)

        _walk(subtree, ())
        return conv_paths, frn_paths, bn_paths, dense_paths

    def _frn_name_for(self, conv_name: str, available: set[str]) -> str | None:
        parts = conv_name.split("/")
        suffix = parts[-1].split("_", 1)[-1]
        parts[-1] = f"FRN_{suffix}"
        candidate = "/".join(parts)
        return candidate if candidate in available else None

    def _bn_name_for(self, conv_name: str, available: set[str]) -> str | None:
        parts = conv_name.split("/")
        suffix = parts[-1].split("_", 1)[-1]
        parts[-1] = f"BatchNorm_{suffix}"
        candidate = "/".join(parts)
        return candidate if candidate in available else None

    def build_problem(self, params: Mapping[str, Any]) -> AlignmentProblem:
        """Derive the graph-native alignment problem directly from the modules."""

        layer_root_path = tuple(self.layer_root.split("."))
        subtree = _maybe_descend(params, layer_root_path)
        if subtree is None:
            raise ValueError(f"layer_root '{self.layer_root}' not found in params.")

        module_root = self._module_root(layer_root_path)
        conv_rel, frn_rel, bn_rel, dense_rel = self._collect_modules(subtree)
        if not conv_rel:
            raise ValueError("No Conv_* layers found for ResNetAdapter.")

        def _format_id(relative: tuple[str, ...]) -> str:
            parts = (*module_root, *relative) if module_root else relative
            return "/".join(parts)

        conv_names = [_format_id(path) for path in conv_rel]
        conv_paths = {
            conv_id: (*layer_root_path, *rel)
            for conv_id, rel in zip(conv_names, conv_rel, strict=False)
        }
        frn_paths = {_format_id(path): (*layer_root_path, *path) for path in frn_rel}
        bn_paths = {_format_id(path): (*layer_root_path, *path) for path in bn_rel}
        dense_rel = [path for path in dense_rel if len(path) == 1]
        dense_paths = {
            _format_id(path): (*layer_root_path, *path) for path in dense_rel
        }
        frn_ids = set(frn_paths)
        bn_ids = set(bn_paths)

        stats_root_path = (
            tuple(self.batch_stats_root.split(".")) if self.batch_stats_root else None
        )
        residual_groups = _infer_residual_groups(self.module_graph, conv_names)

        unions = _GroupUnionFind()
        order_index = {name: idx for idx, name in enumerate(conv_names)}
        for conv_name in conv_names:
            kernel = _descend(params, (*conv_paths[conv_name], "kernel"))
            unions.add(conv_name, int(kernel.shape[-1]))

        for group in residual_groups:
            names = sorted(group, key=order_index.__getitem__)
            for left, right in zip(names, names[1:], strict=False):
                unions.union(left, right)

        for left, right in self.residual_joins:
            if left not in unions.parent or right not in unions.parent:
                raise ValueError(
                    f"Residual join references unknown convs: {(left, right)}."
                )
            unions.union(left, right)

        group_map = {name: unions.find(name) for name in conv_names}
        groups: dict[str, PermutationGroup] = {
            root: PermutationGroup(id=root, size=size)
            for root, size in unions.roots().items()
        }

        tensors: dict[str, TensorSpec] = {}

        def _walk(node: Any, prefix: tuple[str, ...]) -> None:
            if isinstance(node, Mapping):
                for name, value in node.items():
                    _walk(value, prefix + (str(name),))
                return
            shape = tuple(int(dim) for dim in np.shape(node))
            if not shape:
                return
            tensor_id = "/".join(prefix)
            tensors[tensor_id] = TensorSpec(
                id=tensor_id,
                path=prefix,
                shape=shape,
                metadata={"architecture": self.name},
            )

        _walk(params, ())

        bindings: list[AxisBinding] = []
        seen_bindings: set[tuple[str, int, str]] = set()

        def _bind(path: tuple[str, ...], axis: int, group_id: str, role: str) -> str:
            tensor_id = "/".join(path)
            if tensor_id not in tensors:
                value = _descend(params, path)
                tensors[tensor_id] = TensorSpec(
                    id=tensor_id,
                    path=path,
                    shape=tuple(int(dim) for dim in np.shape(value)),
                    metadata={"architecture": self.name},
                )
            key = (tensor_id, axis, group_id)
            if key not in seen_bindings:
                seen_bindings.add(key)
                bindings.append(
                    AxisBinding(
                        tensor_id=tensor_id, axis=axis, group=group_id, role=role
                    )
                )
            return tensor_id

        group_order: list[str] = []
        conv_to_group: dict[str, str] = {}
        conv_to_out_tensors: dict[str, list[str]] = {}

        for idx, conv_name in enumerate(conv_names):
            group_id = group_map[conv_name]
            conv_to_group[conv_name] = group_id
            if group_id not in group_order:
                group_order.append(group_id)
            conv_path = conv_paths[conv_name]
            group_size = groups[group_id].size
            kernel_path = (*conv_path, "kernel")
            _validate_channel_size(
                kernel_path, _descend(params, kernel_path), group_size
            )

            out_tensors: list[str] = [
                _bind(kernel_path, -1, group_id, "out"),
                _bind((*conv_path, "bias"), 0, group_id, "out"),
            ]

            frn_name = self._frn_name_for(conv_name, frn_ids)
            bn_name = self._bn_name_for(conv_name, bn_ids)
            if frn_name and bn_name:
                raise ValueError(
                    f"Conv '{conv_name}' has both FRN ({frn_name}) and "
                    f"BatchNorm ({bn_name}) modules."
                )
            if frn_name:
                self._attach_frn_bindings(
                    params,
                    group_id,
                    group_size,
                    frn_paths[frn_name],
                    _bind,
                    out_tensors,
                )
            if bn_name:
                self._attach_batchnorm_bindings(
                    params,
                    group_id,
                    group_size,
                    bn_paths[bn_name],
                    _bind,
                    layer_root_path,
                    stats_root_path,
                    out_tensors,
                )

            conv_to_out_tensors[conv_name] = list(dict.fromkeys(out_tensors))

            next_conv = conv_names[idx + 1] if idx + 1 < len(conv_names) else None
            if next_conv:
                next_path = (*conv_paths[next_conv], "kernel")
                next_kernel = _descend(params, next_path)
                _validate_channel_size(next_path, next_kernel, group_size, axis=-2)
                _bind(next_path, -2, group_id, "in")

        dense_candidates = sorted(dense_paths, key=_natural_key)
        last_group = group_map[conv_names[-1]]
        for dense_name in dense_candidates:
            dense_path = (*dense_paths[dense_name], "kernel")
            kernel = _descend(params, dense_path)
            in_size = int(kernel.shape[0])
            if groups[last_group].size != in_size:
                raise ValueError(
                    f"Dense layer {dense_name} expects input {in_size}, "
                    f"but last conv group {last_group} has {groups[last_group].size}."
                )
            _bind(dense_path, 0, last_group, "in")

        constraints: list[GraphConstraint] = []

        def _residual_tie(members: tuple[str, ...], source: str) -> GraphConstraint:
            tie_groups = tuple(
                dict.fromkeys(
                    conv_to_group[member]
                    for member in members
                    if member in conv_to_group
                )
            )
            tie_tensors: list[str] = []
            for member in members:
                tie_tensors.extend(conv_to_out_tensors.get(member, ()))
            return GraphConstraint(
                kind="residual_tie",
                groups=tie_groups,
                tensors=tuple(dict.fromkeys(tie_tensors)),
                metadata={"members": members, "source": source},
            )

        for group in residual_groups:
            members = tuple(sorted(group, key=order_index.__getitem__))
            constraints.append(_residual_tie(members, "module_graph"))
        for left, right in self.residual_joins:
            constraints.append(_residual_tie((left, right), "manual"))

        problem = AlignmentProblem(
            groups=groups,
            tensors=tensors,
            axis_bindings=tuple(bindings),
            constraints=tuple(constraints),
            metadata={
                "architecture": self.name,
                "conv_names": conv_names,
                "dense_heads": dense_candidates,
                "group_order": group_order,
                "residual_groups": [sorted(group) for group in residual_groups],
                "manual_residual_joins": list(self.residual_joins),
            },
        )
        problem.validate(params)
        return problem

    def _attach_frn_bindings(
        self,
        params: Mapping[str, Any],
        group_id: str,
        group_size: int,
        frn_path: tuple[str, ...],
        bind,
        out_tensors: list[str],
    ) -> None:
        module = _descend(params, frn_path)
        for param in ("gamma", "beta", "tau"):
            if param in module:
                full_path = (*frn_path, param)
                _validate_channel_size(
                    full_path, _descend(params, full_path), group_size
                )
                out_tensors.append(bind(full_path, -1, group_id, "out"))
        if "eps" in module:
            eps_path = (*frn_path, "eps")
            eps_value = _descend(params, eps_path)
            if int(np.prod(np.shape(eps_value))) == group_size:
                out_tensors.append(bind(eps_path, -1, group_id, "out"))

    def _attach_batchnorm_bindings(
        self,
        params: Mapping[str, Any],
        group_id: str,
        group_size: int,
        bn_path: tuple[str, ...],
        bind,
        layer_root_path: tuple[str, ...],
        stats_root_path: tuple[str, ...] | None,
        out_tensors: list[str],
    ) -> None:
        module = _descend(params, bn_path)
        for param in ("scale", "bias"):
            if param in module:
                full_path = (*bn_path, param)
                _validate_channel_size(
                    full_path, _descend(params, full_path), group_size
                )
                out_tensors.append(bind(full_path, -1, group_id, "out"))

        if stats_root_path is None:
            return
        if not bn_path[: len(layer_root_path)] == layer_root_path:
            return
        bn_relative = bn_path[len(layer_root_path) :]
        stats_module = _maybe_descend(params, (*stats_root_path, *bn_relative))
        if not isinstance(stats_module, Mapping):
            return
        for stat in ("mean", "var"):
            if stat in stats_module:
                full_path = (*stats_root_path, *bn_relative, stat)
                _validate_channel_size(
                    full_path, _descend(params, full_path), group_size
                )
                out_tensors.append(bind(full_path, -1, group_id, "out"))
