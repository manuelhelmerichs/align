"""Residual ConvNet recipe driven by an explicit module DAG."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..symmetry import ResidualChannelTie, SymmetryGraph
from ..symmetry.tensor_ops import _canonical_axis, _descend, _maybe_descend
from ._utils import natural_key as _natural_key
from .graph_builder import SymmetryGraphBuilder
from .recipe import ArchitectureRecipe, register_recipe
from .rules import SymmetryRule
from .schemas import RESIDUAL_CONVNET_OPTIONS


class _GroupUnionFind:
    """Union-find that validates equal channel sizes."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, name: str, size: int) -> None:
        if name in self.parent:
            raise ValueError(f"Duplicate channel producer {name!r}.")
        self.parent[name] = name
        self.size[name] = size

    def find(self, name: str) -> str:
        parent = self.parent[name]
        if parent != name:
            self.parent[name] = self.find(parent)
        return self.parent[name]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] != self.size[right_root]:
            raise ValueError(
                "Residual add requires equal channel sizes: "
                f"{left!r} has {self.size[left_root]}, {right!r} has "
                f"{self.size[right_root]}."
            )
        # Union direction is deterministic and independent of descriptor or
        # parameter mapping order.
        canonical, other = sorted((left_root, right_root), key=_natural_key)
        self.parent[other] = canonical


@dataclass(frozen=True)
class _ModuleNode:
    id: str
    kind: str
    normalizer: str | None = None


@dataclass(frozen=True)
class _ModuleTopology:
    nodes: tuple[_ModuleNode, ...]
    edges: tuple[tuple[str, str], ...]

    @property
    def by_id(self) -> dict[str, _ModuleNode]:
        return {node.id: node for node in self.nodes}

    @property
    def incoming(self) -> dict[str, tuple[str, ...]]:
        values: dict[str, list[str]] = defaultdict(list)
        for source, target in self.edges:
            values[target].append(source)
        return {node.id: tuple(values[node.id]) for node in self.nodes}

    def topological_order(self) -> tuple[str, ...]:
        incoming_count = {node.id: 0 for node in self.nodes}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for source, target in self.edges:
            outgoing[source].append(target)
            incoming_count[target] += 1
        ready = sorted(
            (node_id for node_id, count in incoming_count.items() if count == 0),
            key=_natural_key,
        )
        ordered: list[str] = []
        while ready:
            node_id = ready.pop(0)
            ordered.append(node_id)
            for target in sorted(outgoing[node_id], key=_natural_key):
                incoming_count[target] -= 1
                if incoming_count[target] == 0:
                    ready.append(target)
                    ready.sort(key=_natural_key)
        if len(ordered) != len(self.nodes):
            cyclic = sorted(
                (node_id for node_id, count in incoming_count.items() if count),
                key=_natural_key,
            )
            raise ValueError("residual_topology contains a cycle: " + ", ".join(cyclic))
        return tuple(ordered)


def _strict_keys(label: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} field(s): {', '.join(unknown)}.")


def _load_topology(value: Any) -> _ModuleTopology:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"residual_topology path does not exist: {path}")
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"residual_topology must be valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("residual_topology must be a mapping or JSON file path.")
    _strict_keys("residual_topology", value, {"nodes", "edges", "metadata"})
    raw_nodes, raw_edges = value.get("nodes"), value.get("edges")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("residual_topology.nodes must be a non-empty list.")
    if not isinstance(raw_edges, list):
        raise ValueError("residual_topology.edges must be a list.")

    nodes: list[_ModuleNode] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"residual_topology.nodes[{index}] must be a mapping.")
        _strict_keys(
            f"residual_topology.nodes[{index}]", raw, {"id", "kind", "normalizer"}
        )
        node_id, kind = raw.get("id"), raw.get("kind")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"residual_topology.nodes[{index}].id must be non-empty.")
        if node_id in seen_ids:
            raise ValueError(f"Duplicate residual_topology node id {node_id!r}.")
        if kind not in {"input", "conv", "dense", "add"}:
            raise ValueError(
                f"residual_topology node {node_id!r} has invalid kind {kind!r}."
            )
        normalizer = raw.get("normalizer")
        if normalizer is not None and (
            kind not in {"conv", "add"}
            or not isinstance(normalizer, str)
            or not normalizer
        ):
            raise ValueError(
                f"Node {node_id!r} may declare a normalizer only for conv/add kinds."
            )
        nodes.append(_ModuleNode(node_id, kind, normalizer))
        seen_ids.add(node_id)

    edges: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_edges):
        if not isinstance(raw, Mapping):
            raise ValueError(f"residual_topology.edges[{index}] must be a mapping.")
        _strict_keys(f"residual_topology.edges[{index}]", raw, {"source", "target"})
        source, target = raw.get("source"), raw.get("target")
        if source not in seen_ids or target not in seen_ids:
            raise ValueError(
                f"residual_topology edge {index} references unknown nodes: "
                f"{source!r} -> {target!r}."
            )
        edge = (source, target)
        if source == target or edge in seen_edges:
            raise ValueError(f"Invalid duplicate/self edge {source!r} -> {target!r}.")
        edges.append(edge)
        seen_edges.add(edge)

    topology = _ModuleTopology(tuple(nodes), tuple(edges))
    incoming = topology.incoming
    for node in nodes:
        count = len(incoming[node.id])
        if node.kind == "input" and count != 0:
            raise ValueError(f"Input node {node.id!r} cannot have incoming edges.")
        if node.kind == "conv" and count != 1:
            raise ValueError(
                f"Conv node {node.id!r} must have exactly one incoming edge; use an input/add node."
            )
        if node.kind == "dense" and count != 1:
            raise ValueError(
                f"Dense node {node.id!r} must have exactly one incoming edge."
            )
        if node.kind == "add" and count < 2:
            raise ValueError(
                f"Add node {node.id!r} must have at least two incoming edges."
            )
    topology.topological_order()
    return topology


def _validate_channel_size(
    path: tuple[str, ...], array: Any, expected: int, *, axis: int = -1
) -> None:
    shape = np.shape(array)
    axis = _canonical_axis(len(shape), axis)
    if shape[axis] != expected:
        raise ValueError(
            f"{path} has channel dimension {shape[axis]}, expected {expected}."
        )


@dataclass
class ResidualConvNetRule(SymmetryRule):
    """Emit residual-channel actions from a validated module DAG."""

    kind: ClassVar[str] = "convnet"
    component_id: str = "features"
    parameter_root: str = "core"
    batch_stats_root: str | None = "batch_stats"
    residual_topology: Mapping[str, Any] | str | None = None
    linear_residual_free: bool = False

    @staticmethod
    def _module_root(parameter_root_path: tuple[str, ...]) -> tuple[str, ...]:
        return (
            parameter_root_path[1:]
            if parameter_root_path[:1] == ("params",)
            else parameter_root_path
        )

    def _discover_parameter_modules(
        self,
        subtree: Mapping[str, Any],
        parameter_root_path: tuple[str, ...],
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
        module_root = self._module_root(parameter_root_path)
        convs: dict[str, tuple[str, ...]] = {}
        dense: dict[str, tuple[str, ...]] = {}

        def walk(node: Mapping[str, Any], relative: tuple[str, ...]) -> None:
            if "kernel" in node and not isinstance(node["kernel"], Mapping):
                shape = np.shape(node["kernel"])
                module_id = "/".join((*module_root, *relative))
                path = (*parameter_root_path, *relative)
                if len(shape) >= 3:
                    convs[module_id] = path
                elif len(shape) == 2:
                    dense[module_id] = path
                return
            for name, value in node.items():
                if isinstance(value, Mapping):
                    walk(value, (*relative, str(name)))

        walk(subtree, ())
        if not convs:
            raise ValueError("No convolution modules found for ResidualConvNetRecipe.")
        return convs, dense

    def _linear_topology(
        self, convs: Mapping[str, Any], dense: Mapping[str, Any]
    ) -> _ModuleTopology:
        ordered_convs = sorted(convs, key=_natural_key)
        ordered_dense = sorted(dense, key=_natural_key)
        ordered = ordered_convs + ordered_dense
        nodes = (_ModuleNode("input", "input"),) + tuple(
            _ModuleNode(node_id, "conv" if node_id in convs else "dense")
            for node_id in ordered
        )
        conv_chain = ["input", *ordered_convs]
        edges = tuple(zip(conv_chain, conv_chain[1:], strict=False)) + tuple(
            (ordered_convs[-1], dense_id) for dense_id in ordered_dense
        )
        return _ModuleTopology(nodes, edges)

    def add_to(self, builder: SymmetryGraphBuilder) -> None:
        if not isinstance(self.linear_residual_free, bool):
            raise ValueError("linear_residual_free must be a bool.")
        if self.residual_topology is not None and self.linear_residual_free:
            raise ValueError(
                "Set residual_topology or linear_residual_free=true, not both."
            )
        if self.residual_topology is None and not self.linear_residual_free:
            raise ValueError(
                "ResidualConvNetRecipe requires residual_topology. For a strictly "
                "linear, residual-free conv stack, set linear_residual_free=true."
            )

        params = builder.params
        parameter_root_path = tuple(self.parameter_root.split("."))
        subtree = _maybe_descend(params, parameter_root_path)
        if not isinstance(subtree, Mapping):
            raise ValueError(
                f"parameter_root {self.parameter_root!r} not found in params."
            )
        conv_paths, dense_paths = self._discover_parameter_modules(
            subtree, parameter_root_path
        )
        topology = (
            _load_topology(self.residual_topology)
            if self.residual_topology is not None
            else self._linear_topology(conv_paths, dense_paths)
        )
        node_by_id = topology.by_id
        declared_convs = {node.id for node in topology.nodes if node.kind == "conv"}
        declared_dense = {node.id for node in topology.nodes if node.kind == "dense"}
        if declared_convs != set(conv_paths) or declared_dense != set(dense_paths):
            missing = sorted(
                (set(conv_paths) | set(dense_paths)) - set(node_by_id), key=_natural_key
            )
            unknown = sorted(
                (declared_convs | declared_dense)
                - (set(conv_paths) | set(dense_paths)),
                key=_natural_key,
            )
            raise ValueError(
                "residual_topology must account for every conv/dense parameter module "
                f"exactly once; missing={missing}, unknown={unknown}."
            )

        module_root = self._module_root(parameter_root_path)

        def module_path(module_id: str) -> tuple[str, ...]:
            parts = tuple(module_id.split("/"))
            if module_root and parts[: len(module_root)] != module_root:
                raise ValueError(
                    f"Module id {module_id!r} is outside parameter root "
                    f"{'/'.join(module_root)!r}."
                )
            relative = parts[len(module_root) :] if module_root else parts
            return (*parameter_root_path, *relative)

        normalizer_paths: dict[str, tuple[str, ...]] = {}
        for node in topology.nodes:
            if node.normalizer is None:
                continue
            path = module_path(node.normalizer)
            module = _maybe_descend(params, path)
            if not isinstance(module, Mapping):
                raise ValueError(
                    f"Normalizer {node.normalizer!r} for {node.id!r} was not found."
                )
            normalizer_paths[node.id] = path

        unions = _GroupUnionFind()
        for conv_id, path in conv_paths.items():
            kernel = _descend(params, (*path, "kernel"))
            unions.add(conv_id, int(np.shape(kernel)[-1]))

        incoming = topology.incoming
        source_cache: dict[str, frozenset[str]] = {}

        def source_convs(node_id: str) -> frozenset[str]:
            cached = source_cache.get(node_id)
            if cached is not None:
                return cached
            node = node_by_id[node_id]
            if node.kind == "conv":
                result = frozenset({node_id})
            elif node.kind == "input":
                result = frozenset()
            elif node.kind == "add":
                result = frozenset().union(
                    *(source_convs(source) for source in incoming[node_id])
                )
            else:
                raise ValueError(
                    f"Dense node {node_id!r} cannot feed a channel consumer."
                )
            source_cache[node_id] = result
            return result

        add_members: dict[str, tuple[str, ...]] = {}
        for node_id in topology.topological_order():
            if node_by_id[node_id].kind != "add":
                continue
            members = tuple(
                sorted(
                    frozenset().union(
                        *(source_convs(source) for source in incoming[node_id])
                    ),
                    key=_natural_key,
                )
            )
            for other in members[1:]:
                unions.union(members[0], other)
            add_members[node_id] = members

        root_members: dict[str, list[str]] = defaultdict(list)
        for conv_id in conv_paths:
            root_members[unions.find(conv_id)].append(conv_id)
        canonical_for_root = {
            root: sorted(members, key=_natural_key)[0]
            for root, members in root_members.items()
        }
        group_map = {
            conv_id: canonical_for_root[unions.find(conv_id)] for conv_id in conv_paths
        }
        for group_id in sorted(set(group_map.values()), key=_natural_key):
            builder.add_group(group_id, unions.size[unions.find(group_id)])

        def walk_tensors(node: Any, prefix: tuple[str, ...]) -> None:
            if isinstance(node, Mapping):
                for name, value in node.items():
                    walk_tensors(value, (*prefix, str(name)))
            elif np.shape(node):
                builder.tensor(prefix)

        walk_tensors(params, ())
        seen_bindings: set[tuple[str, int, str]] = set()

        def bind(
            path: tuple[str, ...],
            axis: int,
            group_id: str,
            role: str,
            scale_power: float = 1.0,
        ) -> str:
            tensor_id = "/".join(path)
            key = (tensor_id, axis, group_id)
            if key not in seen_bindings:
                builder.bind(path, axis, group_id, role=role, scale_power=scale_power)
                seen_bindings.add(key)
            return tensor_id

        conv_out_tensors: dict[str, list[str]] = {}
        stats_root_path = (
            tuple(self.batch_stats_root.split(".")) if self.batch_stats_root else None
        )
        normalized_by_add = {
            member
            for add_id, members in add_members.items()
            if add_id in normalizer_paths
            for member in members
        }
        for conv_id in sorted(conv_paths, key=_natural_key):
            path = conv_paths[conv_id]
            group_id = group_map[conv_id]
            group_size = builder.groups[group_id].size
            kernel_path = (*path, "kernel")
            normalizer_path = normalizer_paths.get(conv_id)
            scale_power = (
                0.0
                if normalizer_path is not None or conv_id in normalized_by_add
                else 1.0
            )
            out_tensors = [bind(kernel_path, -1, group_id, "out", scale_power)]
            module = _descend(params, path)
            if "bias" in module:
                bias_path = (*path, "bias")
                _validate_channel_size(bias_path, module["bias"], group_size, axis=0)
                out_tensors.append(bind(bias_path, 0, group_id, "out", scale_power))
            if normalizer_path is not None:
                normalizer = _descend(params, normalizer_path)
                if any(name in normalizer for name in ("gamma", "tau")):
                    self._attach_frn_bindings(
                        params, group_id, group_size, normalizer_path, bind, out_tensors
                    )
                elif any(name in normalizer for name in ("scale", "bias")):
                    self._attach_batchnorm_bindings(
                        params,
                        group_id,
                        group_size,
                        normalizer_path,
                        bind,
                        parameter_root_path,
                        stats_root_path,
                        out_tensors,
                    )
                else:
                    raise ValueError(
                        f"Normalizer {node_by_id[conv_id].normalizer!r} is neither FRN nor BatchNorm."
                    )
            conv_out_tensors[conv_id] = list(dict.fromkeys(out_tensors))

        add_out_tensors: dict[str, list[str]] = defaultdict(list)
        for add_id, members in add_members.items():
            normalizer_path = normalizer_paths.get(add_id)
            if normalizer_path is None:
                continue
            group_id = group_map[members[0]]
            group_size = builder.groups[group_id].size
            normalizer = _descend(params, normalizer_path)
            if any(name in normalizer for name in ("gamma", "tau")):
                self._attach_frn_bindings(
                    params,
                    group_id,
                    group_size,
                    normalizer_path,
                    bind,
                    add_out_tensors[add_id],
                )
            elif any(name in normalizer for name in ("scale", "bias")):
                self._attach_batchnorm_bindings(
                    params,
                    group_id,
                    group_size,
                    normalizer_path,
                    bind,
                    parameter_root_path,
                    stats_root_path,
                    add_out_tensors[add_id],
                )
            else:
                raise ValueError(
                    f"Normalizer {node_by_id[add_id].normalizer!r} is neither FRN nor BatchNorm."
                )

        for node in topology.nodes:
            if node.kind not in {"conv", "dense"} or not incoming[node.id]:
                continue
            source = incoming[node.id][0]
            producers = source_convs(source)
            if not producers:
                # External model input has no modeled channel action.
                continue
            source_groups = {group_map[producer] for producer in producers}
            if len(source_groups) != 1:
                raise ValueError(
                    f"Consumer {node.id!r} receives incompatible channel groups "
                    f"from {source!r}: {sorted(source_groups)}."
                )
            group_id = next(iter(source_groups))
            path = conv_paths[node.id] if node.kind == "conv" else dense_paths[node.id]
            kernel_path = (*path, "kernel")
            axis = -2 if node.kind == "conv" else 0
            _validate_channel_size(
                kernel_path,
                _descend(params, kernel_path),
                builder.groups[group_id].size,
                axis=axis,
            )
            bind(kernel_path, axis, group_id, "in")

        for add_id, members in add_members.items():
            group_id = group_map[members[0]]
            tensors = tuple(
                dict.fromkeys(
                    [
                        tensor
                        for member in members
                        for tensor in conv_out_tensors[member]
                    ]
                    + add_out_tensors[add_id]
                )
            )
            builder.add_constraint(
                ResidualChannelTie(
                    groups=(group_id,),
                    tensors=tensors,
                    members=members,
                    source=add_id,
                )
            )

        builder.add_component(self.component_id, self.kind, tuple(builder.group_order))
        builder.metadata.update(
            {
                "conv_names": sorted(conv_paths, key=_natural_key),
                "dense_heads": sorted(dense_paths, key=_natural_key),
                "module_edges": [list(edge) for edge in topology.edges],
                "residual_adds": {
                    add_id: list(members) for add_id, members in add_members.items()
                },
            }
        )

    def _attach_frn_bindings(
        self, params, group_id, group_size, frn_path, bind, out_tensors
    ) -> None:
        module = _descend(params, frn_path)
        for param in ("gamma", "beta", "tau"):
            if param in module:
                path = (*frn_path, param)
                _validate_channel_size(path, _descend(params, path), group_size)
                out_tensors.append(bind(path, -1, group_id, "out"))
        if "eps" in module:
            path = (*frn_path, "eps")
            if int(np.prod(np.shape(_descend(params, path)))) == group_size:
                out_tensors.append(bind(path, -1, group_id, "out", 0.0))

    def _attach_batchnorm_bindings(
        self,
        params,
        group_id,
        group_size,
        bn_path,
        bind,
        parameter_root_path,
        stats_root_path,
        out_tensors,
    ) -> None:
        module = _descend(params, bn_path)
        for param in ("scale", "bias"):
            if param in module:
                path = (*bn_path, param)
                _validate_channel_size(path, _descend(params, path), group_size)
                out_tensors.append(bind(path, -1, group_id, "out"))
        if (
            stats_root_path is None
            or bn_path[: len(parameter_root_path)] != parameter_root_path
        ):
            return
        relative = bn_path[len(parameter_root_path) :]
        stats_module = _maybe_descend(params, (*stats_root_path, *relative))
        if not isinstance(stats_module, Mapping):
            return
        for stat in ("mean", "var"):
            if stat in stats_module:
                path = (*stats_root_path, *relative, stat)
                _validate_channel_size(path, _descend(params, path), group_size)
                out_tensors.append(bind(path, -1, group_id, "out", 0.0))


@register_recipe
@dataclass
class ResidualConvNetRecipe(ArchitectureRecipe):
    """Recipe for residual conv nets with explicit dataflow topology."""

    name: str = "residual_convnet"
    parameter_root: str = "core"
    batch_stats_root: str | None = "batch_stats"
    config_options: ClassVar[frozenset[str]] = RESIDUAL_CONVNET_OPTIONS
    residual_topology: Mapping[str, Any] | str | None = None
    linear_residual_free: bool = False

    def build_graph(self, params: Mapping[str, Any]) -> SymmetryGraph:
        builder = SymmetryGraphBuilder(params, architecture=self.name)
        ResidualConvNetRule(
            parameter_root=self.parameter_root,
            batch_stats_root=self.batch_stats_root,
            residual_topology=self.residual_topology,
            linear_residual_free=self.linear_residual_free,
        ).add_to(builder)
        return builder.finish()
