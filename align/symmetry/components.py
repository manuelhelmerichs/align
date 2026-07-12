"""Component-level views of symmetry graphs.

Components partition a graph's symmetry groups into named selection units (the FCN
stack, one attention module, ...). This module derives component listings, extracts
self-contained component subgraphs, and matches structurally identical components
across different networks so they can be aligned against each other.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from typing import Any

import numpy as np

from .constraints import MHACircuitConstraint
from .graph import SymmetryGraph
from .tensor_ops import _canonical_axis, binding_axis_intervals


def tensors_for_component(graph: SymmetryGraph, component_id: str) -> tuple[str, ...]:
    """Return tensor ids bound to any of the component's groups, in graph order."""

    component = graph.components[component_id]
    members = set(component.groups)
    return tuple(
        tensor_id
        for tensor_id in graph.tensors
        if any(
            binding.group in members for binding in graph.bindings_for_tensor(tensor_id)
        )
    )


def resolve_component_patterns(
    graph: SymmetryGraph, patterns: Sequence[str]
) -> tuple[str, ...]:
    """Resolve fnmatch ``patterns`` to component ids; every pattern must match."""

    if not graph.components:
        raise ValueError(
            "Graph declares no components; component selectors cannot be resolved."
        )
    resolved: list[str] = []
    for pattern in patterns:
        matches = [
            component_id
            for component_id in graph.components
            if fnmatch.fnmatchcase(component_id, pattern)
        ]
        if not matches:
            available = ", ".join(graph.components)
            raise ValueError(
                f"Component pattern {pattern!r} matches no components. Available: {available}"
            )
        for component_id in matches:
            if component_id not in resolved:
                resolved.append(component_id)
    return tuple(resolved)


def groups_for_components(
    graph: SymmetryGraph, patterns: Sequence[str]
) -> tuple[str, ...]:
    """Resolve component ``patterns`` to the union of their groups, in group order."""

    component_ids = resolve_component_patterns(graph, patterns)
    members = {
        group_id
        for component_id in component_ids
        for group_id in graph.components[component_id].groups
    }
    return tuple(gid for gid in graph.group_order if gid in members)


def describe_symmetry(graph: SymmetryGraph) -> dict[str, Any]:
    """Return a JSON-friendly listing of the graph's components.

    Graphs without declared components are reported as one implicit component ``all``
    so the listing is uniform.
    """

    repeated = graph.repeated_group_terms()
    attention_groups = {
        group_id
        for constraint in graph.constraints
        if isinstance(constraint, MHACircuitConstraint)
        for group_id in constraint.groups
    }

    def _component_entry(
        component_id: str, kind: str, group_ids: Sequence[str]
    ) -> dict[str, Any]:
        tensor_ids = sorted(
            {
                binding.tensor_id
                for binding in graph.axis_bindings
                if binding.group in set(group_ids)
            }
        )
        param_count = int(
            sum(
                int(np.prod(graph.tensors[tensor_id].shape)) for tensor_id in tensor_ids
            )
        )
        notes: list[str] = []
        if any(group_id in repeated for group_id in group_ids):
            notes.append("repeated-group terms: exact LAP unavailable, use sinkhorn")
        if any(group_id in attention_groups for group_id in group_ids):
            notes.append("attention-coupled: lap uses the structured head update")
        return {
            "id": component_id,
            "kind": kind,
            "groups": [
                {"id": group_id, "size": int(graph.groups[group_id].size)}
                for group_id in group_ids
            ],
            "num_tensors": len(tensor_ids),
            "num_params": param_count,
            "tensors": tensor_ids,
            "notes": notes,
        }

    if graph.components:
        components = [
            _component_entry(
                component.id,
                component.kind,
                [gid for gid in graph.group_order if gid in set(component.groups)],
            )
            for component in graph.components.values()
        ]
    else:
        components = [_component_entry("all", "unstructured", list(graph.group_order))]
    return {
        "architecture": graph.metadata.get("architecture"),
        "num_groups": len(graph.groups),
        "num_tensors": len(graph.tensors),
        "components": components,
    }


def format_symmetry_description(graph: SymmetryGraph) -> str:
    """Render :func:`describe_symmetry` as human-readable text."""

    description = describe_symmetry(graph)
    lines = [
        f"architecture: {description['architecture']}",
        f"groups: {description['num_groups']}  tensors: {description['num_tensors']}",
        "components:",
    ]
    for component in description["components"]:
        group_summary = ", ".join(
            f"{group['id']}({group['size']})" for group in component["groups"]
        )
        lines.append(
            f"  {component['id']} [{component['kind']}]"
            f"  groups: {group_summary}"
            f"  tensors: {component['num_tensors']}"
            f"  params: {component['num_params']}"
        )
        for note in component["notes"]:
            lines.append(f"    note: {note}")
    return "\n".join(lines)


def extract_component_graph(
    graph: SymmetryGraph, component_ids: Sequence[str]
) -> SymmetryGraph:
    """Return the self-contained subgraph covering ``component_ids``.

    The subgraph keeps only the selected components' groups, their bindings, and
    the tensors those bindings touch. Constraints referencing dropped groups or
    tensors are dropped. Aligning the subgraph transforms only internal units of
    the selected components, so it stays function-preserving for the full network.
    """

    unknown = sorted(set(component_ids) - set(graph.components))
    if unknown:
        raise ValueError("Unknown component id(s): " + ", ".join(unknown))
    if not component_ids:
        raise ValueError("extract_component_graph requires at least one component id.")

    kept_components = {
        component_id: graph.components[component_id] for component_id in component_ids
    }
    kept_groups = {
        group_id
        for component in kept_components.values()
        for group_id in component.groups
    }
    bindings = tuple(
        binding for binding in graph.axis_bindings if binding.group in kept_groups
    )
    kept_tensor_ids = {binding.tensor_id for binding in bindings}
    constraints = tuple(
        constraint
        for constraint in graph.constraints
        if set(constraint.groups) <= kept_groups
        and set(constraint.tensors) <= kept_tensor_ids
    )
    metadata: dict[str, Any] = {
        "architecture": graph.metadata.get("architecture"),
        "group_order": [gid for gid in graph.group_order if gid in kept_groups],
        "extracted_components": list(component_ids),
    }
    subgraph = SymmetryGraph(
        groups={gid: graph.groups[gid] for gid in kept_groups},
        tensors={tid: graph.tensors[tid] for tid in kept_tensor_ids},
        axis_bindings=bindings,
        constraints=constraints,
        components=kept_components,
        metadata=metadata,
    )
    subgraph.validate()
    return subgraph


def _binding_profile(
    graph: SymmetryGraph,
    tensor_id: str,
    group_index: dict[str, int],
) -> tuple[Any, ...]:
    """Canonical per-tensor profile: shape + component-group binding structure."""

    tensor = graph.tensors[tensor_id]
    entries = []
    for binding in graph.bindings_for_tensor(tensor_id):
        if binding.group not in group_index:
            continue
        selector = tuple(
            sorted(
                (_canonical_axis(len(tensor.shape), sel_axis), sel_index)
                for sel_axis, sel_index in getattr(binding, "selector", ())
            )
        )
        for axis, start, stop in binding_axis_intervals(tensor.shape, binding):
            entries.append(
                (
                    axis,
                    start,
                    stop,
                    binding.role,
                    binding.scale_power,
                    binding.transform_scope,
                    group_index[binding.group],
                    selector,
                )
            )
    return (tuple(tensor.shape), tuple(sorted(entries)))


def component_signature(graph: SymmetryGraph, component_id: str) -> tuple[Any, ...]:
    """Canonical structural signature of one component.

    Two components with equal signatures have identical group sizes and identically
    bound tensors (shapes, axes, roles), so their parameters can be aligned
    against each other regardless of the surrounding architecture.
    """

    component = graph.components[component_id]
    ordered_groups = [gid for gid in graph.group_order if gid in set(component.groups)]
    group_index = {gid: idx for idx, gid in enumerate(ordered_groups)}
    profiles = sorted(
        _binding_profile(graph, tensor_id, group_index)
        for tensor_id in tensors_for_component(graph, component_id)
    )
    return (
        component.kind,
        tuple(
            (int(graph.groups[gid].size), graph.groups[gid].transform_family)
            for gid in ordered_groups
        ),
        tuple(profiles),
    )


def match_component_tensors(
    reference_graph: SymmetryGraph,
    reference_component: str,
    target_graph: SymmetryGraph,
    target_component: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Match tensors and groups of two structurally identical components.

    Returns ``(tensor_map, group_map)`` mapping reference ids to target ids.
    Raises if the component signatures differ or the correspondence is ambiguous.
    """

    reference_signature = component_signature(reference_graph, reference_component)
    target_signature = component_signature(target_graph, target_component)
    if reference_signature != target_signature:
        raise ValueError(
            f"Component {reference_component!r} and {target_component!r} have incompatible "
            "structures; cannot align across networks."
        )

    reference_groups = [
        gid
        for gid in reference_graph.group_order
        if gid in set(reference_graph.components[reference_component].groups)
    ]
    target_groups = [
        gid
        for gid in target_graph.group_order
        if gid in set(target_graph.components[target_component].groups)
    ]
    group_map = dict(zip(reference_groups, target_groups, strict=True))

    reference_index = {gid: idx for idx, gid in enumerate(reference_groups)}
    target_index = {gid: idx for idx, gid in enumerate(target_groups)}

    def _profiles(graph, component_id, group_index):
        profiles: dict[tuple[Any, ...], list[str]] = {}
        for tensor_id in tensors_for_component(graph, component_id):
            profile = _binding_profile(graph, tensor_id, group_index)
            profiles.setdefault(profile, []).append(tensor_id)
        return profiles

    reference_profiles = _profiles(
        reference_graph, reference_component, reference_index
    )
    target_profiles = _profiles(target_graph, target_component, target_index)

    tensor_map: dict[str, str] = {}
    for profile, reference_ids in reference_profiles.items():
        target_ids = target_profiles.get(profile, [])
        if len(reference_ids) != 1 or len(target_ids) != 1:
            raise ValueError(
                f"Ambiguous tensor correspondence between components {reference_component!r} "
                f"and {target_component!r}: profile {profile!r} matches "
                f"{reference_ids} vs {target_ids}."
            )
        tensor_map[reference_ids[0]] = target_ids[0]
    return tensor_map, group_map


__all__ = [
    "component_signature",
    "describe_symmetry",
    "extract_component_graph",
    "format_symmetry_description",
    "groups_for_components",
    "match_component_tensors",
    "resolve_component_patterns",
    "tensors_for_component",
]
