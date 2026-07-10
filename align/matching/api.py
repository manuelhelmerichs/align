"""Matching API surface backed by the graph-native alignment core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax

from ..samples import ParamTree
from ..symmetry import SymmetryGraph, extract_component_graph, match_component_tensors
from .objectives import get_objective
from .sequence import SolverSequence
from .solvers import SolverStep
from .state import TransformState


def default_lap_schedule(
    max_sweeps: int = 25, tolerance: float = 0.0
) -> tuple[SolverStep, ...]:
    """Return the default weight-matching-equivalent LAP schedule."""

    return (SolverStep(solver="lap", max_sweeps=max_sweeps, tolerance=tolerance),)


def build_solver_sequence(
    *,
    objective: str = "euclidean",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[SolverStep | Mapping[str, Any]] | None = None,
) -> SolverSequence:
    """Construct an objective and solver sequence from config-like values."""

    objective_obj = get_objective(objective, **dict(objective_kwargs or {}))
    steps = schedule or default_lap_schedule()
    return SolverSequence.from_steps(objective_obj, steps)


def match_sample(
    graph: SymmetryGraph,
    reference_params: ParamTree,
    params: ParamTree,
    *,
    solver_sequence: SolverSequence | None = None,
    objective: str = "euclidean",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[SolverStep | Mapping[str, Any]] | None = None,
    rng_key: jax.Array | None = None,
    reference_data: Mapping[str, Any] | None = None,
    reference_backend: str | None = None,
    is_reference: bool = False,
) -> tuple[ParamTree, Mapping[str, Any], dict[str, Any] | None]:
    """Match one target parameter tree using a ``SymmetryGraph``."""

    solver_sequence = solver_sequence or build_solver_sequence(
        objective=objective,
        objective_kwargs=objective_kwargs,
        schedule=schedule,
    )
    backend = reference_backend or solver_sequence.backend
    if reference_data is None:
        reference_data = graph.materialize(
            reference_params, backend=backend, cache=True
        )

    if is_reference:
        state = TransformState.identity(graph, backend=backend)
        return params, state.to_artifacts(), {"reference": True}

    target_data = graph.materialize(params, backend=backend, cache=False)
    state, aux_info = solver_sequence.solve(
        graph,
        reference_data,
        target_data,
        rng_key=rng_key,
    )
    aligned_params = graph.apply_transforms(params, state)
    return aligned_params, state.to_artifacts(), aux_info


def match_batch(
    graph: SymmetryGraph,
    reference_params: ParamTree,
    params_batch: Sequence[ParamTree],
    *,
    solver_sequence: SolverSequence | None = None,
    objective: str = "euclidean",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[SolverStep | Mapping[str, Any]] | None = None,
    rng_key: jax.Array | None = None,
    reference_data: Mapping[str, Any] | None = None,
    reference_backend: str | None = None,
) -> list[tuple[ParamTree, Mapping[str, Any], dict[str, Any] | None]]:
    """Match a sequence of target parameter trees."""

    if not params_batch:
        return []
    solver_sequence = solver_sequence or build_solver_sequence(
        objective=objective,
        objective_kwargs=objective_kwargs,
        schedule=schedule,
    )
    backend = reference_backend or solver_sequence.backend
    if reference_data is None:
        reference_data = graph.materialize(
            reference_params, backend=backend, cache=True
        )

    states, aux_batch = solver_sequence.solve_batch(
        graph,
        reference_data,
        params_batch,
        rng_key=rng_key,
        backend=backend,
    )
    results = []
    for params, state, aux in zip(params_batch, states, aux_batch, strict=True):
        aligned_params = graph.apply_transforms(params, state)
        results.append((aligned_params, state.to_artifacts(), aux))
    return results


def match_component_across(
    reference_graph: SymmetryGraph,
    reference_params: ParamTree,
    reference_component: str,
    target_graph: SymmetryGraph,
    target_params: ParamTree,
    target_component: str,
    *,
    solver_sequence: SolverSequence | None = None,
    objective: str = "euclidean",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[SolverStep | Mapping[str, Any]] | None = None,
    rng_key: jax.Array | None = None,
) -> tuple[ParamTree, Mapping[str, Any], dict[str, Any] | None]:
    """Align one component of ``target_params`` against the same component of another net.

    The two components may come from different architectures; they only need
    structurally identical component subgraphs (see
    :func:`align.symmetry.match_component_tensors`). Only the target component's groups
    are transformed, so the returned tree is function-equivalent to
    ``target_params``.
    """

    reference_subgraph = extract_component_graph(reference_graph, [reference_component])
    target_subgraph = extract_component_graph(target_graph, [target_component])
    tensor_map, group_map = match_component_tensors(
        reference_subgraph, reference_component, target_subgraph, target_component
    )

    solver_sequence = solver_sequence or build_solver_sequence(
        objective=objective,
        objective_kwargs=objective_kwargs,
        schedule=schedule,
    )
    backend = solver_sequence.backend
    reference_tensors = reference_subgraph.materialize(
        reference_params, backend=backend, cache=True
    )
    reference_data = {
        tensor_map[reference_tensor_id]: reference_tensors[reference_tensor_id]
        for reference_tensor_id in reference_subgraph.tensors
    }
    target_data = target_subgraph.materialize(
        target_params, backend=backend, cache=False
    )
    state, aux_info = solver_sequence.solve(
        target_subgraph,
        reference_data,
        target_data,
        rng_key=rng_key,
    )
    aligned_params = target_subgraph.apply_transforms(target_params, state)
    aux_payload = dict(aux_info or {})
    aux_payload["reference_component"] = reference_component
    aux_payload["target_component"] = target_component
    aux_payload["group_map"] = dict(group_map)
    return aligned_params, state.to_artifacts(), aux_payload


__all__ = [
    "build_solver_sequence",
    "default_lap_schedule",
    "match_batch",
    "match_component_across",
    "match_sample",
]
