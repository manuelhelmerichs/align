"""Rebasin API surface backed by the graph-native alignment core."""

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
    max_sweeps: int = 25, tol: float = 0.0
) -> tuple[SolverStep, ...]:
    """Return the default weight-matching-equivalent LAP schedule."""

    return (SolverStep(solver="lap", max_sweeps=max_sweeps, tol=tol),)


def build_solver_sequence(
    *,
    objective: str = "l2_weight",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[SolverStep | Mapping[str, Any]] | None = None,
) -> SolverSequence:
    """Construct an objective and scheduler from config-like values."""

    objective_obj = get_objective(objective, **dict(objective_kwargs or {}))
    steps = schedule or default_lap_schedule()
    return SolverSequence.from_config(objective_obj, steps)


def match_sample(
    problem: SymmetryGraph,
    ref_params: ParamTree,
    params: ParamTree,
    *,
    scheduler: SolverSequence | None = None,
    objective: str = "l2_weight",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[SolverStep | Mapping[str, Any]] | None = None,
    rng_key: jax.Array | None = None,
    ref_data: Mapping[str, Any] | None = None,
    ref_backend: str | None = None,
    is_reference: bool = False,
) -> tuple[ParamTree, Mapping[str, Any], dict[str, Any] | None]:
    """Rebasin one target parameter tree using an ``SymmetryGraph``."""

    scheduler = scheduler or build_solver_sequence(
        objective=objective,
        objective_kwargs=objective_kwargs,
        schedule=schedule,
    )
    backend = ref_backend or scheduler.backend
    if ref_data is None:
        ref_data = problem.materialize(ref_params, backend=backend, cache=True)

    if is_reference:
        state = TransformState.identity(problem, backend=backend)
        return params, state.to_artifacts(), {"reference": True}

    target_data = problem.materialize(params, backend=backend, cache=False)
    state, aux_info = scheduler.solve(
        problem,
        ref_data,
        target_data,
        rng_key=rng_key,
    )
    aligned_params = problem.apply(params, state)
    return aligned_params, state.to_artifacts(), aux_info


def match_batch(
    problem: SymmetryGraph,
    ref_params: ParamTree,
    params_batch: Sequence[ParamTree],
    *,
    scheduler: SolverSequence | None = None,
    objective: str = "l2_weight",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[SolverStep | Mapping[str, Any]] | None = None,
    rng_key: jax.Array | None = None,
    ref_data: Mapping[str, Any] | None = None,
    ref_backend: str | None = None,
) -> list[tuple[ParamTree, Mapping[str, Any], dict[str, Any] | None]]:
    """Rebasin a sequence of target parameter trees."""

    if not params_batch:
        return []
    scheduler = scheduler or build_solver_sequence(
        objective=objective,
        objective_kwargs=objective_kwargs,
        schedule=schedule,
    )
    backend = ref_backend or scheduler.backend
    if ref_data is None:
        ref_data = problem.materialize(ref_params, backend=backend, cache=True)

    states, aux_batch = scheduler.solve_batch(
        problem,
        ref_data,
        params_batch,
        rng_key=rng_key,
        backend=backend,
    )
    results = []
    for params, state, aux in zip(params_batch, states, aux_batch, strict=True):
        aligned_params = problem.apply(params, state)
        results.append((aligned_params, state.to_artifacts(), aux))
    return results


def match_component_across(
    ref_problem: SymmetryGraph,
    ref_params: ParamTree,
    ref_block: str,
    target_problem: SymmetryGraph,
    target_params: ParamTree,
    target_block: str,
    *,
    scheduler: SolverSequence | None = None,
    objective: str = "l2_weight",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[SolverStep | Mapping[str, Any]] | None = None,
    rng_key: jax.Array | None = None,
) -> tuple[ParamTree, Mapping[str, Any], dict[str, Any] | None]:
    """Align one component of ``target_params`` against the same component of another net.

    The two components may come from different architectures; they only need
    structurally identical component sub-problems (see
    :func:`align.symmetry.match_component_tensors`). Only the target component's groups
    are permuted, so the returned tree is function-equivalent to
    ``target_params``.
    """

    ref_sub = extract_component_graph(ref_problem, [ref_block])
    target_sub = extract_component_graph(target_problem, [target_block])
    tensor_map, group_map = match_component_tensors(
        ref_sub, ref_block, target_sub, target_block
    )

    scheduler = scheduler or build_solver_sequence(
        objective=objective,
        objective_kwargs=objective_kwargs,
        schedule=schedule,
    )
    backend = scheduler.backend
    ref_data_own = ref_sub.materialize(ref_params, backend=backend, cache=True)
    ref_data = {tensor_map[ref_id]: ref_data_own[ref_id] for ref_id in ref_sub.tensors}
    target_data = target_sub.materialize(target_params, backend=backend, cache=False)
    state, aux_info = scheduler.solve(
        target_sub,
        ref_data,
        target_data,
        rng_key=rng_key,
    )
    aligned_params = target_sub.apply(target_params, state)
    aux_payload = dict(aux_info or {})
    aux_payload["ref_block"] = ref_block
    aux_payload["target_block"] = target_block
    aux_payload["group_map"] = dict(group_map)
    return aligned_params, state.to_artifacts(), aux_payload


__all__ = [
    "build_solver_sequence",
    "default_lap_schedule",
    "match_batch",
    "match_component_across",
    "match_sample",
]
