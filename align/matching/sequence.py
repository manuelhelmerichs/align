"""Optimization schedules for graph-native alignment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import numpy as np

from ..symmetry import materialize_many
from .attention import (
    attention_module_specs,
    gqa_module_specs,
    update_attention_module,
    update_gqa_attention_module,
)
from .solvers import (
    LAPGroupSolver,
    ProcrustesGroupSolver,
    SinkhornSolver,
    SolverStep,
    _scheduled_groups,
)
from .state import TransformState


def _check_permute_only_folded(problem, *data_mappings) -> None:
    """Preflight for orthogonal groups with ``permute_only`` bindings.

    General orthogonal maps skip permute-only bindings (RMSNorm scales), which
    is exact only when the bound tensors are the folded constant 1. Enforced
    once per solve so a schedule on unfolded parameters fails loudly instead
    of silently corrupting the symmetry.
    """

    for binding in problem.axis_bindings:
        if binding.transform_scope != "permute_only":
            continue
        if problem.groups[binding.group].transform_family != "orthogonal":
            continue
        for data in data_mappings:
            tensor = np.asarray(data[binding.tensor_id])
            if float(np.max(np.abs(tensor - 1.0))) > 1e-3:
                raise ValueError(
                    f"Group {binding.group!r} declares the orthogonal transform "
                    f"class, but {binding.tensor_id!r} is not folded to 1 "
                    "(max deviation "
                    f"{float(np.max(np.abs(tensor - 1.0))):.3g}). Run scale "
                    "normalization (RMSNorm gamma folding) before rebasin, or "
                    "declare the stream as 'signed_permutation'."
                )


class SolverSequence:
    """Run an explicit sequence of hard and soft alignment solvers."""

    def __init__(self, objective, schedule: Sequence[SolverStep]) -> None:
        if not schedule:
            raise ValueError("Rebasin schedule must contain at least one solver step.")
        self.objective = objective
        self.schedule = tuple(schedule)
        for step in self.schedule:
            if step.solver not in {"lap", "procrustes", "sinkhorn"}:
                raise ValueError(f"Unknown rebasin solver {step.solver!r}.")

    @classmethod
    def from_config(cls, objective, schedule) -> SolverSequence:
        steps = [
            step if isinstance(step, SolverStep) else SolverStep.from_mapping(step)
            for step in schedule
        ]
        return cls(objective, steps)

    @property
    def prefers_gpu(self) -> bool:
        return any(step.solver == "sinkhorn" for step in self.schedule)

    @property
    def supports_batching(self) -> bool:
        return all(step.solver == "sinkhorn" for step in self.schedule)

    @property
    def backend(self) -> str:
        return "jax" if self.prefers_gpu else "numpy"

    def solve(
        self,
        problem,
        ref_data,
        target_data,
        *,
        rng_key=None,
        initial_state: TransformState | None = None,
    ) -> tuple[TransformState, dict[str, Any]]:
        state = initial_state or TransformState.identity(
            problem, backend="jax" if self.prefers_gpu else "numpy"
        )
        _check_permute_only_folded(problem, ref_data, target_data)
        aux_steps: list[dict[str, Any]] = []
        for index, step in enumerate(self.schedule):
            if step.solver == "lap":
                state, aux = self._run_lap_step(
                    problem, ref_data, target_data, state, step
                )
            elif step.solver == "procrustes":
                state, aux = self._run_procrustes_step(
                    problem, ref_data, target_data, state, step
                )
            elif step.solver == "sinkhorn":
                step_key = None
                if rng_key is not None:
                    step_key = jax.random.fold_in(rng_key, index)
                state, aux = SinkhornSolver(self.objective, step).solve(
                    problem,
                    ref_data,
                    target_data,
                    state,
                    rng_key=step_key,
                )
                state = _unbatch_state(state, sample_index=0)
            else:  # pragma: no cover - guarded in __init__
                raise ValueError(step.solver)
            aux_steps.append({"index": index, **aux})

        values = self.objective.value(problem, ref_data, target_data, state)
        aux_payload = {
            "objective": getattr(self.objective, "name", type(self.objective).__name__),
            "schedule": [step.to_dict() for step in self.schedule],
            "steps": aux_steps,
            "objective_final": float(np.ravel(np.asarray(values))[0]),
        }
        return state, aux_payload

    def solve_batch(
        self,
        problem,
        ref_data,
        target_params_batch,
        *,
        rng_key=None,
        backend: str | None = None,
    ) -> tuple[list[TransformState], list[dict[str, Any]]]:
        if not target_params_batch:
            return [], []
        if not self.supports_batching:
            results: list[TransformState] = []
            aux_payload: list[dict[str, Any]] = []
            for idx, params in enumerate(target_params_batch):
                data = problem.materialize(
                    params, backend=backend or self.backend, cache=False
                )
                sample_key = (
                    jax.random.fold_in(rng_key, idx) if rng_key is not None else None
                )
                state, aux = self.solve(
                    problem,
                    ref_data,
                    data,
                    rng_key=sample_key,
                )
                results.append(state)
                aux_payload.append(aux)
            return results, aux_payload

        target_data = materialize_many(
            problem, target_params_batch, backend=backend or self.backend
        )
        _check_permute_only_folded(problem, ref_data, target_data)
        state = TransformState.identity(problem, backend="jax")
        aux_steps: list[dict[str, Any]] = []
        for index, step in enumerate(self.schedule):
            step_key = (
                jax.random.fold_in(rng_key, index) if rng_key is not None else None
            )
            state, aux = SinkhornSolver(self.objective, step).solve(
                problem, ref_data, target_data, state, rng_key=step_key
            )
            aux_steps.append({"index": index, **aux})

        batch_size = len(target_params_batch)
        states = [_unbatch_state(state, sample_index=idx) for idx in range(batch_size)]
        losses = np.ravel(
            np.asarray(self.objective.value(problem, ref_data, target_data, state))
        )
        aux_batch: list[dict[str, Any]] = []
        for sample_idx in range(batch_size):
            sample_steps = []
            for step_aux in aux_steps:
                sample_aux = dict(step_aux)
                for key in ("loss_initial", "loss_final", "loss_history"):
                    value = sample_aux.get(key)
                    if isinstance(value, list) and len(value) == batch_size:
                        sample_aux[key] = value[sample_idx]
                sample_steps.append(sample_aux)
            aux_batch.append(
                {
                    "objective": getattr(
                        self.objective, "name", type(self.objective).__name__
                    ),
                    "schedule": [step.to_dict() for step in self.schedule],
                    "steps": sample_steps,
                    "objective_final": float(losses[sample_idx]),
                }
            )
        return states, aux_batch

    def _run_lap_step(self, problem, ref_data, target_data, state, step):
        groups = _scheduled_groups(problem, step)
        scheduled = set(groups)
        module_by_head = {
            spec.head_group: spec
            for spec in attention_module_specs(problem)
            if spec.head_group in scheduled
        }
        gqa_by_kv = {
            spec.kv_group: spec
            for spec in gqa_module_specs(problem)
            if spec.kv_group in scheduled
        }
        # Intra groups of a scheduled head/kv group are updated inside the
        # structured module update, not as standalone LAP groups.
        structured_intras = {
            group_id
            for spec in module_by_head.values()
            for group_id in (*spec.qk_groups, *spec.vo_groups)
            if group_id in scheduled
        } | {
            group_id
            for spec in gqa_by_kv.values()
            for group_id in (*spec.query_head_groups, *spec.vo_groups)
            if group_id in scheduled
        }
        solver = LAPGroupSolver(self.objective)
        sweeps = 0
        max_delta = 0.0
        for _ in range(step.max_sweeps):
            sweeps += 1
            sweep_delta = 0.0
            for group_id in groups:
                if group_id in structured_intras:
                    continue
                if group_id in module_by_head:
                    state, update_aux = update_attention_module(
                        problem,
                        self.objective,
                        ref_data,
                        target_data,
                        state,
                        module_by_head[group_id],
                        scheduled_groups=scheduled,
                    )
                elif group_id in gqa_by_kv:
                    state, update_aux = update_gqa_attention_module(
                        problem,
                        self.objective,
                        ref_data,
                        target_data,
                        state,
                        gqa_by_kv[group_id],
                        scheduled_groups=scheduled,
                    )
                else:
                    state, update_aux = solver.update(
                        problem, ref_data, target_data, state, group_id
                    )
                sweep_delta = max(sweep_delta, float(update_aux["delta"]))
            max_delta = sweep_delta
            if sweep_delta <= step.tol:
                break
        return state, {"solver": "lap", "sweeps": sweeps, "max_delta": max_delta}

    def _run_procrustes_step(self, problem, ref_data, target_data, state, step):
        scheduled = _scheduled_groups(problem, step)
        eligible = tuple(
            group_id
            for group_id in scheduled
            if problem.groups[group_id].transform_family == "orthogonal"
        )
        if step.groups is not None or step.components is not None:
            ineligible = sorted(set(scheduled) - set(eligible))
            if ineligible:
                raise ValueError(
                    "procrustes steps only update 'orthogonal' groups; "
                    "scheduled non-orthogonal group(s): " + ", ".join(ineligible)
                )
        if not eligible:
            raise ValueError(
                "procrustes step scheduled, but the problem declares no "
                "'orthogonal' groups."
            )
        solver = ProcrustesGroupSolver(self.objective)
        sweeps = 0
        max_delta = 0.0
        for _ in range(step.max_sweeps):
            sweeps += 1
            sweep_delta = 0.0
            for group_id in eligible:
                state, update_aux = solver.update(
                    problem, ref_data, target_data, state, group_id
                )
                sweep_delta = max(sweep_delta, float(update_aux["delta"]))
            max_delta = sweep_delta
            if sweep_delta <= step.tol:
                break
        return state, {
            "solver": "procrustes",
            "sweeps": sweeps,
            "max_delta": max_delta,
            "groups": list(eligible),
        }


def _unbatch_state(state: TransformState, *, sample_index: int) -> TransformState:
    matrices = {}
    for group_id, matrix in state.matrices.items():
        arr = np.asarray(matrix)
        matrices[group_id] = arr[sample_index] if arr.ndim == 3 else arr
    return TransformState(
        group_order=state.group_order,
        matrices=matrices,
        logits=None,
        metadata=dict(state.metadata),
    )


__all__ = ["SolverSequence"]
