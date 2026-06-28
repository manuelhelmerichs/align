"""Optimization schedules for graph-native alignment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import numpy as np

from ..alignment import materialize_many
from .permutation_state import PermutationState
from .solvers import (
    LAPGroupSolver,
    SinkhornBlockSolver,
    SolverScheduleStep,
    _scheduled_groups,
)


class SolverScheduler:
    """Run an explicit sequence of hard and soft alignment solvers."""

    def __init__(self, objective, schedule: Sequence[SolverScheduleStep]) -> None:
        if not schedule:
            raise ValueError("Rebasin schedule must contain at least one solver step.")
        self.objective = objective
        self.schedule = tuple(schedule)
        for step in self.schedule:
            if step.solver not in {"lap", "sinkhorn"}:
                raise ValueError(f"Unknown rebasin solver {step.solver!r}.")

    @classmethod
    def from_config(cls, objective, schedule) -> SolverScheduler:
        steps = [
            step
            if isinstance(step, SolverScheduleStep)
            else SolverScheduleStep.from_mapping(step)
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
        initial_state: PermutationState | None = None,
    ) -> tuple[PermutationState, dict[str, Any]]:
        state = initial_state or PermutationState.identity(
            problem, backend="jax" if self.prefers_gpu else "numpy"
        )
        aux_steps: list[dict[str, Any]] = []
        for index, step in enumerate(self.schedule):
            if step.solver == "lap":
                state, aux = self._run_lap_step(
                    problem, ref_data, target_data, state, step
                )
            elif step.solver == "sinkhorn":
                step_key = None
                if rng_key is not None:
                    step_key = jax.random.fold_in(rng_key, index)
                state, aux = SinkhornBlockSolver(self.objective, step).solve(
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
    ) -> tuple[list[PermutationState], list[dict[str, Any]]]:
        if not target_params_batch:
            return [], []
        if not self.supports_batching:
            results: list[PermutationState] = []
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
        state = PermutationState.identity(problem, backend="jax")
        aux_steps: list[dict[str, Any]] = []
        for index, step in enumerate(self.schedule):
            step_key = (
                jax.random.fold_in(rng_key, index) if rng_key is not None else None
            )
            state, aux = SinkhornBlockSolver(self.objective, step).solve(
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
        solver = LAPGroupSolver(self.objective)
        sweeps = 0
        max_delta = 0.0
        for _ in range(step.max_sweeps):
            sweeps += 1
            sweep_delta = 0.0
            for group_id in groups:
                state, update_aux = solver.update(
                    problem, ref_data, target_data, state, group_id
                )
                sweep_delta = max(sweep_delta, float(update_aux["delta"]))
            max_delta = sweep_delta
            if sweep_delta <= step.tol:
                break
        return state, {"solver": "lap", "sweeps": sweeps, "max_delta": max_delta}


def _unbatch_state(state: PermutationState, *, sample_index: int) -> PermutationState:
    hard = {}
    for group_id, matrix in state.hard.items():
        arr = np.asarray(matrix)
        hard[group_id] = arr[sample_index] if arr.ndim == 3 else arr
    return PermutationState(
        group_order=state.group_order,
        hard=hard,
        logits=None,
        metadata=dict(state.metadata),
    )


__all__ = ["SolverScheduler"]
