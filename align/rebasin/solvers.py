"""Group and block solvers for graph-native alignment."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..alignment import groups_for_blocks
from ..options import RebasinScheduleStep as SolverScheduleStep
from .objectives import UnsupportedGroupLinearization
from .permutation_state import (
    PermutationState,
    as_permutation_matrix,
    sinkhorn_operator,
    solve_lap_maximize,
)


def _scheduled_groups(problem, step: SolverScheduleStep) -> tuple[str, ...]:
    if step.blocks is not None:
        return groups_for_blocks(problem, step.blocks)
    groups = step.groups or problem.group_order
    unknown = sorted(set(groups) - set(problem.groups))
    if unknown:
        raise ValueError("Schedule references unknown group(s): " + ", ".join(unknown))
    return tuple(groups)


class LAPGroupSolver:
    """Exact LAP update for one group when the objective is linear in it."""

    name = "lap"

    def __init__(self, objective) -> None:
        self.objective = objective

    def update(self, problem, ref_data, target_data, state, group_id: str):
        cost = self.objective.linearize_group(
            problem, ref_data, target_data, state, group_id
        )
        updated = solve_lap_maximize(np.asarray(cost))
        previous = as_permutation_matrix(
            state.hard[group_id],
            size=problem.groups[group_id].size,
            dtype=updated.dtype,
        )
        delta = float(np.max(np.abs(updated - previous)))
        return state.with_hard({group_id: updated}), {"delta": delta}


class SinkhornBlockSolver:
    """Optimize soft permutation logits against a complete objective."""

    name = "sinkhorn"

    def __init__(self, objective, step: SolverScheduleStep) -> None:
        self.objective = objective
        self.step = step

    def _groups(self, problem) -> tuple[str, ...]:
        return _scheduled_groups(problem, self.step)

    def _init_logits(
        self, problem, state, *, batch_size: int, rng_key
    ) -> dict[str, jnp.ndarray]:
        groups = self._groups(problem)
        keys = jax.random.split(rng_key, len(groups))
        logits: dict[str, jnp.ndarray] = {}
        for key, group_id in zip(keys, groups, strict=True):
            group = problem.groups[group_id]
            base = jnp.asarray(
                as_permutation_matrix(
                    state.hard[group_id], size=group.size, dtype=np.float32
                )
            )
            if base.ndim == 2:
                base = jnp.broadcast_to(base, (batch_size, group.size, group.size))
            noise = self.step.init_scale * jax.random.normal(
                key, (batch_size, group.size, group.size), dtype=base.dtype
            )
            logits[group_id] = base + noise
        return logits

    def solve(self, problem, ref_data, target_data, state, *, rng_key=None):
        batch_size = _batch_size(problem, target_data)
        rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        logits = self._init_logits(
            problem, state, batch_size=batch_size, rng_key=rng_key
        )
        groups = self._groups(problem)
        optimizer = optax.adam(self.step.lr)
        opt_state = optimizer.init(tuple(logits[gid] for gid in groups))

        def pack(logit_tuple):
            soft = {
                gid: sinkhorn_operator(
                    logit, tau=self.step.tau, n_iters=self.step.n_sinkhorn_iters
                )
                for gid, logit in zip(groups, logit_tuple, strict=True)
            }
            hard = dict(state.hard)
            hard.update(soft)
            return PermutationState(group_order=problem.group_order, hard=hard)

        def loss_fn(logit_tuple):
            soft_state = pack(logit_tuple)
            values = self.objective.value(problem, ref_data, target_data, soft_state)
            return jnp.sum(values), values

        value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
        logit_tuple = tuple(logits[gid] for gid in groups)
        history = (
            [[] for _ in range(batch_size)] if self.step.record_loss_history else None
        )
        initial_losses: list[float | None] = [None] * batch_size
        final_losses: list[float | None] = [None] * batch_size
        prev_losses = None
        steps_taken = 0

        for _ in range(self.step.max_steps):
            (loss_sum, losses), grads = value_and_grad(logit_tuple)
            del loss_sum
            loss_values = jnp.ravel(jnp.asarray(losses))
            for idx in range(batch_size):
                value = float(loss_values[idx])
                if history is not None:
                    history[idx].append(value)
                if initial_losses[idx] is None:
                    initial_losses[idx] = value
                final_losses[idx] = value
            updates, opt_state = optimizer.update(grads, opt_state, logit_tuple)
            logit_tuple = optax.apply_updates(logit_tuple, updates)
            steps_taken += 1
            if prev_losses is not None:
                delta = float(jnp.max(jnp.abs(loss_values - prev_losses)))
                if delta <= self.step.tol:
                    break
            prev_losses = loss_values

        final_logits = {
            gid: logit for gid, logit in zip(groups, logit_tuple, strict=True)
        }
        projected_state = pack(logit_tuple)
        soft_state = projected_state.with_logits(final_logits)
        output_state = projected_state.harden() if self.step.harden else soft_state
        aux: dict[str, Any] = {
            "solver": "sinkhorn",
            "steps": int(steps_taken),
            "loss_initial": [
                float(value) if value is not None else 0.0 for value in initial_losses
            ],
            "loss_final": [
                float(value) if value is not None else 0.0 for value in final_losses
            ],
            "hardening_method": "hungarian" if self.step.harden else None,
        }
        if history is not None:
            aux["loss_history"] = history
        return output_state, aux


def _batch_size(problem, target_data) -> int:
    if not problem.tensors:
        return 1
    first_id = next(iter(problem.tensors))
    target = jnp.asarray(target_data[first_id])
    expected = len(problem.tensors[first_id].shape)
    return int(target.shape[0]) if target.ndim == expected + 1 else 1


def available_solvers() -> list[str]:
    return ["lap", "sinkhorn"]


__all__ = [
    "SolverScheduleStep",
    "LAPGroupSolver",
    "SinkhornBlockSolver",
    "UnsupportedGroupLinearization",
    "available_solvers",
]
