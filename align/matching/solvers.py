"""Group and component solvers for graph-native alignment."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..config.stages import SolverStep
from ..symmetry import groups_for_components
from .objectives import UnsupportedGroupLinearization
from .state import (
    TransformState,
    as_permutation_matrix,
    sinkhorn_operator,
    solve_lap_maximize,
)


def _scheduled_groups(problem, step: SolverStep) -> tuple[str, ...]:
    if step.components is not None:
        return groups_for_components(problem, step.components)
    groups = step.groups or problem.group_order
    unknown = sorted(set(groups) - set(problem.groups))
    if unknown:
        raise ValueError("Schedule references unknown group(s): " + ", ".join(unknown))
    return tuple(groups)


def solve_signed_lap_maximize(signed: np.ndarray, invariant: np.ndarray) -> np.ndarray:
    """Exact signed-permutation update from a split linear payoff.

    For any assignment the optimal sign of each matched pair is the sign of
    its (sign-covariant) cross term, so the assignment maximizes
    ``|signed| + invariant`` and the matched entries carry those signs.
    """

    perm = solve_lap_maximize(np.abs(signed) + invariant)
    return perm * np.where(np.asarray(signed) >= 0.0, 1.0, -1.0)


def solve_rotation_pairs_maximize(cross: np.ndarray) -> np.ndarray:
    """Exact per-pair rotation update maximizing ``<M, cross>``.

    Pairs follow the half-split convention ``(p, p + n/2)``; each pair's
    optimal angle is the closed-form 2-D Procrustes-to-rotation solution
    ``atan2(C[q,p] - C[p,q], C[p,p] + C[q,q])``.
    """

    cross = np.asarray(cross, dtype=np.float64)
    size = cross.shape[0]
    if size % 2:
        raise ValueError(f"rotation_pairs update needs an even size, got {size}.")
    half = size // 2
    idx = np.arange(half)
    trace_term = cross[idx, idx] + cross[idx + half, idx + half]
    skew_term = cross[idx + half, idx] - cross[idx, idx + half]
    angles = np.arctan2(skew_term, trace_term)
    cos, sin = np.cos(angles), np.sin(angles)
    matrix = np.zeros_like(cross)
    matrix[idx, idx] = cos
    matrix[idx + half, idx + half] = cos
    matrix[idx + half, idx] = sin
    matrix[idx, idx + half] = -sin
    return matrix


def solve_orthogonal_maximize(cross: np.ndarray) -> np.ndarray:
    """Exact orthogonal Procrustes update maximizing ``<Q, cross>``."""

    left, _, right_t = np.linalg.svd(np.asarray(cross, dtype=np.float64))
    return left @ right_t


def update_group_transform(
    problem, objective, ref_data, target_data, state, group_id: str
):
    """One exact coordinate update of a group within its declared class.

    All classes consume the same split linear payoff from the objective:
    permutations solve an LAP on ``signed + invariant``, signed permutations
    an LAP on ``|signed| + invariant`` with matched-entry signs, and
    rotation-pair groups the closed-form per-pair rotation projection of the
    sign-covariant cross term. Orthogonal-capable groups are updated at their
    signed-permutation subgroup here; the full orthogonal update is the
    ``procrustes`` schedule step.
    """

    group = problem.groups[group_id]
    signed, invariant = objective.linearize_group_split(
        problem, ref_data, target_data, state, group_id
    )
    signed = np.asarray(signed)
    invariant = np.asarray(invariant)
    if group.transform_family == "permutation":
        updated = solve_lap_maximize(signed + invariant)
    elif group.transform_family in ("signed_permutation", "orthogonal"):
        updated = solve_signed_lap_maximize(signed, invariant)
    elif group.transform_family == "rotation_pairs":
        updated = solve_rotation_pairs_maximize(signed)
    else:  # pragma: no cover - guarded by problem validation
        raise ValueError(f"Unknown group transform family {group.transform_family!r}.")
    previous = as_permutation_matrix(
        state.matrices[group_id], size=group.size, dtype=np.float64
    )
    delta = float(np.max(np.abs(updated - previous)))
    return state.with_matrices({group_id: updated}), delta


class LAPGroupSolver:
    """Exact transform-family-dispatched update for one group."""

    name = "lap"

    def __init__(self, objective) -> None:
        self.objective = objective

    def update(self, problem, ref_data, target_data, state, group_id: str):
        state, delta = update_group_transform(
            problem, self.objective, ref_data, target_data, state, group_id
        )
        return state, {"delta": delta}


class ProcrustesGroupSolver:
    """Full orthogonal Procrustes update for orthogonal-capable groups."""

    name = "procrustes"

    def __init__(self, objective) -> None:
        self.objective = objective

    def update(self, problem, ref_data, target_data, state, group_id: str):
        group = problem.groups[group_id]
        if group.transform_family != "orthogonal":
            raise ValueError(
                f"Group {group_id!r} declares transform family "
                f"{group.transform_family!r}; the procrustes solver only updates "
                "'orthogonal' groups."
            )
        signed, _ = self.objective.linearize_group_split(
            problem, ref_data, target_data, state, group_id
        )
        updated = solve_orthogonal_maximize(np.asarray(signed))
        previous = as_permutation_matrix(
            state.matrices[group_id], size=group.size, dtype=np.float64
        )
        delta = float(np.max(np.abs(updated - previous)))
        return state.with_matrices({group_id: updated}), {"delta": delta}


class SinkhornSolver:
    """Optimize soft permutation logits against a complete objective."""

    name = "sinkhorn"

    def __init__(self, objective, step: SolverStep) -> None:
        self.objective = objective
        self.step = step

    def _groups(self, problem) -> tuple[str, ...]:
        # Rotation-pair groups contain no non-trivial permutations, so the
        # doubly stochastic relaxation is not a valid symmetry relaxation for
        # them; their hard matrices stay fixed through Sinkhorn steps.
        return tuple(
            group_id
            for group_id in _scheduled_groups(problem, self.step)
            if problem.groups[group_id].transform_family != "rotation_pairs"
        )

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
                    state.matrices[group_id], size=group.size, dtype=np.float32
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
        optimizer = optax.adam(self.step.learning_rate)
        opt_state = optimizer.init(tuple(logits[gid] for gid in groups))

        def pack(logit_tuple):
            soft = {
                gid: sinkhorn_operator(
                    logit, tau=self.step.tau, n_iters=self.step.sinkhorn_iterations
                )
                for gid, logit in zip(groups, logit_tuple, strict=True)
            }
            matrices = dict(state.matrices)
            matrices.update(soft)
            return TransformState(group_order=problem.group_order, matrices=matrices)

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
                if delta <= self.step.tolerance:
                    break
            prev_losses = loss_values

        final_logits = {
            gid: logit for gid, logit in zip(groups, logit_tuple, strict=True)
        }
        projected_state = pack(logit_tuple)
        soft_state = projected_state.with_logits(final_logits)
        output_state = (
            projected_state.harden(groups=groups) if self.step.harden else soft_state
        )
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
    return ["lap", "procrustes", "sinkhorn"]


__all__ = [
    "SolverStep",
    "LAPGroupSolver",
    "ProcrustesGroupSolver",
    "SinkhornSolver",
    "UnsupportedGroupLinearization",
    "available_solvers",
    "solve_orthogonal_maximize",
    "solve_rotation_pairs_maximize",
    "solve_signed_lap_maximize",
    "update_group_transform",
]
