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
    MatrixTransform,
    PermutationTransform,
    SignedPermutationTransform,
    TransformState,
    sinkhorn_operator,
    solve_lap_maximize,
    transform_matrix,
)


def _scheduled_groups(graph, step: SolverStep) -> tuple[str, ...]:
    if step.components is not None:
        return groups_for_components(graph, step.components)
    groups = step.groups or graph.group_order
    unknown = sorted(set(groups) - set(graph.groups))
    if unknown:
        raise ValueError("Schedule references unknown group(s): " + ", ".join(unknown))
    return tuple(groups)


def solve_signed_lap_maximize(
    signed: np.ndarray, invariant: np.ndarray
) -> SignedPermutationTransform:
    """Exact signed-permutation update from a split linear payoff.

    For any assignment the optimal sign of each matched pair is the sign of
    its (sign-covariant) cross term, so the assignment maximizes
    ``|signed| + invariant`` and the matched entries carry those signs.
    """

    indices = solve_lap_maximize(np.abs(signed) + invariant)
    signs = np.where(np.asarray(signed)[np.arange(indices.size), indices] >= 0.0, 1, -1)
    return SignedPermutationTransform(indices=indices, signs=signs.astype(np.int8))


def _transform_delta(previous, updated) -> float:
    if isinstance(previous, PermutationTransform) and isinstance(
        updated, PermutationTransform
    ):
        return 0.0 if np.array_equal(previous.indices, updated.indices) else 1.0
    if isinstance(previous, SignedPermutationTransform) and isinstance(
        updated, SignedPermutationTransform
    ):
        unchanged = np.array_equal(
            previous.indices, updated.indices
        ) and np.array_equal(previous.signs, updated.signs)
        return 0.0 if unchanged else 1.0
    previous_matrix = np.asarray(transform_matrix(previous), dtype=np.float64)
    updated_matrix = np.asarray(transform_matrix(updated), dtype=np.float64)
    return float(np.max(np.abs(updated_matrix - previous_matrix)))


def assignment_objective_margin(score: np.ndarray) -> float | None:
    """Exact gap between the best and second-best assignment objectives."""

    score = np.asarray(score, dtype=np.float64)
    if score.shape[0] <= 1:
        return None
    best_indices = solve_lap_maximize(score)
    rows = np.arange(score.shape[0])
    best = float(np.sum(score[rows, best_indices]))
    span = float(np.ptp(score))
    forbidden = float(np.min(score) - (span + 1.0) * (score.shape[0] + 1))
    second = -np.inf
    for row, column in enumerate(best_indices):
        candidate_score = np.array(score, copy=True)
        candidate_score[row, column] = forbidden
        candidate = solve_lap_maximize(candidate_score)
        if candidate[row] == column:
            continue
        second = max(second, float(np.sum(score[rows, candidate])))
    return None if not np.isfinite(second) else max(best - second, 0.0)


def group_assignment_ambiguity(
    graph, objective, reference_data, target_data, state, group_id: str
) -> dict[str, float] | None:
    """Return an opt-in assignment margin for a discrete group update."""

    family = graph.groups[group_id].transform_family
    if family == "rotation_pairs":
        return None
    signed, invariant = objective.linearize_group_split(
        graph, reference_data, target_data, state, group_id
    )
    score = (
        np.abs(np.asarray(signed)) + np.asarray(invariant)
        if family in ("signed_permutation", "orthogonal")
        else np.asarray(signed) + np.asarray(invariant)
    )
    margin = assignment_objective_margin(score)
    return None if margin is None else {"assignment_objective_margin": margin}


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
    graph, objective, reference_data, target_data, state, group_id: str
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

    group = graph.groups[group_id]
    signed, invariant = objective.linearize_group_split(
        graph, reference_data, target_data, state, group_id
    )
    signed = np.asarray(signed)
    invariant = np.asarray(invariant)
    if group.transform_family == "permutation":
        updated = PermutationTransform(solve_lap_maximize(signed + invariant))
    elif group.transform_family in ("signed_permutation", "orthogonal"):
        updated = solve_signed_lap_maximize(signed, invariant)
    elif group.transform_family == "rotation_pairs":
        updated = MatrixTransform(
            solve_rotation_pairs_maximize(signed), family="rotation_pairs"
        )
    else:  # pragma: no cover - guarded by graph validation
        raise ValueError(f"Unknown group transform family {group.transform_family!r}.")
    delta = _transform_delta(state.transforms[group_id], updated)
    state = state.with_transforms({group_id: updated})
    objective.notify_group_updated(graph, group_id)
    return state, delta


class LAPGroupSolver:
    """Exact transform-family-dispatched update for one group."""

    name = "lap"

    def __init__(self, objective, *, record_ambiguity: bool = False) -> None:
        self.objective = objective
        self.record_ambiguity = bool(record_ambiguity)

    def update(self, graph, reference_data, target_data, state, group_id: str):
        ambiguity = (
            group_assignment_ambiguity(
                graph,
                self.objective,
                reference_data,
                target_data,
                state,
                group_id,
            )
            if self.record_ambiguity
            else None
        )
        state, delta = update_group_transform(
            graph, self.objective, reference_data, target_data, state, group_id
        )
        aux = {"delta": delta}
        if ambiguity is not None:
            aux["ambiguity"] = {group_id: ambiguity}
        return state, aux


class ProcrustesGroupSolver:
    """Full orthogonal Procrustes update for orthogonal-capable groups."""

    name = "procrustes"

    def __init__(self, objective) -> None:
        self.objective = objective

    def update(self, graph, reference_data, target_data, state, group_id: str):
        group = graph.groups[group_id]
        if group.transform_family != "orthogonal":
            raise ValueError(
                f"Group {group_id!r} declares transform family "
                f"{group.transform_family!r}; the procrustes solver only updates "
                "'orthogonal' groups."
            )
        signed, _ = self.objective.linearize_group_split(
            graph, reference_data, target_data, state, group_id
        )
        updated = MatrixTransform(
            solve_orthogonal_maximize(np.asarray(signed)), family="orthogonal"
        )
        delta = _transform_delta(state.transforms[group_id], updated)
        state = state.with_transforms({group_id: updated})
        self.objective.notify_group_updated(graph, group_id)
        return state, {"delta": delta}


class SinkhornSolver:
    """Optimize soft permutation logits against a complete objective."""

    name = "sinkhorn"

    def __init__(self, objective, step: SolverStep) -> None:
        self.objective = objective
        self.step = step
        self._compiled_loops: dict[tuple[Any, ...], Any] = {}

    def _groups(self, graph) -> tuple[str, ...]:
        groups = _scheduled_groups(graph, self.step)
        unsupported = [
            group_id
            for group_id in groups
            if graph.groups[group_id].transform_family != "permutation"
        ]
        if unsupported:
            families = ", ".join(
                f"{group_id}={graph.groups[group_id].transform_family}"
                for group_id in unsupported
            )
            raise ValueError(
                "sinkhorn currently supports only plain permutation groups; "
                f"scheduled unsupported groups: {families}. Use lap/procrustes "
                "for signed, rotation-pair, or orthogonal transforms."
            )
        return groups

    def _init_logits(
        self, graph, state, *, batch_size: int, rng_keys
    ) -> dict[str, jnp.ndarray]:
        groups = self._groups(graph)
        logits: dict[str, jnp.ndarray] = {}
        sample_keys = jnp.asarray(rng_keys)
        for group_index, group_id in enumerate(groups):
            group = graph.groups[group_id]
            base = jnp.asarray(state.matrix(group_id, dtype=jnp.float32))
            if base.ndim == 2:
                base = jnp.broadcast_to(base, (batch_size, group.size, group.size))
            group_keys = jax.vmap(
                lambda key, index=group_index: jax.random.fold_in(key, index)
            )(sample_keys)
            size = int(group.size)
            dtype = base.dtype
            noise = self.step.init_scale * jax.vmap(
                lambda key, size=size, dtype=dtype: jax.random.normal(
                    key, (size, size), dtype=dtype
                )
            )(group_keys)
            logits[group_id] = base + noise
        return logits

    def _compiled_loop(self, graph, groups):
        signature = (
            id(graph),
            groups,
            tuple(graph.tensors),
            self.step.max_steps,
            self.step.tau,
            self.step.learning_rate,
            self.step.sinkhorn_iterations,
            self.step.tolerance,
            self.step.record_loss_history,
        )
        cached = self._compiled_loops.get(signature)
        if cached is not None:
            return cached

        tensor_ids = tuple(graph.tensors)
        all_group_ids = graph.group_order
        template = TransformState.identity(graph, backend="jax")
        optimizer = optax.adam(self.step.learning_rate)

        def pack(logit_tuple, base_matrices):
            matrices = {
                group_id: matrix
                for group_id, matrix in zip(all_group_ids, base_matrices, strict=True)
            }
            projected = tuple(
                sinkhorn_operator(
                    logit,
                    tau=self.step.tau,
                    n_iters=self.step.sinkhorn_iterations,
                )
                for logit in logit_tuple
            )
            matrices.update(
                {gid: matrix for gid, matrix in zip(groups, projected, strict=True)}
            )
            return template.with_relaxation({}, matrices), projected

        def losses_for(logit_tuple, base_matrices, reference_values, target_values):
            soft_state, projected = pack(logit_tuple, base_matrices)
            reference = dict(zip(tensor_ids, reference_values, strict=True))
            target = dict(zip(tensor_ids, target_values, strict=True))
            losses = jnp.ravel(
                jnp.asarray(self.objective.value(graph, reference, target, soft_state))
            )
            return losses, projected

        def summed_loss(logit_tuple, base_matrices, reference_values, target_values):
            losses, _ = losses_for(
                logit_tuple, base_matrices, reference_values, target_values
            )
            return jnp.sum(losses), losses

        value_and_grad = jax.value_and_grad(summed_loss, has_aux=True)

        def run(logit_tuple, base_matrices, reference_values, target_values):
            opt_state = optimizer.init(logit_tuple)
            initial_losses, _ = losses_for(
                logit_tuple, base_matrices, reference_values, target_values
            )
            history = jnp.full(
                (self.step.max_steps, initial_losses.shape[0]),
                jnp.nan,
                dtype=initial_losses.dtype,
            )

            def cond(loop_state):
                step_index, _, _, _, delta, _ = loop_state
                return (step_index < self.step.max_steps) & (
                    (step_index < 2) | (delta > self.step.tolerance)
                )

            def body(loop_state):
                step_index, current, current_opt, previous_losses, _, loss_history = (
                    loop_state
                )
                (_, losses), grads = value_and_grad(
                    current, base_matrices, reference_values, target_values
                )
                updates, next_opt = optimizer.update(grads, current_opt, current)
                next_logits = optax.apply_updates(current, updates)
                delta = jnp.where(
                    step_index == 0,
                    jnp.asarray(jnp.inf, dtype=losses.dtype),
                    jnp.max(jnp.abs(losses - previous_losses)),
                )
                if self.step.record_loss_history:
                    loss_history = loss_history.at[step_index].set(losses)
                return (
                    step_index + 1,
                    next_logits,
                    next_opt,
                    losses,
                    delta,
                    loss_history,
                )

            loop_state = jax.lax.while_loop(
                cond,
                body,
                (
                    jnp.asarray(0, dtype=jnp.int32),
                    logit_tuple,
                    opt_state,
                    initial_losses,
                    jnp.asarray(jnp.inf, dtype=initial_losses.dtype),
                    history,
                ),
            )
            steps, final_logits, _, _, _, history = loop_state
            final_losses, projected = losses_for(
                final_logits, base_matrices, reference_values, target_values
            )
            return final_logits, projected, steps, initial_losses, final_losses, history

        compiled = jax.jit(run)
        self._compiled_loops[signature] = compiled
        return compiled

    def solve(
        self,
        graph,
        reference_data,
        target_data,
        state,
        *,
        rng_key=None,
        rng_keys=None,
    ):
        batch_size = _batch_size(graph, target_data)
        if rng_keys is None:
            base_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
            rng_keys = (
                base_key[None, :]
                if batch_size == 1
                else jax.vmap(lambda index: jax.random.fold_in(base_key, index))(
                    jnp.arange(batch_size, dtype=jnp.int32)
                )
            )
        if len(rng_keys) != batch_size:
            raise ValueError(
                f"Sinkhorn received {len(rng_keys)} RNG keys for batch size {batch_size}."
            )
        logits = self._init_logits(
            graph, state, batch_size=batch_size, rng_keys=rng_keys
        )
        groups = self._groups(graph)
        logit_tuple = tuple(logits[gid] for gid in groups)
        base_matrices = tuple(
            jnp.asarray(state.matrix(group_id, dtype=jnp.float32))
            for group_id in graph.group_order
        )
        reference_values = tuple(
            jnp.asarray(reference_data[key]) for key in graph.tensors
        )
        target_values = tuple(jnp.asarray(target_data[key]) for key in graph.tensors)
        loop = self._compiled_loop(graph, groups)
        final_tuple, projected_tuple, steps, initial, final, history = loop(
            logit_tuple, base_matrices, reference_values, target_values
        )
        projected = {
            gid: matrix for gid, matrix in zip(groups, projected_tuple, strict=True)
        }
        final_logits = {
            gid: value for gid, value in zip(groups, final_tuple, strict=True)
        }
        relaxed = state.with_relaxation(final_logits, projected)
        output_state = relaxed.harden(groups=groups)
        steps_taken = int(np.asarray(steps))
        initial_losses = np.asarray(initial).tolist()
        final_losses = np.asarray(final).tolist()
        aux: dict[str, Any] = {
            "solver": "sinkhorn",
            "steps": steps_taken,
            "converged": steps_taken < self.step.max_steps,
            "loss_initial": [float(value) for value in initial_losses],
            "loss_final": [float(value) for value in final_losses],
            "hardening_method": "hungarian",
        }
        if self.step.record_loss_history:
            history_values = np.asarray(history)[:steps_taken]
            aux["loss_history"] = history_values.T.tolist()
        if self.step.record_ambiguity:
            aux["ambiguity"] = _transport_ambiguity(projected, batch_size)
        return output_state, aux


def _transport_ambiguity(
    projected, batch_size: int
) -> dict[str, list[dict[str, float]]]:
    """Compact row-entropy and assignment-margin diagnostics for soft plans."""

    payload: dict[str, list[dict[str, float]]] = {}
    for group_id, value in projected.items():
        matrices = np.asarray(value)
        if matrices.ndim == 2:
            matrices = matrices[None, ...]
        group_values: list[dict[str, float]] = []
        for matrix in matrices:
            clipped = np.clip(matrix, 1e-12, 1.0)
            entropy = -np.sum(clipped * np.log(clipped), axis=-1)
            ordered = np.sort(matrix, axis=-1)
            margin = (
                ordered[:, -1] - ordered[:, -2] if matrix.shape[-1] > 1 else np.ones(1)
            )
            group_values.append(
                {
                    "row_entropy_mean": float(np.mean(entropy)),
                    "row_entropy_max": float(np.max(entropy)),
                    "assignment_margin_mean": float(np.mean(margin)),
                    "assignment_margin_min": float(np.min(margin)),
                }
            )
        if len(group_values) != batch_size:
            raise ValueError("Sinkhorn ambiguity batch shape mismatch.")
        payload[group_id] = group_values
    return payload


def _batch_size(graph, target_data) -> int:
    if not graph.tensors:
        return 1
    first_id = next(iter(graph.tensors))
    target = jnp.asarray(target_data[first_id])
    expected = len(graph.tensors[first_id].shape)
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
    "assignment_objective_margin",
    "group_assignment_ambiguity",
    "solve_orthogonal_maximize",
    "solve_rotation_pairs_maximize",
    "solve_signed_lap_maximize",
    "update_group_transform",
]
