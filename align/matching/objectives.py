"""Objectives for graph-native alignment problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..symmetry import apply_matrix_to_axis, binding_axis_interval
from ..symmetry.tensor_ops import (
    binding_indexer,
    binding_selector,
    binding_sort_key,
    binding_transform_matrix,
)


class Objective(ABC):
    """Complete graph objective used by alignment schedulers."""

    name: str

    @abstractmethod
    def value(self, problem, ref_data, target_data, state):
        """Return scalar or batched objective value."""

    def linearize_group(self, problem, ref_data, target_data, state, group_id: str):
        """Return an LAP maximization cost matrix for one group."""

        signed, invariant = self.linearize_group_split(
            problem, ref_data, target_data, state, group_id
        )
        return signed + invariant

    def linearize_group_split(
        self, problem, ref_data, target_data, state, group_id: str
    ):
        """Return the group's linear payoff split as ``(signed, invariant)``.

        ``signed[i, j]`` is the payoff of assigning target channel ``j`` to
        reference slot ``i`` that flips sign under a channel sign flip (the
        cross terms of ``linear``-scope bindings); ``invariant[i, j]`` is the
        sign-invariant remainder (``permute_only`` binding cross terms and,
        for weighted objectives, the assignment-dependent self-energy). Class
        dispatch: permutation updates maximize ``signed + invariant``, signed
        updates ``|signed| + invariant``, and continuous (rotation/orthogonal)
        updates project ``signed`` alone — exact for objectives whose
        self-energy is transform-invariant (plain L2), a documented heuristic
        for weighted metrics.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support LAP linearization."
        )


OBJECTIVES: dict[str, type[Objective]] = {}


def register_objective(name: str):
    """Register an objective class."""

    def decorator(cls: type[Objective]) -> type[Objective]:
        OBJECTIVES[name.lower()] = cls
        cls.name = name.lower()
        return cls

    return decorator


def get_objective(name: str, **kwargs: Any) -> Objective:
    key = name.lower()
    if key not in OBJECTIVES:
        available = ", ".join(sorted(OBJECTIVES))
        raise ValueError(f"Unknown objective {name!r}. Available: {available}")
    return OBJECTIVES[key](**kwargs)


def available_objectives() -> list[str]:
    return sorted(OBJECTIVES)


def _is_batched_tensor(problem, tensor_id: str, tensor) -> bool:
    return jnp.asarray(tensor).ndim == len(problem.tensors[tensor_id].shape) + 1


def _apply_matrix_axis(tensor, matrix, *, axis: int, tensor_is_batched: bool):
    """Apply a hard or soft matrix along one axis, preserving batching if present."""

    tensor_arr = jnp.asarray(tensor)
    matrix_arr = jnp.asarray(matrix)
    if matrix_arr.ndim == 1:
        return jnp.take(tensor_arr, matrix_arr.astype(jnp.int32), axis=axis)
    matrix_arr = matrix_arr.astype(tensor_arr.dtype)
    if matrix_arr.ndim == 2:
        moved = jnp.moveaxis(tensor_arr, axis, 0)
        flat = moved.reshape((moved.shape[0], -1))
        out = matrix_arr @ flat
        out = out.reshape(moved.shape)
        return jnp.moveaxis(out, 0, axis)

    if matrix_arr.ndim != 3:
        raise ValueError(
            f"Permutation matrix must be 2D or 3D, got {matrix_arr.ndim}D."
        )
    if not tensor_is_batched:
        tensor_arr = jnp.broadcast_to(
            tensor_arr, (matrix_arr.shape[0], *tensor_arr.shape)
        )
        axis += 1
    if axis == 0:
        raise ValueError(
            "Batched permutation matrices require a tensor axis, not batch."
        )
    moved = jnp.moveaxis(tensor_arr, axis, 1)
    flat = moved.reshape((moved.shape[0], moved.shape[1], -1))
    out = jnp.einsum("bij,bjr->bir", matrix_arr, flat, optimize=True)
    out = out.reshape(moved.shape)
    return jnp.moveaxis(out, 1, axis)


def _sorted_bindings(problem, tensor_id: str):
    shape = problem.tensors[tensor_id].shape
    return sorted(
        problem.bindings_for_tensor(tensor_id),
        key=lambda binding: binding_sort_key(shape, binding),
    )


def _apply_state_to_tensor(problem, tensor_id: str, tensor, state):
    updated = jnp.asarray(tensor)
    shape = problem.tensors[tensor_id].shape
    for binding in _sorted_bindings(problem, tensor_id):
        tensor_is_batched = _is_batched_tensor(problem, tensor_id, updated)
        matrix = binding_transform_matrix(
            state.hard[binding.group],
            transform_family=problem.groups[binding.group].transform_family,
            scope=binding.transform_scope,
        )
        if matrix is None:
            continue
        matrix_arr = jnp.asarray(matrix)
        if matrix_arr.ndim == 3 and not tensor_is_batched:
            updated = jnp.broadcast_to(updated, (matrix_arr.shape[0], *updated.shape))
            tensor_is_batched = True
        spec_axis, start, stop = binding_axis_interval(shape, binding)
        selector = binding_selector(shape, binding)
        offset = 1 if tensor_is_batched else 0
        axis = spec_axis + offset
        axis_size = int(updated.shape[axis])
        if not selector and start == 0 and stop == axis_size:
            updated = _apply_matrix_axis(
                updated,
                matrix,
                axis=axis,
                tensor_is_batched=tensor_is_batched,
            )
            continue
        indexer = binding_indexer(
            updated.ndim, spec_axis, start, stop, selector, offset=offset
        )
        segment = updated[indexer]
        transformed = _apply_matrix_axis(
            segment,
            matrix,
            axis=axis,
            tensor_is_batched=tensor_is_batched,
        )
        updated = updated.at[indexer].set(transformed)
    return updated


def _apply_other_groups_hard(
    problem, tensor_id: str, tensor, state, skip_groups: frozenset[str] | set[str]
):
    """Apply all hard group actions except ``skip_groups`` to one tensor."""

    updated = np.asarray(tensor)
    shape = problem.tensors[tensor_id].shape
    for binding in _sorted_bindings(problem, tensor_id):
        if binding.group in skip_groups:
            continue
        matrix = binding_transform_matrix(
            np.asarray(state.hard[binding.group]),
            transform_family=problem.groups[binding.group].transform_family,
            scope=binding.transform_scope,
        )
        if matrix is None:
            continue
        axis, start, stop = binding_axis_interval(shape, binding)
        selector = binding_selector(shape, binding)
        if not selector and start == 0 and stop == int(updated.shape[axis]):
            updated = apply_matrix_to_axis(updated, matrix, axis=axis)
            continue
        indexer = binding_indexer(updated.ndim, axis, start, stop, selector)
        segment = updated[indexer]
        transformed = apply_matrix_to_axis(segment, matrix, axis=axis)
        updated = np.array(updated, copy=True)
        updated[indexer] = transformed
    return updated


def _require_lap_linearizable(problem, group_id: str) -> None:
    """Reject groups whose exact hard update is not a linear assignment."""

    for constraint in problem.constraints:
        if (
            constraint.kind == "attention_block"
            and constraint.metadata.get("head_group") == group_id
        ):
            raise UnsupportedGroupLinearization(
                f"Group {group_id!r} is an attention head group; its generic "
                "LAP linearization is inexact once intra-head permutations "
                "are non-identity. Use the structured attention update "
                "(lap schedules do this automatically) or a Sinkhorn "
                "schedule."
            )
        if (
            constraint.kind == "gqa_attention_block"
            and constraint.metadata.get("kv_group") == group_id
        ):
            raise UnsupportedGroupLinearization(
                f"Group {group_id!r} is a GQA kv-group head group; its "
                "generic LAP linearization is inexact once per-slot "
                "query-head or vo permutations are non-identity. Use the "
                "structured GQA update (lap schedules do this automatically) "
                "or a Sinkhorn schedule."
            )

    repeated = problem.repeated_group_terms().get(group_id, ())
    if repeated:
        joined = ", ".join(repeated)
        raise UnsupportedGroupLinearization(
            f"Group {group_id!r} appears multiple times in tensor term(s) "
            f"{joined}. The exact hard update is a quadratic assignment "
            "problem, not an LAP. Use a Sinkhorn schedule for this problem."
        )


@register_objective("l2_weight")
class EuclideanObjective(Objective):
    """Sum of squared tensor differences after applying the alignment state."""

    name = "l2_weight"

    def __init__(self, repeated_group_policy: str = "reject") -> None:
        if repeated_group_policy != "reject":
            raise ValueError(
                "Only repeated_group_policy='reject' is implemented for exact LAP "
                "linearization. Use a Sinkhorn schedule for repeated-group tensor terms."
            )
        self.repeated_group_policy = repeated_group_policy

    def value(self, problem, ref_data, target_data, state):
        loss = None
        for tensor_id in problem.tensors:
            ref = jnp.asarray(ref_data[tensor_id])
            target = jnp.asarray(target_data[tensor_id])
            aligned = _apply_state_to_tensor(problem, tensor_id, target, state)
            if aligned.ndim == ref.ndim + 1:
                diff = ref[None, ...] - aligned
                axes = tuple(range(1, diff.ndim))
            else:
                diff = ref - aligned
                axes = None
            term = jnp.sum(jnp.square(diff), axis=axes)
            loss = term if loss is None else loss + term
        if loss is None:
            return jnp.asarray(0.0)
        return loss

    def linearize_group_split(
        self, problem, ref_data, target_data, state, group_id: str
    ):
        _require_lap_linearizable(problem, group_id)

        group = problem.groups[group_id]
        signed = np.zeros((group.size, group.size), dtype=np.float64)
        invariant = np.zeros((group.size, group.size), dtype=np.float64)
        for tensor_id in problem.tensors:
            bindings = [
                binding
                for binding in problem.bindings_for_tensor(tensor_id)
                if binding.group == group_id
            ]
            if not bindings:
                continue
            # Multiple bindings on one axis (e.g. a conv->flatten boundary
            # binding each spatial position's channel slice) contribute a sum
            # of independent linear terms; multi-axis repeats are rejected by
            # the linearizability check above.
            shape = problem.tensors[tensor_id].shape
            ref = np.asarray(ref_data[tensor_id])
            target = _apply_other_groups_hard(
                problem, tensor_id, target_data[tensor_id], state, {group_id}
            )
            for binding in bindings:
                axis, start, stop = binding_axis_interval(shape, binding)
                selector = binding_selector(shape, binding)
                indexer = binding_indexer(ref.ndim, axis, start, stop, selector)
                ref_mat = np.moveaxis(ref[indexer], axis, 0).reshape(group.size, -1)
                target_mat = np.moveaxis(target[indexer], axis, 0).reshape(
                    group.size, -1
                )
                bucket = (
                    invariant if binding.transform_scope == "permute_only" else signed
                )
                bucket += ref_mat @ target_mat.T
        return signed, invariant


@register_objective("fisher_l2")
class DiagonalFisherObjective(Objective):
    """Diagonal Fisher-metric weight distance.

    Scores ``sum_t sum_i M_t[i] * (ref_t[i] - aligned_t[i])^2`` with fixed
    positive per-tensor weights ``M_t`` in *reference* coordinates -- the
    squared Riemannian length of the residual under a diagonal metric
    evaluated at the reference sample. With ``M_t`` the diagonal Gauss-Newton
    Fisher (see :mod:`align.matching.fisher`) this pulls function-space L2
    geometry back into weight space: matching prioritizes coordinates the
    function is sensitive to and discounts flat directions.

    Unlike plain L2, the target's self-energy ``sum M * aligned^2`` is not
    permutation-invariant, so the exact LAP cost carries an extra self term.
    """

    name = "fisher_l2"

    def __init__(self, tensor_weights=None, weights_path=None) -> None:
        if (tensor_weights is None) == (weights_path is None):
            raise ValueError(
                "fisher_l2 requires exactly one of 'tensor_weights' (mapping "
                "of tensor id to weight array) or 'weights_path' (npz file)."
            )
        if weights_path is not None:
            from .fisher import load_tensor_weights_npz

            tensor_weights = load_tensor_weights_npz(weights_path)
        self.tensor_weights = {
            str(tensor_id): np.asarray(value, dtype=np.float64)
            for tensor_id, value in dict(tensor_weights).items()
        }
        for tensor_id, weight in self.tensor_weights.items():
            if not np.all(weight >= 0.0):
                raise ValueError(
                    f"fisher_l2 weights for tensor {tensor_id!r} must be non-negative."
                )

    def _weight(self, problem, tensor_id: str) -> np.ndarray:
        weight = self.tensor_weights.get(tensor_id)
        if weight is None:
            raise ValueError(
                f"fisher_l2 has no weights for tensor {tensor_id!r}; weights "
                "must cover every problem tensor."
            )
        expected = tuple(problem.tensors[tensor_id].shape)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"fisher_l2 weights for tensor {tensor_id!r} have shape "
                f"{weight.shape}, expected {expected}."
            )
        return weight

    def value(self, problem, ref_data, target_data, state):
        loss = None
        for tensor_id in problem.tensors:
            ref = jnp.asarray(ref_data[tensor_id])
            target = jnp.asarray(target_data[tensor_id])
            weight = jnp.asarray(self._weight(problem, tensor_id), dtype=ref.dtype)
            aligned = _apply_state_to_tensor(problem, tensor_id, target, state)
            if aligned.ndim == ref.ndim + 1:
                diff = ref[None, ...] - aligned
                axes = tuple(range(1, diff.ndim))
            else:
                diff = ref - aligned
                axes = None
            term = jnp.sum(weight * jnp.square(diff), axis=axes)
            loss = term if loss is None else loss + term
        if loss is None:
            return jnp.asarray(0.0)
        return loss

    def linearize_group_split(
        self, problem, ref_data, target_data, state, group_id: str
    ):
        _require_lap_linearizable(problem, group_id)

        group = problem.groups[group_id]
        signed = np.zeros((group.size, group.size), dtype=np.float64)
        invariant = np.zeros((group.size, group.size), dtype=np.float64)
        for tensor_id in problem.tensors:
            bindings = [
                binding
                for binding in problem.bindings_for_tensor(tensor_id)
                if binding.group == group_id
            ]
            if not bindings:
                continue
            # Same-axis multi-bindings sum independent linear terms (see the
            # L2 linearization); multi-axis repeats are rejected above.
            shape = problem.tensors[tensor_id].shape
            ref = np.asarray(ref_data[tensor_id])
            weight = self._weight(problem, tensor_id)
            target = _apply_other_groups_hard(
                problem, tensor_id, target_data[tensor_id], state, {group_id}
            )
            for binding in bindings:
                axis, start, stop = binding_axis_interval(shape, binding)
                selector = binding_selector(shape, binding)
                indexer = binding_indexer(ref.ndim, axis, start, stop, selector)
                ref_mat = np.moveaxis(ref[indexer], axis, 0).reshape(group.size, -1)
                weight_mat = np.moveaxis(weight[indexer], axis, 0).reshape(
                    group.size, -1
                )
                target_mat = np.moveaxis(target[indexer], axis, 0).reshape(
                    group.size, -1
                )
                cross = (weight_mat * ref_mat) @ target_mat.T
                bucket = (
                    invariant if binding.transform_scope == "permute_only" else signed
                )
                bucket += cross
                # The weighted target self-energy depends on the assignment
                # but not on channel signs, so it is sign-invariant.
                invariant -= 0.5 * weight_mat @ np.square(target_mat).T
        return signed, invariant


class UnsupportedGroupLinearization(ValueError):
    """Raised when an exact LAP update is mathematically invalid."""


__all__ = [
    "Objective",
    "EuclideanObjective",
    "DiagonalFisherObjective",
    "UnsupportedGroupLinearization",
    "register_objective",
    "get_objective",
    "available_objectives",
]
