"""Objectives for graph-native alignment problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..alignment import apply_perm_to_axis, axis_slice, binding_axis_interval


class Objective(ABC):
    """Complete graph objective used by alignment schedulers."""

    name: str

    @abstractmethod
    def value(self, problem, ref_data, target_data, state):
        """Return scalar or batched objective value."""

    def linearize_group(self, problem, ref_data, target_data, state, group_id: str):
        """Return an LAP maximization cost matrix for one group."""

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


def _apply_state_to_tensor(problem, tensor_id: str, tensor, state):
    updated = jnp.asarray(tensor)
    for binding in problem.bindings_for_tensor(tensor_id):
        tensor_is_batched = _is_batched_tensor(problem, tensor_id, updated)
        matrix = state.hard[binding.group]
        matrix_arr = jnp.asarray(matrix)
        if matrix_arr.ndim == 3 and not tensor_is_batched:
            updated = jnp.broadcast_to(updated, (matrix_arr.shape[0], *updated.shape))
            tensor_is_batched = True
        spec_axis, start, stop = binding_axis_interval(
            problem.tensors[tensor_id].shape, binding
        )
        axis = spec_axis + 1 if tensor_is_batched else spec_axis
        axis_size = int(updated.shape[axis])
        if start == 0 and stop == axis_size:
            updated = _apply_matrix_axis(
                updated,
                matrix,
                axis=axis,
                tensor_is_batched=tensor_is_batched,
            )
            continue
        indexer = axis_slice(updated.ndim, axis, start, stop)
        segment = updated[indexer]
        transformed = _apply_matrix_axis(
            segment,
            matrix,
            axis=axis,
            tensor_is_batched=tensor_is_batched,
        )
        updated = updated.at[indexer].set(transformed)
    return updated


def _apply_other_groups_hard(problem, tensor_id: str, tensor, state, group_id: str):
    updated = np.asarray(tensor)
    for binding in problem.bindings_for_tensor(tensor_id):
        if binding.group == group_id:
            continue
        axis, start, stop = binding_axis_interval(
            problem.tensors[tensor_id].shape, binding
        )
        if start == 0 and stop == int(updated.shape[axis]):
            updated = apply_perm_to_axis(
                updated,
                np.asarray(state.hard[binding.group]),
                axis=axis,
            )
            continue
        indexer = axis_slice(updated.ndim, axis, start, stop)
        segment = updated[indexer]
        transformed = apply_perm_to_axis(
            segment,
            np.asarray(state.hard[binding.group]),
            axis=axis,
        )
        updated = np.array(updated, copy=True)
        updated[indexer] = transformed
    return updated


@register_objective("l2_weight")
class L2WeightObjective(Objective):
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

    def linearize_group(self, problem, ref_data, target_data, state, group_id: str):
        repeated = problem.repeated_group_terms().get(group_id, ())
        if repeated:
            joined = ", ".join(repeated)
            raise UnsupportedGroupLinearization(
                f"Group {group_id!r} appears multiple times in tensor term(s) "
                f"{joined}. The exact hard update is a quadratic assignment "
                "problem, not an LAP. Use a Sinkhorn schedule for this problem."
            )

        group = problem.groups[group_id]
        cost = np.zeros((group.size, group.size), dtype=np.float64)
        for tensor_id in problem.tensors:
            bindings = [
                binding
                for binding in problem.bindings_for_tensor(tensor_id)
                if binding.group == group_id
            ]
            if not bindings:
                continue
            if len(bindings) > 1:  # Defensive; repeated check above should catch this.
                raise UnsupportedGroupLinearization(group_id)
            binding = bindings[0]
            ref = np.asarray(ref_data[tensor_id])
            target = _apply_other_groups_hard(
                problem, tensor_id, target_data[tensor_id], state, group_id
            )
            axis, start, stop = binding_axis_interval(
                problem.tensors[tensor_id].shape, binding
            )
            indexer = axis_slice(ref.ndim, axis, start, stop)
            ref_mat = np.moveaxis(ref[indexer], axis, 0).reshape(group.size, -1)
            target_mat = np.moveaxis(target[indexer], axis, 0).reshape(group.size, -1)
            cost += ref_mat @ target_mat.T
        return cost


class UnsupportedGroupLinearization(ValueError):
    """Raised when an exact LAP update is mathematically invalid."""


__all__ = [
    "Objective",
    "L2WeightObjective",
    "UnsupportedGroupLinearization",
    "register_objective",
    "get_objective",
    "available_objectives",
]
