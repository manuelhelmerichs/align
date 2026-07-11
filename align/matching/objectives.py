"""Objectives for graph-native symmetry matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..symmetry import (
    GQARoPECircuitConstraint,
    MHACircuitConstraint,
    apply_matrix_to_axis,
    binding_axis_intervals,
)
from ..symmetry.tensor_ops import (
    binding_indexer,
    binding_selector,
    binding_transform_matrix,
)


class Objective(ABC):
    """Complete graph objective used by alignment schedulers."""

    name: str

    def begin_lap_execution(self, graph, target_data) -> None:
        """Create a sweep-local cache of target tensors with other groups applied."""

        self._lap_execution_cache = _LAPExecutionCache(graph, target_data)

    def end_lap_execution(self) -> None:
        self._lap_execution_cache = None

    def notify_group_updated(self, graph, group_id: str) -> None:
        cache = getattr(self, "_lap_execution_cache", None)
        if cache is not None:
            cache.invalidate_group(group_id, graph.tensors_for_group(group_id))

    def apply_other_groups_hard(
        self, graph, tensor_id: str, tensor, state, skip_groups
    ):
        cache = getattr(self, "_lap_execution_cache", None)
        if cache is None:
            return _apply_other_groups_hard(
                graph, tensor_id, tensor, state, skip_groups
            )
        return cache.get(tensor_id, state, skip_groups)

    @abstractmethod
    def value(self, graph, reference_data, target_data, state):
        """Return scalar or batched objective value."""

    def linearize_group(self, graph, reference_data, target_data, state, group_id: str):
        """Return an LAP maximization cost matrix for one group."""

        signed, invariant = self.linearize_group_split(
            graph, reference_data, target_data, state, group_id
        )
        return signed + invariant

    def linearize_group_split(
        self, graph, reference_data, target_data, state, group_id: str
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


class _LAPExecutionCache:
    """Incremental cache invalidated through graph adjacency after group updates."""

    def __init__(self, graph, target_data) -> None:
        self.graph = graph
        self.target_data = target_data
        self.values: dict[tuple[str, frozenset[str]], np.ndarray] = {}

    def get(self, tensor_id: str, state, skip_groups) -> np.ndarray:
        key = (tensor_id, frozenset(skip_groups))
        cached = self.values.get(key)
        if cached is None:
            cached = _apply_other_groups_hard(
                self.graph,
                tensor_id,
                self.target_data[tensor_id],
                state,
                key[1],
            )
            self.values[key] = cached
        return cached

    def invalidate_group(self, group_id: str, tensor_ids) -> None:
        """Drop only cached views that actually include the changed group."""

        touched = set(tensor_ids)
        self.values = {
            key: value
            for key, value in self.values.items()
            if key[0] not in touched or group_id in key[1]
        }


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


def _is_batched_tensor(graph, tensor_id: str, tensor) -> bool:
    return jnp.asarray(tensor).ndim == len(graph.tensors[tensor_id].shape) + 1


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


def _sorted_bindings(graph, tensor_id: str):
    return graph.sorted_bindings_for_tensor(tensor_id)


def _apply_state_to_tensor(graph, tensor_id: str, tensor, state):
    updated = jnp.asarray(tensor)
    shape = graph.tensors[tensor_id].shape
    for binding in _sorted_bindings(graph, tensor_id):
        tensor_is_batched = _is_batched_tensor(graph, tensor_id, updated)
        matrix = binding_transform_matrix(
            state.matrix(binding.group),
            transform_family=graph.groups[binding.group].transform_family,
            scope=binding.transform_scope,
        )
        if matrix is None:
            continue
        matrix_arr = jnp.asarray(matrix)
        if matrix_arr.ndim == 3 and not tensor_is_batched:
            updated = jnp.broadcast_to(updated, (matrix_arr.shape[0], *updated.shape))
            tensor_is_batched = True
        selector = binding_selector(shape, binding)
        offset = 1 if tensor_is_batched else 0
        intervals = binding_axis_intervals(shape, binding)
        axis = intervals[0][0] + offset
        axis_size = int(updated.shape[axis])
        if (
            len(intervals) == 1
            and not selector
            and intervals[0][1] == 0
            and intervals[0][2] == axis_size
        ):
            updated = _apply_matrix_axis(
                updated,
                matrix,
                axis=axis,
                tensor_is_batched=tensor_is_batched,
            )
            continue
        for spec_axis, start, stop in intervals:
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
    graph, tensor_id: str, tensor, state, skip_groups: frozenset[str] | set[str]
):
    """Apply all hard group actions except ``skip_groups`` to one tensor."""

    updated = np.asarray(tensor)
    shape = graph.tensors[tensor_id].shape
    for binding in _sorted_bindings(graph, tensor_id):
        if binding.group in skip_groups:
            continue
        matrix = binding_transform_matrix(
            np.asarray(state.matrix(binding.group)),
            transform_family=graph.groups[binding.group].transform_family,
            scope=binding.transform_scope,
        )
        if matrix is None:
            continue
        selector = binding_selector(shape, binding)
        intervals = binding_axis_intervals(shape, binding)
        axis = intervals[0][0]
        if (
            len(intervals) == 1
            and not selector
            and intervals[0][1] == 0
            and intervals[0][2] == int(updated.shape[axis])
        ):
            updated = apply_matrix_to_axis(updated, matrix, axis=axis)
            continue
        for axis, start, stop in intervals:
            indexer = binding_indexer(updated.ndim, axis, start, stop, selector)
            segment = updated[indexer]
            transformed = apply_matrix_to_axis(segment, matrix, axis=axis)
            updated = np.array(updated, copy=True)
            updated[indexer] = transformed
    return updated


def _require_lap_linearizable(graph, group_id: str) -> None:
    """Reject groups whose exact hard update is not a linear assignment."""

    for constraint in graph.constraints:
        if isinstance(constraint, MHACircuitConstraint) and (
            constraint.head_group == group_id
        ):
            raise UnsupportedGroupLinearization(
                f"Group {group_id!r} is an attention head group; its generic "
                "LAP linearization is inexact once intra-head permutations "
                "are non-identity. Use the structured attention update "
                "(lap schedules do this automatically) or a Sinkhorn "
                "schedule."
            )
        if isinstance(constraint, GQARoPECircuitConstraint) and (
            constraint.kv_group == group_id
        ):
            raise UnsupportedGroupLinearization(
                f"Group {group_id!r} is a GQA kv-group head group; its "
                "generic LAP linearization is inexact once per-slot "
                "query-head or vo permutations are non-identity. Use the "
                "structured GQA update (lap schedules do this automatically) "
                "or a Sinkhorn schedule."
            )

    repeated = graph.repeated_group_terms().get(group_id, ())
    if repeated:
        joined = ", ".join(repeated)
        raise UnsupportedGroupLinearization(
            f"Group {group_id!r} appears multiple times in tensor term(s) "
            f"{joined}. The exact hard update is a quadratic assignment "
            "problem, not an LAP. Use a Sinkhorn schedule for this update."
        )


@register_objective("euclidean")
class EuclideanObjective(Objective):
    """Sum of squared tensor differences after applying the alignment state."""

    name = "euclidean"

    def __init__(self, repeated_group_policy: str = "reject") -> None:
        if repeated_group_policy != "reject":
            raise ValueError(
                "Only repeated_group_policy='reject' is implemented for exact LAP "
                "linearization. Use a Sinkhorn schedule for repeated-group tensor terms."
            )
        self.repeated_group_policy = repeated_group_policy

    def value(self, graph, reference_data, target_data, state):
        loss = None
        for tensor_id in graph.tensors:
            ref = jnp.asarray(reference_data[tensor_id])
            target = jnp.asarray(target_data[tensor_id])
            aligned = _apply_state_to_tensor(graph, tensor_id, target, state)
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
        self, graph, reference_data, target_data, state, group_id: str
    ):
        _require_lap_linearizable(graph, group_id)

        group = graph.groups[group_id]
        signed = np.zeros((group.size, group.size), dtype=np.float64)
        invariant = np.zeros((group.size, group.size), dtype=np.float64)
        for tensor_id in graph.tensors_for_group(group_id):
            bindings = [
                binding
                for binding in graph.bindings_for_group(group_id)
                if binding.tensor_id == tensor_id
            ]
            if not bindings:
                continue
            # Multiple bindings on one axis (e.g. a conv->flatten boundary
            # binding each spatial position's channel slice) contribute a sum
            # of independent linear terms; multi-axis repeats are rejected by
            # the linearizability check above.
            shape = graph.tensors[tensor_id].shape
            ref = np.asarray(reference_data[tensor_id])
            target = self.apply_other_groups_hard(
                graph, tensor_id, target_data[tensor_id], state, {group_id}
            )
            for binding in bindings:
                selector = binding_selector(shape, binding)
                for axis, start, stop in binding_axis_intervals(shape, binding):
                    indexer = binding_indexer(ref.ndim, axis, start, stop, selector)
                    ref_mat = np.moveaxis(ref[indexer], axis, 0).reshape(group.size, -1)
                    target_mat = np.moveaxis(target[indexer], axis, 0).reshape(
                        group.size, -1
                    )
                    bucket = (
                        invariant
                        if binding.transform_scope == "permute_only"
                        else signed
                    )
                    bucket += ref_mat @ target_mat.T
        return signed, invariant


@register_objective("diagonal_fisher")
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

    name = "diagonal_fisher"

    def __init__(self, tensor_weights=None, weights_path=None) -> None:
        if (tensor_weights is None) == (weights_path is None):
            raise ValueError(
                "diagonal_fisher requires exactly one of 'tensor_weights' (mapping "
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
                    f"diagonal_fisher weights for tensor {tensor_id!r} must be non-negative."
                )

    def _weight(self, graph, tensor_id: str) -> np.ndarray:
        weight = self.tensor_weights.get(tensor_id)
        if weight is None:
            raise ValueError(
                f"diagonal_fisher has no weights for tensor {tensor_id!r}; weights "
                "must cover every graph tensor."
            )
        expected = tuple(graph.tensors[tensor_id].shape)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"diagonal_fisher weights for tensor {tensor_id!r} have shape "
                f"{weight.shape}, expected {expected}."
            )
        return weight

    def value(self, graph, reference_data, target_data, state):
        loss = None
        for tensor_id in graph.tensors:
            ref = jnp.asarray(reference_data[tensor_id])
            target = jnp.asarray(target_data[tensor_id])
            weight = jnp.asarray(self._weight(graph, tensor_id), dtype=ref.dtype)
            aligned = _apply_state_to_tensor(graph, tensor_id, target, state)
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
        self, graph, reference_data, target_data, state, group_id: str
    ):
        _require_lap_linearizable(graph, group_id)

        group = graph.groups[group_id]
        signed = np.zeros((group.size, group.size), dtype=np.float64)
        invariant = np.zeros((group.size, group.size), dtype=np.float64)
        for tensor_id in graph.tensors_for_group(group_id):
            bindings = [
                binding
                for binding in graph.bindings_for_group(group_id)
                if binding.tensor_id == tensor_id
            ]
            if not bindings:
                continue
            # Same-axis multi-bindings sum independent linear terms (see the
            # L2 linearization); multi-axis repeats are rejected above.
            shape = graph.tensors[tensor_id].shape
            ref = np.asarray(reference_data[tensor_id])
            weight = self._weight(graph, tensor_id)
            target = self.apply_other_groups_hard(
                graph, tensor_id, target_data[tensor_id], state, {group_id}
            )
            for binding in bindings:
                selector = binding_selector(shape, binding)
                for axis, start, stop in binding_axis_intervals(shape, binding):
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
                        invariant
                        if binding.transform_scope == "permute_only"
                        else signed
                    )
                    bucket += cross
                    invariant -= 0.5 * weight_mat @ np.square(target_mat).T
        return signed, invariant


@register_objective("relative_fisher")
class RelativeFisherObjective(Objective):
    """Activation-Gram metric on each tensor's incoming coordinates.

    A tensor residual ``D`` with flattened incoming coordinate ``a`` is scored
    as ``sum_r D[:, r].T @ G @ D[:, r]``.  Here ``G = E[a a.T]`` is fixed at
    the reference and ``r`` indexes all remaining tensor axes.  The full Gram
    makes soft input- and output-axis assignments interact quadratically, so
    this objective intentionally supports Sinkhorn only and does not expose an
    LAP linearization.
    """

    name = "relative_fisher"

    def __init__(self, tensor_metrics=None, metrics_path=None) -> None:
        if (tensor_metrics is None) == (metrics_path is None):
            raise ValueError(
                "relative_fisher requires exactly one of 'tensor_metrics' or "
                "'metrics_path'."
            )
        if metrics_path is not None:
            from .relative_fisher import load_activation_gram_metrics_npz

            tensor_metrics = load_activation_gram_metrics_npz(metrics_path)
        self.tensor_metrics = {
            str(tensor_id): {
                "input_axes": tuple(int(axis) for axis in metric["input_axes"]),
                "gram": np.asarray(metric["gram"], dtype=np.float64),
            }
            for tensor_id, metric in dict(tensor_metrics).items()
        }
        self._validated_metrics = {}

    def _metric(self, graph, tensor_id: str):
        metric = self.tensor_metrics.get(tensor_id)
        if metric is None:
            raise ValueError(
                f"relative_fisher has no metric for tensor {tensor_id!r}; metrics "
                "must cover every graph tensor."
            )
        shape = tuple(graph.tensors[tensor_id].shape)
        cache_key = (tensor_id, shape)
        if cache_key in self._validated_metrics:
            return self._validated_metrics[cache_key]
        ndim = len(shape)
        invalid = [
            axis for axis in metric["input_axes"] if axis < -ndim or axis >= ndim
        ]
        if invalid:
            raise ValueError(
                f"relative_fisher input axes for tensor {tensor_id!r} are out of "
                f"bounds for {ndim} dimensions: {invalid}."
            )
        axes = tuple(axis % ndim for axis in metric["input_axes"])
        if len(set(axes)) != len(axes):
            raise ValueError(
                f"relative_fisher input axes for tensor {tensor_id!r} are not unique."
            )
        size = (
            int(np.prod([shape[axis] for axis in axes], dtype=np.int64)) if axes else 1
        )
        gram = metric["gram"]
        if gram.shape != (size, size):
            raise ValueError(
                f"relative_fisher Gram for tensor {tensor_id!r} has shape "
                f"{gram.shape}, expected {(size, size)} for input axes {axes}."
            )
        if not np.all(np.isfinite(gram)):
            raise ValueError(
                f"relative_fisher Gram for tensor {tensor_id!r} must be finite."
            )
        if not np.allclose(gram, gram.T, rtol=1e-7, atol=1e-10):
            raise ValueError(
                f"relative_fisher Gram for tensor {tensor_id!r} must be symmetric."
            )
        minimum = float(np.min(np.linalg.eigvalsh(gram)))
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(gram))))
        if minimum < -tolerance:
            raise ValueError(
                f"relative_fisher Gram for tensor {tensor_id!r} must be positive "
                f"semidefinite (minimum eigenvalue {minimum:.3g})."
            )
        self._validated_metrics[cache_key] = (axes, gram)
        return axes, gram

    @staticmethod
    def _quadratic(diff, axes, gram, *, tensor_ndim: int):
        batched = diff.ndim == tensor_ndim + 1
        offset = 1 if batched else 0
        metric_axes = tuple(axis + offset for axis in axes)
        batch_axes = (0,) if batched else ()
        other_axes = tuple(
            axis for axis in range(offset, diff.ndim) if axis not in metric_axes
        )
        order = (*batch_axes, *metric_axes, *other_axes)
        moved = jnp.transpose(diff, order) if order else diff
        input_size = (
            int(np.prod([diff.shape[axis] for axis in metric_axes])) if axes else 1
        )
        if batched:
            flat = moved.reshape((moved.shape[0], input_size, -1))
            return jnp.einsum("bir,ij,bjr->b", flat, gram, flat, optimize=True)
        flat = moved.reshape((input_size, -1))
        return jnp.einsum("ir,ij,jr->", flat, gram, flat, optimize=True)

    def value(self, graph, reference_data, target_data, state):
        unknown = sorted(set(self.tensor_metrics) - set(graph.tensors))
        if unknown:
            raise ValueError(
                "relative_fisher has metrics for unknown graph tensor(s): "
                + ", ".join(unknown)
            )
        loss = None
        for tensor_id, tensor_spec in graph.tensors.items():
            ref = jnp.asarray(reference_data[tensor_id])
            target = jnp.asarray(target_data[tensor_id])
            aligned = _apply_state_to_tensor(graph, tensor_id, target, state)
            diff = (
                ref[None, ...] - aligned
                if aligned.ndim == ref.ndim + 1
                else ref - aligned
            )
            axes, gram = self._metric(graph, tensor_id)
            term = self._quadratic(
                diff,
                axes,
                jnp.asarray(gram, dtype=ref.dtype),
                tensor_ndim=len(tensor_spec.shape),
            )
            loss = term if loss is None else loss + term
        return jnp.asarray(0.0) if loss is None else loss

    def linearize_group(self, graph, reference_data, target_data, state, group_id: str):
        del graph, reference_data, target_data, state, group_id
        raise UnsupportedGroupLinearization(
            "relative_fisher is Sinkhorn-only: its full activation Gram makes "
            "the assignment-dependent target self-energy a quadratic assignment term."
        )

    def linearize_group_split(
        self, graph, reference_data, target_data, state, group_id: str
    ):
        return self.linearize_group(graph, reference_data, target_data, state, group_id)


class UnsupportedGroupLinearization(ValueError):
    """Raised when an exact LAP update is mathematically invalid."""


__all__ = [
    "Objective",
    "EuclideanObjective",
    "DiagonalFisherObjective",
    "RelativeFisherObjective",
    "UnsupportedGroupLinearization",
    "register_objective",
    "get_objective",
    "available_objectives",
]
