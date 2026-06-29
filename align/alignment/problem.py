"""Alignment graph data model and tensor materialization."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict

from .tensor_ops import (
    _canonical_axis,
    _descend,
    _set_path,
    apply_perm_to_axis,
    axis_slice,
    binding_axis_interval,
)


@dataclass
class PermutationGroup:
    """A set of channels/units that must share a permutation."""

    id: str
    size: int


@dataclass(frozen=True)
class TensorSpec:
    """Logical tensor participating in alignment."""

    id: str
    path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype_role: str = "param"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AxisBinding:
    """Bind one tensor axis to one permutation group.

    ``role`` records how the bound channel relates to the group for *scale*
    actions: an ``"out"`` axis produces the channel (kernel output axis and its
    attached bias) and is divided by the group's scale, while an ``"in"`` axis
    consumes the channel (the next tensor's input axis) and is multiplied by it.
    Permutation actions are role-independent and ignore this field.
    """

    tensor_id: str
    axis: int
    group: str
    start: int | None = None
    stop: int | None = None
    role: Literal["out", "in"] = "out"


@dataclass(frozen=True)
class GraphConstraint:
    """Graph-level architecture constraint recorded beside axis bindings."""

    kind: str
    groups: tuple[str, ...] = ()
    tensors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _array_namespace(backend: Literal["jax", "numpy"]):
    return jnp if backend == "jax" else np


def _set_axis_slice(tensor: Any, indexer: tuple[slice, ...], value: Any):
    if isinstance(tensor, jnp.ndarray):
        return tensor.at[indexer].set(value)
    updated = np.array(tensor, copy=True)
    updated[indexer] = value
    return updated


def _scale_axis(tensor: Any, scale: Any, *, axis: int, divide: bool) -> Any:
    """Multiply or divide ``tensor`` along ``axis`` by a per-channel ``scale``."""

    xp = jnp if isinstance(tensor, jnp.ndarray) else np
    factor = xp.asarray(scale)
    broadcast_shape = [1] * xp.ndim(tensor)
    broadcast_shape[axis] = factor.shape[0]
    factor = factor.reshape(broadcast_shape)
    return tensor / factor if divide else tensor * factor


class MaterializedTensors(Mapping[str, Any]):
    """Lazy mapping from logical tensor id to backend array."""

    def __init__(
        self,
        *,
        problem: AlignmentProblem,
        params: Mapping[str, Any],
        backend: Literal["jax", "numpy"],
        cache: bool,
    ) -> None:
        self._problem = problem
        self._params = params
        self._xp = _array_namespace(backend)
        self._cache_enabled = bool(cache)
        self._cache: dict[str, Any] = {}

    def __getitem__(self, tensor_id: str) -> Any:
        if tensor_id not in self._problem.tensors:
            raise KeyError(tensor_id)
        if self._cache_enabled and tensor_id in self._cache:
            return self._cache[tensor_id]
        spec = self._problem.tensors[tensor_id]
        value = self._xp.asarray(_descend(self._params, spec.path))
        if tuple(int(dim) for dim in value.shape) != tuple(spec.shape):
            raise ValueError(
                f"Tensor {tensor_id!r} at {spec.path} has shape {value.shape}, "
                f"expected {spec.shape}."
            )
        if self._cache_enabled:
            self._cache[tensor_id] = value
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._problem.tensors)

    def __len__(self) -> int:
        return len(self._problem.tensors)

    def materialize(self) -> dict[str, Any]:
        """Return all tensors as a concrete dictionary."""

        return {tensor_id: self[tensor_id] for tensor_id in self}

    def clear_cache(self) -> None:
        self._cache.clear()


@dataclass
class AlignmentProblem:
    """Architecture adapter output consumed by graph-native rebasin solvers."""

    groups: dict[str, PermutationGroup]
    tensors: dict[str, TensorSpec]
    axis_bindings: tuple[AxisBinding, ...] = ()
    constraints: tuple[GraphConstraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.groups = dict(self.groups)
        self.tensors = dict(self.tensors)
        self.axis_bindings = tuple(self.axis_bindings)
        self.constraints = tuple(self.constraints)
        self.metadata = dict(self.metadata)

    @property
    def group_order(self) -> tuple[str, ...]:
        """Stable group order used by schedulers and artifact serialization."""

        metadata_order = self.metadata.get("group_order")
        if metadata_order is not None:
            ordered = tuple(str(gid) for gid in metadata_order)
            if set(ordered) != set(self.groups):
                raise ValueError(
                    "metadata['group_order'] must contain exactly the problem groups."
                )
            return ordered
        return tuple(self.groups)

    def bindings_for_tensor(self, tensor_id: str) -> tuple[AxisBinding, ...]:
        return tuple(
            binding for binding in self.axis_bindings if binding.tensor_id == tensor_id
        )

    def bindings_for_group(self, group_id: str) -> tuple[AxisBinding, ...]:
        return tuple(
            binding for binding in self.axis_bindings if binding.group == group_id
        )

    def repeated_group_terms(self) -> dict[str, tuple[str, ...]]:
        """Return tensor ids where a group binds more than once."""

        repeated: dict[str, list[str]] = {}
        for tensor_id, tensor in self.tensors.items():
            axes_by_group: dict[str, set[int]] = {}
            for binding in self.bindings_for_tensor(tensor_id):
                axis, _, _ = binding_axis_interval(tensor.shape, binding)
                axes_by_group.setdefault(binding.group, set()).add(axis)
            for group_id, axes in axes_by_group.items():
                if len(axes) > 1:
                    repeated.setdefault(group_id, []).append(tensor_id)
        return {group_id: tuple(tensors) for group_id, tensors in repeated.items()}

    def materialize(
        self,
        params: Mapping[str, Any],
        *,
        backend: Literal["jax", "numpy"] = "jax",
        cache: bool = True,
    ) -> MaterializedTensors:
        """Lazily materialize tensors for one parameter tree."""

        return MaterializedTensors(
            problem=self,
            params=params,
            backend=backend,
            cache=cache,
        )

    def validate(self, params: Mapping[str, Any] | None = None) -> None:
        """Validate tensor existence, axis sizes, constraints, and group order."""

        for group_id, group in self.groups.items():
            if group.id != group_id:
                raise ValueError(
                    f"Group mapping key {group_id!r} does not match id {group.id!r}."
                )
            if int(group.size) <= 0:
                raise ValueError(f"Group {group_id!r} has non-positive size.")

        for tensor_id, tensor in self.tensors.items():
            if tensor.id != tensor_id:
                raise ValueError(
                    f"Tensor mapping key {tensor_id!r} does not match id {tensor.id!r}."
                )
            if not tensor.path:
                raise ValueError(f"Tensor {tensor_id!r} has an empty path.")
            if any(int(dim) <= 0 for dim in tensor.shape):
                raise ValueError(
                    f"Tensor {tensor_id!r} has invalid shape {tensor.shape}."
                )
            if params is not None:
                actual = _descend(params, tensor.path)
                actual_shape = tuple(int(dim) for dim in np.shape(actual))
                if actual_shape != tuple(tensor.shape):
                    raise ValueError(
                        f"Tensor {tensor_id!r} at {tensor.path} has shape "
                        f"{actual_shape}, expected {tensor.shape}."
                    )

        bound_segments: dict[tuple[str, int], list[tuple[int, int, str]]] = {}
        for binding in self.axis_bindings:
            if binding.tensor_id not in self.tensors:
                raise ValueError(
                    f"Axis binding references unknown tensor {binding.tensor_id!r}."
                )
            if binding.group not in self.groups:
                raise ValueError(
                    f"Axis binding references unknown group {binding.group!r}."
                )
            if binding.role not in ("out", "in"):
                raise ValueError(
                    f"Axis binding for tensor {binding.tensor_id!r} has invalid role "
                    f"{binding.role!r}; expected 'out' or 'in'."
                )
            tensor = self.tensors[binding.tensor_id]
            axis, start, stop = binding_axis_interval(tensor.shape, binding)
            axis_key = (binding.tensor_id, axis)
            for prev_start, prev_stop, previous_group in bound_segments.get(
                axis_key, []
            ):
                if max(start, prev_start) < min(stop, prev_stop):
                    raise ValueError(
                        f"Tensor {binding.tensor_id!r} axis {binding.axis} interval "
                        f"[{start}, {stop}) overlaps an existing binding to group "
                        f"{previous_group!r}; cannot also bind it to group "
                        f"{binding.group!r}."
                    )
            bound_segments.setdefault(axis_key, []).append((start, stop, binding.group))
            group = self.groups[binding.group]
            interval_size = stop - start
            if interval_size != int(group.size):
                raise ValueError(
                    f"Tensor {binding.tensor_id!r} axis {binding.axis} interval "
                    f"[{start}, {stop}) has size {interval_size}, expected group "
                    f"{binding.group!r} size {group.size}."
                )

        for constraint in self.constraints:
            for group_id in constraint.groups:
                if group_id not in self.groups:
                    raise ValueError(
                        f"Constraint {constraint.kind!r} references unknown group "
                        f"{group_id!r}."
                    )
            for tensor_id in constraint.tensors:
                if tensor_id not in self.tensors:
                    raise ValueError(
                        f"Constraint {constraint.kind!r} references unknown tensor "
                        f"{tensor_id!r}."
                    )
            if constraint.kind == "residual_tie":
                if len(set(constraint.groups)) != 1:
                    raise ValueError(
                        "residual_tie constraints must be canonicalized to exactly "
                        "one shared group."
                    )

        _ = self.group_order

    def _transform_bound_tensors(
        self,
        params: Mapping[str, Any],
        transform,
    ) -> Mapping[str, Any]:
        """Walk every bound tensor and apply ``transform`` per binding.

        ``transform(segment, axis, binding)`` returns the transformed axis-slice
        (or whole axis). Shared by :meth:`apply` (permutation) and
        :meth:`apply_scales` (scaling) so the binding-walk and slice handling
        cannot drift between the two graph actions. ``FrozenDict`` inputs are
        preserved.
        """

        is_frozen = isinstance(params, frozen_dict.FrozenDict)
        mutable = frozen_dict.unfreeze(params) if is_frozen else copy.deepcopy(params)
        for tensor_id, tensor in self.tensors.items():
            bindings = self.bindings_for_tensor(tensor_id)
            if not bindings:
                continue
            updated = _descend(mutable, tensor.path)
            for binding in sorted(
                bindings,
                key=lambda item: (
                    _canonical_axis(len(tensor.shape), item.axis),
                    item.group,
                ),
            ):
                axis, start, stop = binding_axis_interval(tensor.shape, binding)
                if start == 0 and stop == int(tensor.shape[axis]):
                    updated = transform(updated, axis, binding)
                    continue
                indexer = axis_slice(np.ndim(updated), axis, start, stop)
                segment = updated[indexer]
                transformed = transform(segment, axis, binding)
                updated = _set_axis_slice(updated, indexer, transformed)
            _set_path(mutable, tensor.path, updated)
        return frozen_dict.freeze(mutable) if is_frozen else mutable

    def apply(self, params: Mapping[str, Any], state) -> Mapping[str, Any]:
        """Apply a hard permutation state to all bound tensor axes."""

        state.validate(self, hard=True)

        def _permute(segment, axis, binding):
            return apply_perm_to_axis(
                segment,
                state.hard[binding.group],
                axis=axis,
            )

        return self._transform_bound_tensors(params, _permute)

    def apply_scales(self, params: Mapping[str, Any], scale_state) -> Mapping[str, Any]:
        """Apply a per-group positive scale to all bound tensor axes.

        ``out``-role axes (the channel producer kernel and its bias) are divided
        by the group's scale; ``in``-role axes (consumer input axes) are
        multiplied by it. For ReLU-style positive-homogeneous activations this is
        an exact function-preserving symmetry.
        """

        scale_state.validate(self)

        def _scale(segment, axis, binding):
            return _scale_axis(
                segment,
                scale_state.scales[binding.group],
                axis=axis,
                divide=binding.role == "out",
            )

        return self._transform_bound_tensors(params, _scale)


def materialize_many(
    problem: AlignmentProblem,
    params_batch: Sequence[Mapping[str, Any]],
    *,
    backend: Literal["jax", "numpy"] = "jax",
) -> dict[str, Any]:
    """Materialize and stack tensors for a sequence of target samples."""

    xp = _array_namespace(backend)
    per_sample = [
        problem.materialize(params, backend=backend, cache=False).materialize()
        for params in params_batch
    ]
    if not per_sample:
        return {}
    return {
        tensor_id: xp.stack([sample[tensor_id] for sample in per_sample], axis=0)
        for tensor_id in problem.tensors
    }


__all__ = [
    "AlignmentProblem",
    "AxisBinding",
    "GraphConstraint",
    "MaterializedTensors",
    "TensorSpec",
    "binding_axis_interval",
    "materialize_many",
]
