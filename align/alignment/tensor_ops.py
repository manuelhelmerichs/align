"""Low-level array and pytree primitives shared by the alignment core.

These helpers are backend-agnostic (NumPy or ``jax.numpy``) and underpin both the
graph problem (axis bindings, scale/permutation actions) and the architecture
adapters that emit it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np


def _descend(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    """Traverse ``mapping`` following ``path`` keys."""
    node: Any = mapping
    for key in path:
        node = node[key]  # type: ignore[index]
    return node


def _maybe_descend(mapping: Mapping[str, Any], path: Sequence[str]) -> Any | None:
    """Traverse ``mapping`` following ``path`` keys, returning None if missing."""
    node: Any = mapping
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def _set_path(
    mapping: dict[str, Any], path: Sequence[str], value: Any
) -> None:  # pragma: no cover - trivial
    node: dict[str, Any] = mapping
    for key in path[:-1]:
        node = node[key]  # type: ignore[index]
    node[path[-1]] = value


def _array_backend(arr: Any):
    """Return the array namespace (numpy or jax.numpy) matching ``arr``."""
    if isinstance(arr, jnp.ndarray):
        return jnp
    return np


def _canonical_axis(ndim: int, axis: int) -> int:
    """Normalize possibly-negative ``axis`` and validate the result."""
    if axis < 0:
        axis = ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Axis {axis} is out of bounds for ndim={ndim}.")
    return axis


def binding_axis_interval(shape: Sequence[int], binding: Any) -> tuple[int, int, int]:
    """Return canonical ``(axis, start, stop)`` for a tensor-axis binding."""

    axis = _canonical_axis(len(shape), binding.axis)
    axis_size = int(shape[axis])
    start = 0 if binding.start is None else int(binding.start)
    stop = axis_size if binding.stop is None else int(binding.stop)
    if start < 0:
        start += axis_size
    if stop < 0:
        stop += axis_size
    if start < 0 or stop > axis_size or start >= stop:
        raise ValueError(
            f"Axis binding for tensor {binding.tensor_id!r} uses invalid interval "
            f"[{binding.start}, {binding.stop}) on axis size {axis_size}."
        )
    return axis, start, stop


def binding_selector(shape: Sequence[int], binding: Any) -> tuple[tuple[int, int], ...]:
    """Return canonical ``((axis, index), ...)`` selector coordinates.

    A selector restricts a binding to a fixed index on other axes (e.g. one
    attention head slot). Coordinates are canonicalized against ``shape`` and
    sorted by axis.
    """

    raw = tuple(getattr(binding, "selector", ()) or ())
    canonical: list[tuple[int, int]] = []
    seen: set[int] = set()
    for sel_axis, sel_index in raw:
        axis = _canonical_axis(len(shape), int(sel_axis))
        index = int(sel_index)
        if index < 0:
            index += int(shape[axis])
        if index < 0 or index >= int(shape[axis]):
            raise ValueError(
                f"Axis binding for tensor {binding.tensor_id!r} uses selector "
                f"index {sel_index} out of range for axis {sel_axis} "
                f"(size {shape[axis]})."
            )
        if axis in seen:
            raise ValueError(
                f"Axis binding for tensor {binding.tensor_id!r} repeats selector "
                f"axis {sel_axis}."
            )
        seen.add(axis)
        canonical.append((axis, index))
    return tuple(sorted(canonical))


def binding_sort_key(shape: Sequence[int], binding: Any) -> tuple[int, int, str]:
    """Deterministic application order for one tensor's bindings.

    Selector-free bindings apply first (in axis order), selector-carrying
    bindings afterwards. This fixes the composition convention: a selector
    coordinate refers to the axis position *after* the selector-free bindings
    (e.g. the head permutation) have been applied, i.e. reference-slot space.
    """

    selector = tuple(getattr(binding, "selector", ()) or ())
    return (
        1 if selector else 0,
        _canonical_axis(len(shape), binding.axis),
        binding.group,
    )


def axis_slice(ndim: int, axis: int, start: int, stop: int) -> tuple[slice, ...]:
    """Return an index tuple selecting ``[start, stop)`` along one axis."""

    indexer = [slice(None)] * ndim
    indexer[axis] = slice(start, stop)
    return tuple(indexer)


def binding_indexer(
    ndim: int,
    axis: int,
    start: int,
    stop: int,
    selector: Sequence[tuple[int, int]] = (),
    *,
    offset: int = 0,
) -> tuple[slice, ...]:
    """Index tuple selecting a binding's sub-slice, preserving dimensionality.

    Selector axes use length-1 slices so axis numbering inside the selected
    segment matches the full tensor. ``offset`` shifts all axes (for a leading
    batch dimension).
    """

    indexer = [slice(None)] * ndim
    indexer[axis + offset] = slice(start, stop)
    for sel_axis, sel_index in selector:
        indexer[sel_axis + offset] = slice(sel_index, sel_index + 1)
    return tuple(indexer)


def apply_perm_to_axis(tensor: Any, perm: Any, *, axis: int):
    """Apply a permutation (matrix or indices) to ``tensor`` along ``axis``."""
    xp = _array_backend(tensor)
    perm_arr = xp.asarray(perm)
    if perm_arr.ndim == 1:
        indices = perm_arr.astype(int)
    elif perm_arr.ndim == 2:
        indices = xp.argmax(perm_arr, axis=1)
    else:
        raise ValueError(
            f"Expected permutation as 1D indices or 2D matrix, got ndim={perm_arr.ndim}."
        )
    moved = xp.moveaxis(tensor, axis, 0)
    permuted = moved[indices]
    return xp.moveaxis(permuted, 0, axis)


def apply_matrix_to_axis(tensor: Any, matrix: Any, *, axis: int):
    """Apply a group transform (indices or 2-D matrix) along ``axis``.

    Index vectors take the permutation fast path; 2-D matrices are applied by
    matrix multiplication (``new_i = sum_j M[i, j] old_j``), which is exact
    for permutation matrices and required for signed permutations, rotations,
    and orthogonal transforms.
    """

    xp = _array_backend(tensor)
    arr = xp.asarray(matrix)
    if arr.ndim == 1:
        return apply_perm_to_axis(tensor, arr, axis=axis)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected transform as 1D indices or 2D matrix, got ndim={arr.ndim}."
        )
    moved = xp.moveaxis(tensor, axis, 0)
    flat = moved.reshape((moved.shape[0], -1))
    out = arr.astype(flat.dtype) @ flat
    return xp.moveaxis(out.reshape(moved.shape), 0, axis)


def binding_transform_matrix(matrix: Any, *, transforms: str, scope: str) -> Any | None:
    """Return the effective matrix for one binding, or ``None`` to skip.

    ``scope == "linear"`` bindings carry the full group transform.
    ``scope == "permute_only"`` bindings carry only its permutation content
    (RMSNorm scales permute with the stream but are exempt from signs and
    orthogonal maps, whose action their consumers absorb): under a
    ``signed_permutation`` group the entrywise absolute value is applied
    (also a no-op for nonnegative soft relaxations), and under an
    ``orthogonal`` group the binding is skipped entirely — exact only when
    the bound tensor is the folded constant, which the solver preflight
    enforces. The decision is class-based, never value-based, so it is safe
    under JAX tracing.
    """

    if scope == "linear":
        return matrix
    if scope != "permute_only":
        raise ValueError(f"Unknown binding transform scope {scope!r}.")
    arr = matrix if hasattr(matrix, "ndim") else np.asarray(matrix)
    if arr.ndim == 1 or transforms == "permutation":
        return matrix
    if transforms == "signed_permutation":
        return _array_backend(arr).abs(arr)
    if transforms == "orthogonal":
        return None
    raise ValueError(f"permute_only bindings are undefined for {transforms!r} groups.")


__all__ = [
    "apply_matrix_to_axis",
    "apply_perm_to_axis",
    "axis_slice",
    "binding_axis_interval",
    "binding_indexer",
    "binding_selector",
    "binding_sort_key",
    "binding_transform_matrix",
]
