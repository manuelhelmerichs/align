"""Low-level array and pytree primitives shared by the alignment core.

These helpers are backend-agnostic (NumPy or ``jax.numpy``) and underpin both the
graph problem (axis bindings, scale/permutation actions) and the architecture
adapters that emit it.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

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


def _perm_indices(
    perm: Any, *, direction: Literal["left", "right"]
):  # pragma: no cover - small helper
    xp = _array_backend(perm)
    perm_arr = xp.asarray(perm)
    # Accept either an index vector (already a permutation) or a permutation matrix.
    #
    # NOTE: All permutation application in this codebase is done via indexing
    # ("take" along an axis). For that usage, the same index order should be
    # applied regardless of whether the rule is tagged "left" or "right".
    #
    # For a permutation matrix P with P[i, j] = 1 meaning "new[i] = old[j]",
    # the corresponding indices are argmax over axis=1.
    if perm_arr.ndim == 1:
        return perm_arr.astype(int)
    if perm_arr.ndim != 2:
        raise ValueError(
            f"Expected permutation as 1D indices or 2D matrix, got ndim={perm_arr.ndim}."
        )
    return xp.argmax(perm_arr, axis=1)


def apply_perm_to_axis(
    tensor: Any, perm: Any, *, axis: int, direction: Literal["left", "right"]
):
    """Apply a permutation (matrix or indices) to ``tensor`` along ``axis``."""
    xp = _array_backend(tensor)
    perm_arr = xp.asarray(perm)
    if perm_arr.ndim == 1:
        indices = perm_arr.astype(int)
    else:
        indices = _perm_indices(perm_arr, direction=direction)
    moved = xp.moveaxis(tensor, axis, 0)
    permuted = moved[indices]
    return xp.moveaxis(permuted, 0, axis)


__all__ = ["apply_perm_to_axis"]
