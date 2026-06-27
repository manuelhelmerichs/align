"""Architecture adapters and alignment helpers."""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass
class PermutationGroup:
    """A set of channels/units that must share a permutation."""

    id: str
    size: int


class ArchitectureAdapter(ABC):
    """Interface for architecture-aware alignment.

    Adapters emit a single graph-native :class:`~align.alignment.AlignmentProblem`
    per parameter tree. Both stages consume it: rebasin via permutation actions
    and normalization via the scale action.
    """

    name: str

    @abstractmethod
    def build_problem(self, params: Mapping[str, Any]):
        """Return the graph-native alignment problem for ``params``."""


_ADAPTERS: dict[str, type[ArchitectureAdapter]] = {}
_BUILTIN_ADAPTERS: dict[str, tuple[str, str]] = {
    "dense_mlp": ("align.architectures.dense_mlp", "DenseMLPAdapter"),
    "resnet": ("align.architectures.resnet", "ResNetAdapter"),
}


def register_adapter(cls: type[ArchitectureAdapter]) -> type[ArchitectureAdapter]:
    """Decorator to register an ArchitectureAdapter implementation."""

    if not getattr(cls, "name", None):
        raise ValueError("Adapter classes must define a 'name' attribute.")
    _ADAPTERS[cls.name] = cls
    return cls


def _load_builtin_adapter(name: str) -> type[ArchitectureAdapter] | None:
    """Import and return the requested built-in adapter class, if known."""
    module_spec = _BUILTIN_ADAPTERS.get(name)
    if module_spec is None:
        return None

    module_name, class_name = module_spec
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    if not isinstance(adapter_cls, type) or not issubclass(
        adapter_cls, ArchitectureAdapter
    ):
        raise TypeError(
            f"Built-in adapter '{name}' resolved to invalid class {class_name!r}."
        )
    _ADAPTERS.setdefault(name, adapter_cls)
    return adapter_cls


def get_adapter(name: str, **kwargs: Any) -> ArchitectureAdapter:
    adapter_cls = _ADAPTERS.get(name)
    if adapter_cls is None:
        adapter_cls = _load_builtin_adapter(name)
    if adapter_cls is None:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown architecture adapter '{name}'. Available: {available_adapters()}"
        )
    return adapter_cls(**kwargs)


def available_adapters() -> list[str]:
    return sorted(set(_BUILTIN_ADAPTERS) | set(_ADAPTERS))


__all__ = [
    "ArchitectureAdapter",
    "PermutationGroup",
    "available_adapters",
    "get_adapter",
    "register_adapter",
    "apply_perm_to_axis",
]
