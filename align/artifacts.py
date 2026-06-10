"""Shared helpers for rebasin artifact serialization."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_PERMUTATION_DTYPE_CACHE: dict[str, np.dtype] = {}


def _get_permutation_dtype(method: str) -> np.dtype:
    """Get permutation dtype for a method, using cache to avoid strategy instantiation."""
    method_lower = method.lower()
    if method_lower not in _PERMUTATION_DTYPE_CACHE:
        from .strategies import get_strategy

        strategy = get_strategy(method_lower)
        _PERMUTATION_DTYPE_CACHE[method_lower] = strategy.permutation_dtype
    return _PERMUTATION_DTYPE_CACHE[method_lower]


def write_permutations_artifact(
    path: str | Path, perms: Sequence[Any] | Mapping[str, Any], *, method: str
) -> Path | None:
    """Persist permutation matrices with consistent encoding for both runner and workers.

    The encoding dtype is determined by the strategy's permutation_dtype property.
    Uses a cache to avoid repeated strategy instantiation.
    """
    path = Path(path)
    if not perms:
        if path.exists():
            path.unlink()
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}

    dtype = _get_permutation_dtype(method)

    if isinstance(perms, Mapping):
        items = perms.items()
    else:
        items = [(f"P{idx + 1}", matrix) for idx, matrix in enumerate(perms)]

    for key, matrix in items:
        array = np.asarray(matrix)
        name = str(key)
        if dtype == np.uint8:
            encoded = (array > 0.5).astype(np.uint8, copy=False)
        else:
            encoded = array.astype(dtype, copy=False)
        payload[name] = encoded
    np.savez_compressed(path, **payload)
    return path


def write_aux_artifact(path: str | Path, aux: Mapping[str, Any] | None) -> Path | None:
    """Persist optional auxiliary metadata, deleting the file when payloads are empty."""

    path = Path(path)
    if not aux:
        if path.exists():
            path.unlink()
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aux, separators=(",", ":")))
    return path


def write_scale_factors_artifact(
    path: str | Path, scales: Sequence[Any]
) -> Path | None:
    """Persist scale factors for each hidden layer."""

    path = Path(path)
    if not scales:
        if path.exists():
            path.unlink()
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"S{idx + 1}": np.asarray(scale) for idx, scale in enumerate(scales)}
    np.savez_compressed(path, **payload)
    return path
