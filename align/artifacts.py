"""Shared helpers for alignment artifact serialization."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_HARD_PERMUTATION_DTYPE = np.dtype(np.uint8)


def write_permutations_artifact(
    path: str | Path,
    perms: Sequence[Any] | Mapping[str, Any],
    *,
    dtype: np.dtype = _HARD_PERMUTATION_DTYPE,
) -> Path | None:
    """Persist final hard permutation matrices keyed by group id."""

    path = Path(path)
    if not perms:
        if path.exists():
            path.unlink()
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    if isinstance(perms, Mapping):
        items = perms.items()
    else:
        items = [(f"P{idx + 1}", matrix) for idx, matrix in enumerate(perms)]

    for key, matrix in items:
        array = np.asarray(matrix)
        encoded = (array > 0.5).astype(dtype, copy=False)
        payload[str(key)] = encoded
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
