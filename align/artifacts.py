"""Shared helpers for alignment artifact serialization."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def write_transforms_artifact(
    path: str | Path,
    transforms: Mapping[str, Any],
) -> Path | None:
    """Persist final hard group transforms keyed by group id.

    ``transforms`` is the self-describing mapping produced by
    :meth:`align.matching.TransformState.to_artifacts`: plain permutations are
    already compact ``uint8`` while signed permutations, rotation-pair blocks,
    and orthogonal matrices are ``float32``. The matrices are stored verbatim —
    unlike the retired permutation writer, nothing is thresholded to binary, so
    every transform family round-trips exactly.
    """

    path = Path(path)
    if not transforms:
        if path.exists():
            path.unlink()
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(group_id): np.asarray(matrix) for group_id, matrix in transforms.items()
    }
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
