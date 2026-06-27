"""Canonical weight sample representation and artifact codecs."""

from __future__ import annotations

import pickle
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

from .sample_formats import DEFAULT_SAMPLE_FORMAT, normalize_sample_format

ParamTree: TypeAlias = Any


@dataclass(frozen=True)
class WeightSample:
    """Canonical in-memory representation of one weight sample."""

    params: ParamTree
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_params(self, params: ParamTree) -> WeightSample:
        """Return a sample with updated parameters and unchanged metadata."""

        return WeightSample(params=params, metadata=dict(self.metadata))


class WeightSampleCodec(Protocol):
    """Codec boundary between filesystem artifacts and ``WeightSample``."""

    name: str

    def load(self, path: str | Path) -> WeightSample:
        """Decode one sample artifact."""

    def save(self, path: str | Path, sample: WeightSample) -> None:
        """Encode one sample artifact."""

    def copy_structure(self, target_dir: str | Path, *, artifact_name: str) -> Path:
        """Copy any shared structure file required to reload saved samples."""


def tree_leaves_with_names(
    params: ParamTree,
) -> tuple[list[str], list[Any], jax.tree_util.PyTreeDef]:
    """Return flattened leaf names and leaves for ``params``."""

    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
    names: list[str] = []
    leaves: list[Any] = []
    for path, leaf in leaves_with_path:
        parts: list[str] = []
        for key in path:
            if isinstance(key, DictKey):
                parts.append(str(key.key))
            elif isinstance(key, GetAttrKey):
                parts.append(str(key.name))
            elif isinstance(key, SequenceKey):
                parts.append(str(key.idx))
            else:
                parts.append(str(key))
        names.append(".".join(parts))
        leaves.append(leaf)
    return names, leaves, treedef


class PyTreeNpzCodec:
    """Codec for SMILE/MILE-style leaf-order NPZ files plus a pickled PyTreeDef."""

    name = DEFAULT_SAMPLE_FORMAT

    def __init__(self, tree_path: str | Path) -> None:
        self.tree_path = Path(tree_path)
        self._treedef: jax.tree_util.PyTreeDef | None = None

    @property
    def treedef(self) -> jax.tree_util.PyTreeDef:
        if self._treedef is None:
            with self.tree_path.open("rb") as handle:
                self._treedef = pickle.load(handle)
        return self._treedef

    def load(self, path: str | Path) -> WeightSample:
        sample_path = Path(path)
        with np.load(sample_path, allow_pickle=False) as data:
            leaves = [jnp.asarray(data[key]) for key in data.files]
            leaf_keys = tuple(data.files)

        expected = int(self.treedef.num_leaves)
        if len(leaves) != expected:
            raise ValueError(
                f"Sample {sample_path} has {len(leaves)} leaves, "
                f"but tree {self.tree_path} expects {expected}."
            )
        params = jax.tree_util.tree_unflatten(self.treedef, leaves)
        return WeightSample(
            params=params,
            metadata={
                "format": self.name,
                "path": str(sample_path),
                "leaf_keys": leaf_keys,
            },
        )

    def save(self, path: str | Path, sample: WeightSample) -> None:
        sample_path = Path(path)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        names, leaves, _ = tree_leaves_with_names(sample.params)
        arrays = {
            name or f"leaf_{idx}": np.asarray(leaf)
            for idx, (name, leaf) in enumerate(zip(names, leaves, strict=True))
        }
        np.savez_compressed(sample_path, **arrays)

    def copy_structure(self, target_dir: str | Path, *, artifact_name: str) -> Path:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        if not self.tree_path.exists():
            raise FileNotFoundError(f"Tree path not found: {self.tree_path}")
        dest = target / artifact_name
        if not dest.exists() or not self.tree_path.samefile(dest):
            shutil.copy2(self.tree_path, dest)
        return dest


def create_sample_codec(
    sample_format: str | None,
    *,
    tree_path: str | Path | None,
) -> WeightSampleCodec:
    """Create the codec for a configured sample format."""

    normalized = normalize_sample_format(sample_format)
    if normalized == DEFAULT_SAMPLE_FORMAT:
        if tree_path is None:
            raise ValueError("tree_path is required for sample_format='pytree_npz'.")
        return PyTreeNpzCodec(tree_path)
    available = ", ".join([DEFAULT_SAMPLE_FORMAT])
    raise ValueError(f"Unknown sample format '{normalized}'. Available: {available}")


__all__ = [
    "ParamTree",
    "PyTreeNpzCodec",
    "WeightSample",
    "WeightSampleCodec",
    "create_sample_codec",
    "tree_leaves_with_names",
]
