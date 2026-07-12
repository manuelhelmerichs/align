"""Canonical weight sample representation and the NPZ artifact codec."""

from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

from .run_state import write_npz_compressed

type ParamTree = Any


def load_pytree_definition(path: str | Path) -> jax.tree_util.PyTreeDef:
    """Load a pickled JAX PyTree definition from a producer artifact."""

    with Path(path).open("rb") as handle:
        treedef = pickle.load(handle)
    if not isinstance(treedef, jax.tree_util.PyTreeDef):
        raise TypeError(
            f"Expected a JAX PyTreeDef in {path}, got {type(treedef).__name__}."
        )
    return treedef


@dataclass(frozen=True)
class WeightSample:
    """Canonical in-memory representation of one weight sample."""

    params: ParamTree


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

    def __init__(self, tree_path: str | Path) -> None:
        self.tree_path = Path(tree_path)
        self._treedef: jax.tree_util.PyTreeDef | None = None

    @property
    def treedef(self) -> jax.tree_util.PyTreeDef:
        if self._treedef is None:
            self._treedef = load_pytree_definition(self.tree_path)
        return self._treedef

    def load(self, path: str | Path) -> WeightSample:
        sample_path = Path(path)
        with np.load(sample_path, allow_pickle=False) as data:
            leaves = [jnp.asarray(data[key]) for key in data.files]

        expected = int(self.treedef.num_leaves)
        if len(leaves) != expected:
            raise ValueError(
                f"Sample {sample_path} has {len(leaves)} leaves, "
                f"but tree {self.tree_path} expects {expected}."
            )
        params = jax.tree_util.tree_unflatten(self.treedef, leaves)
        return WeightSample(params=params)

    def save(self, path: str | Path, sample: WeightSample) -> int:
        sample_path = Path(path)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        names, leaves, _ = tree_leaves_with_names(sample.params)
        arrays = {
            name or f"leaf_{idx}": np.asarray(leaf)
            for idx, (name, leaf) in enumerate(zip(names, leaves, strict=True))
        }
        return write_npz_compressed(sample_path, arrays)

    def copy_structure(self, target_dir: str | Path, *, artifact_name: str) -> Path:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        if not self.tree_path.exists():
            raise FileNotFoundError(f"Tree path not found: {self.tree_path}")
        dest = target / artifact_name
        if not dest.exists() or not self.tree_path.samefile(dest):
            shutil.copy2(self.tree_path, dest)
        return dest


__all__ = [
    "ParamTree",
    "PyTreeNpzCodec",
    "WeightSample",
    "load_pytree_definition",
    "tree_leaves_with_names",
]
