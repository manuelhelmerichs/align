"""Tests for sample codecs, tree metadata, and sample loading."""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.runtime.loaders import SampleLoader
from align.sample_manifest import SampleManifest
from align.samples import PyTreeNpzCodec, WeightSample


def _params_tree():
    return {
        "params": {
            "dense0": {
                "kernel": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
                "bias": jnp.array([0.5, -0.25, 1.0], dtype=jnp.float32),
            },
            "dense1": {
                "kernel": jnp.arange(3, dtype=jnp.float32).reshape(3, 1),
                "bias": jnp.array([0.125], dtype=jnp.float32),
            },
        }
    }


def _write_tree(path: Path, params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(jax.tree_util.tree_structure(params), handle)


def _assert_same_tree(left, right) -> None:
    left_leaves, left_treedef = jax.tree_util.tree_flatten(left)
    right_leaves, right_treedef = jax.tree_util.tree_flatten(right)
    assert left_treedef == right_treedef
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(left_leaf), np.asarray(right_leaf))


def test_pytree_npz_codec_loads_leaf_ordered_sample(tmp_path) -> None:
    params = _params_tree()
    leaves, _ = jax.tree_util.tree_flatten(params)
    tree_path = tmp_path / "tree"
    sample_path = tmp_path / "sample.npz"
    _write_tree(tree_path, params)

    np.savez(
        sample_path,
        third=np.asarray(leaves[0]),
        first=np.asarray(leaves[1]),
        second=np.asarray(leaves[2]),
        fourth=np.asarray(leaves[3]),
    )

    codec = PyTreeNpzCodec(tree_path)
    sample = codec.load(sample_path)

    assert isinstance(sample, WeightSample)
    _assert_same_tree(sample.params, params)


def test_pytree_npz_codec_save_round_trips(tmp_path) -> None:
    params = _params_tree()
    tree_path = tmp_path / "tree"
    sample_path = tmp_path / "out" / "sample.npz"
    _write_tree(tree_path, params)

    codec = PyTreeNpzCodec(tree_path)
    codec.save(sample_path, WeightSample(params))
    loaded = codec.load(sample_path)

    _assert_same_tree(loaded.params, params)


def test_pytree_npz_codec_rejects_leaf_count_mismatch(tmp_path) -> None:
    params = _params_tree()
    tree_path = tmp_path / "tree"
    sample_path = tmp_path / "sample.npz"
    _write_tree(tree_path, params)
    np.savez(sample_path, only=np.array([1.0], dtype=np.float32))

    codec = PyTreeNpzCodec(tree_path)
    with pytest.raises(ValueError, match="expects 4"):
        codec.load(sample_path)


def test_sample_loader_returns_weight_samples(tmp_path) -> None:
    exp_root = tmp_path / "experiment"
    samples_dir = exp_root / "samples" / "0"
    samples_dir.mkdir(parents=True)
    params = _params_tree()
    tree_path = exp_root / "tree"
    _write_tree(tree_path, params)

    codec = PyTreeNpzCodec(tree_path)
    codec.save(samples_dir / "sample_0.npz", WeightSample(params))

    manifest = SampleManifest.build(
        exp_root / "samples",
        tree_path,
        reference_chain=0,
        reference_sample=0,
    )
    manifest_path = exp_root / "align" / "state" / "sample_manifest.json"
    manifest.save(manifest_path)
    manifest.ensure_shared_tree(manifest_path)

    loaded = SampleLoader(manifest).load_reference()

    assert isinstance(loaded, WeightSample)
    _assert_same_tree(loaded.params, params)
