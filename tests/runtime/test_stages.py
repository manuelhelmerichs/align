"""Tests for canonicalize and match stage executors."""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from align.config import SolverStep
from align.config.stages import (
    CanonicalizeConfig,
    CenterSoftmaxHeadConfig,
    MatchConfig,
)
from align.runtime.loaders import SampleLoader
from align.runtime.stages import (
    CanonicalizeExecutor,
    CenterSoftmaxHeadExecutor,
    MatchExecutor,
    StageResult,
)
from align.sample_manifest import SampleManifest


def _build_manifest(tmp_path: Path, sample_count: int = 2) -> SampleManifest:
    exp_root = tmp_path / "experiment"
    samples_root = exp_root / "samples"
    chain_dir = samples_root / "0"
    chain_dir.mkdir(parents=True, exist_ok=True)

    base_tree = {
        "params": {
            "fcn": {
                "dense0": {
                    "kernel": jnp.ones((1, 1), dtype=jnp.float32),
                    "bias": jnp.zeros((1,), dtype=jnp.float32),
                },
                "dense1": {
                    "kernel": jnp.ones((1, 1), dtype=jnp.float32),
                    "bias": jnp.zeros((1,), dtype=jnp.float32),
                },
            }
        }
    }
    leaves, treedef = jax.tree_util.tree_flatten(base_tree)
    tree_path = exp_root / "tree"
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    with tree_path.open("wb") as handle:
        pickle.dump(treedef, handle)

    for idx in range(sample_count):
        payload = {
            f"arr_{leaf_idx}": np.asarray(leaf + idx)
            for leaf_idx, leaf in enumerate(leaves)
        }
        sample_path = chain_dir / f"sample_{idx}.npz"
        np.savez(sample_path, **payload)

    manifest = SampleManifest.build(
        samples_dir=samples_root,
        tree_path=tree_path,
        chain_indices=None,
        samples_per_chain=None,
        sample_step=1,
        max_total=None,
        reference_chain=0,
        reference_sample=0,
    )
    manifest_path = exp_root / "align" / "state" / "sample_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(manifest_path)
    return manifest


def _build_residual_convnet_manifest(
    tmp_path: Path, sample_count: int = 2
) -> SampleManifest:
    exp_root = tmp_path / "experiment_residual_convnet"
    samples_root = exp_root / "samples"
    chain_dir = samples_root / "0"
    chain_dir.mkdir(parents=True, exist_ok=True)

    base_tree = {
        "core": {
            "Conv_0": {
                "kernel": jnp.ones((1, 1, 1, 2), dtype=jnp.float32),
                "bias": jnp.zeros((2,), dtype=jnp.float32),
            },
            "Conv_1": {
                "kernel": jnp.ones((1, 1, 2, 2), dtype=jnp.float32),
                "bias": jnp.zeros((2,), dtype=jnp.float32),
            },
            "Dense_0": {
                "kernel": jnp.ones((2, 1), dtype=jnp.float32),
                "bias": jnp.zeros((1,), dtype=jnp.float32),
            },
        }
    }
    leaves, treedef = jax.tree_util.tree_flatten(base_tree)
    tree_path = exp_root / "tree"
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    with tree_path.open("wb") as handle:
        pickle.dump(treedef, handle)

    for idx in range(sample_count):
        payload = {
            f"arr_{leaf_idx}": np.asarray(leaf + idx)
            for leaf_idx, leaf in enumerate(leaves)
        }
        sample_path = chain_dir / f"sample_{idx}.npz"
        np.savez(sample_path, **payload)

    manifest = SampleManifest.build(
        samples_dir=samples_root,
        tree_path=tree_path,
        chain_indices=None,
        samples_per_chain=None,
        sample_step=1,
        max_total=None,
        reference_chain=0,
        reference_sample=0,
    )
    manifest_path = exp_root / "align" / "state" / "sample_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(manifest_path)
    return manifest


def test_center_softmax_head_executor_detects_and_centers(tmp_path) -> None:
    manifest = _build_residual_convnet_manifest(tmp_path, sample_count=1)
    loader = SampleLoader(manifest)
    reference_sample = loader.load_reference()

    executor = CenterSoftmaxHeadExecutor(
        CenterSoftmaxHeadConfig(),
        family="residual_convnet",
        recipe_kwargs={},
    )
    executor.prepare(manifest, reference_sample)
    assert executor.head_path == ("core", "Dense_0")

    result = executor.process_single(manifest.reference_record, reference_sample)
    assert isinstance(result, StageResult)
    assert result.diagnostics
    assert result.diagnostics["plan"] == "softmax_head_translation"
    assert result.diagnostics["head"] == "core.Dense_0"
    kernel = np.asarray(result.sample.params["core"]["Dense_0"]["kernel"])
    np.testing.assert_allclose(kernel.mean(axis=1), 0.0, atol=1e-6)

    explicit = CenterSoftmaxHeadExecutor(
        CenterSoftmaxHeadConfig(head="core.Dense_0"),
        family="residual_convnet",
        recipe_kwargs={},
    )
    explicit.prepare(manifest, reference_sample)
    assert explicit.head_path == ("core", "Dense_0")


def test_canonicalize_executor_process_single(tmp_path) -> None:
    manifest = _build_manifest(tmp_path, sample_count=1)
    loader = SampleLoader(manifest)
    reference_sample = loader.load_reference()

    executor = CanonicalizeExecutor(
        CanonicalizeConfig(),
        family="mlp",
        recipe_kwargs={},
    )
    executor.prepare(manifest, reference_sample)

    result = executor.process_single(manifest.reference_record, reference_sample)
    assert isinstance(result, StageResult)
    assert result.scales
    assert result.diagnostics and result.diagnostics["plan"] == "dense_chain"
    assert jax.tree_util.tree_structure(
        result.sample.params
    ) == jax.tree_util.tree_structure(reference_sample.params)


def test_match_executor_single_and_batch(tmp_path) -> None:
    manifest = _build_manifest(tmp_path, sample_count=2)
    loader = SampleLoader(manifest)
    reference_sample = loader.load_reference()

    config = MatchConfig()
    executor = MatchExecutor(
        config,
        reference_index=manifest.reference_index,
        seed=0,
        batch_size=2,
        family="mlp",
        recipe_kwargs={},
    )
    executor.prepare(manifest, reference_sample)

    target_record = manifest.records[1]
    target_sample = loader.load(target_record)

    single = executor.process_single(target_record, target_sample)
    assert isinstance(single, StageResult)
    assert single.transforms
    assert single.sample.params is not None

    records = [manifest.reference_record, target_record]
    samples = [loader.load(rec) for rec in records]
    batch_results = executor.process_batch(records, samples)
    assert len(batch_results) == len(records)
    # Reference sample should carry the reference flag
    assert any(
        res.diagnostics and res.diagnostics.get("reference") for res in batch_results
    )


def test_match_executor_preserves_solver_steps(tmp_path) -> None:
    manifest = _build_manifest(tmp_path, sample_count=1)
    loader = SampleLoader(manifest)
    reference_sample = loader.load_reference()

    step = SolverStep(
        solver="lap",
        max_sweeps=3,
        max_steps=17,
        record_loss_history=True,
    )
    executor = MatchExecutor(
        MatchConfig(solvers=(step,)),
        reference_index=manifest.reference_index,
        seed=0,
        batch_size=1,
        family="mlp",
        recipe_kwargs={},
    )
    executor.prepare(manifest, reference_sample)

    assert executor.solver_sequence is not None
    assert executor.solver_sequence.steps == (step,)


def test_residual_convnet_stages_round_trip(tmp_path) -> None:
    manifest = _build_residual_convnet_manifest(tmp_path, sample_count=2)
    loader = SampleLoader(manifest)
    reference_sample = loader.load_reference()

    canonicalize_executor = CanonicalizeExecutor(
        CanonicalizeConfig(),
        family="residual_convnet",
        recipe_kwargs={},
    )
    canonicalize_executor.prepare(manifest, reference_sample)
    canonicalize_result = canonicalize_executor.process_single(
        manifest.reference_record, reference_sample
    )
    assert isinstance(canonicalize_result, StageResult)
    assert canonicalize_result.diagnostics is not None
    assert canonicalize_result.diagnostics["plan"] == "conv_balanced"
    assert canonicalize_result.scales

    match_executor = MatchExecutor(
        MatchConfig(),
        reference_index=manifest.reference_index,
        seed=0,
        batch_size=2,
        family="residual_convnet",
        recipe_kwargs={},
    )
    match_executor.prepare(manifest, reference_sample)

    target_record = manifest.records[1]
    target_sample = loader.load(target_record)
    single = match_executor.process_single(target_record, target_sample)
    assert isinstance(single, StageResult)
    assert single.transforms
