"""Tests for run-state creation, resume gating, and bitmap progress."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from align.cli import _load_or_build_manifest, _load_or_create_run_state
from align.config import (
    CanonicalizeConfig,
    PathConfig,
    RunConfig,
    SelectionConfig,
)
from align.run_state import ArtifactChecksumStore, RunState
from align.sample_manifest import SampleManifest


def _write_dummy_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weight=np.array([1.0], dtype=np.float32))


def _build_manifest_workspace(tmp_path: Path) -> tuple[SampleManifest, Path, Path]:
    samples_dir = tmp_path / "samples"
    tree_path = tmp_path / "tree.pkl"
    tree_path.write_bytes(b"tree")
    chain_dir = samples_dir / "0"
    _write_dummy_sample(chain_dir / "sample_0.npz")
    manifest = SampleManifest.build(
        samples_dir,
        tree_path,
        chain_indices=None,
        samples_per_chain=1,
        reference_chain=0,
        reference_sample=0,
    )
    return manifest, samples_dir, tree_path


def test_run_state_progress_and_unprocess(tmp_path: Path) -> None:
    manifest, _, _ = _build_manifest_workspace(tmp_path)
    state_dir = tmp_path / "output" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = state_dir / "sample_manifest.json"
    manifest.save(manifest_path)

    run_state_path = state_dir / "run_state.json"
    run_state = RunState(
        path=run_state_path,
        experiment_root=tmp_path,
        output_dir=state_dir.parent,
        manifest_path=manifest_path,
        config_digest="digest",
        manifest_digest=manifest.digest(),
        stages=["canonicalize"],
        reference={"chain": 0, "sample": 0, "index": 0},
        filters={},
        total_samples=manifest.total,
    )

    run_state.record_progress(sample_index=0)
    assert run_state.processed_count() == 1

    run_state.mark_unprocessed(0)
    assert run_state.processed_count() == 0

    run_state.save()
    reloaded = RunState.load(run_state_path)
    assert reloaded.processed_count() == 0


def test_artifact_checksum_store_rejects_layout_mismatch(tmp_path: Path) -> None:
    checksum_path = tmp_path / "artifact_checksums.npy"
    checksum_path.write_bytes(np.zeros((1, 1), dtype=np.uint64).tobytes())

    with pytest.raises(ValueError, match="Checksum store .* expected"):
        ArtifactChecksumStore(
            checksum_path,
            total_samples=1,
            kinds=("aligned_sample", "transforms"),
        )


def test_load_or_build_manifest_respects_resume_flag(tmp_path: Path) -> None:
    selection = SelectionConfig(reference_chain=0, reference_sample=0)
    manifest_path = tmp_path / "state" / "sample_manifest.json"
    manifest, samples_dir, tree_path = _build_manifest_workspace(tmp_path)
    built = _load_or_build_manifest(
        manifest_path,
        samples_dir,
        tree_path,
        selection=selection,
        sample_format="pytree_npz",
        resume=False,
    )
    assert built.digest() == manifest.digest()

    shutil.rmtree(samples_dir)
    loaded = _load_or_build_manifest(
        manifest_path,
        samples_dir,
        tree_path,
        selection=selection,
        sample_format="pytree_npz",
        resume=True,
    )
    assert loaded.digest() == built.digest()


def test_load_or_create_run_state_handles_digest_and_resume(
    tmp_path: Path, caplog
) -> None:
    manifest, _, _ = _build_manifest_workspace(tmp_path)
    state_dir = tmp_path / "output" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = state_dir / "sample_manifest.json"
    manifest.save(manifest_path)

    config = RunConfig(
        paths=PathConfig(experiment_root=tmp_path),
        selection=SelectionConfig(reference_chain=0, reference_sample=0),
        pipeline=("canonicalize",),
        canonicalize=CanonicalizeConfig(),
        match=None,
    )
    run_state_path = state_dir / "run_state.json"
    runtime_settings = {"parallelism": 2}

    run_state = _load_or_create_run_state(
        run_state_path,
        run_exists=False,
        manifest=manifest,
        output_dir=state_dir.parent,
        exp_root=tmp_path,
        digest="digest",
        config=config,
        manifest_path=manifest_path,
        runtime_settings=runtime_settings,
        ignore_digest=False,
    )
    assert run_state_path.exists()
    assert run_state.config_digest == "digest"

    loaded = _load_or_create_run_state(
        run_state_path,
        run_exists=True,
        manifest=manifest,
        output_dir=state_dir.parent,
        exp_root=tmp_path,
        digest="digest",
        config=config,
        manifest_path=manifest_path,
        runtime_settings=runtime_settings,
        ignore_digest=False,
    )
    assert loaded.config_digest == "digest"

    with pytest.raises(ValueError):
        _load_or_create_run_state(
            run_state_path,
            run_exists=True,
            manifest=manifest,
            output_dir=state_dir.parent,
            exp_root=tmp_path,
            digest="other",
            config=config,
            manifest_path=manifest_path,
            runtime_settings=runtime_settings,
            ignore_digest=False,
        )

    with caplog.at_level("WARNING", logger="align"):
        resumed = _load_or_create_run_state(
            run_state_path,
            run_exists=True,
            manifest=manifest,
            output_dir=state_dir.parent,
            exp_root=tmp_path,
            digest="other",
            config=config,
            manifest_path=manifest_path,
            runtime_settings=runtime_settings,
            ignore_digest=True,
        )
    assert resumed.config_digest == "digest"
    assert "Config digest mismatch ignored" in caplog.text

    missing_state = state_dir / "missing.json"
    with pytest.raises(FileNotFoundError):
        _load_or_create_run_state(
            missing_state,
            run_exists=True,
            manifest=manifest,
            output_dir=state_dir.parent,
            exp_root=tmp_path,
            digest="digest",
            config=config,
            manifest_path=manifest_path,
            runtime_settings=runtime_settings,
            ignore_digest=False,
        )
