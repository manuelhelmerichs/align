"""Tests for runtime loaders, logging, resume behavior, and runners."""

import json
import logging
import pickle
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.config import (
    CanonicalizeConfig,
    MatchConfig,
    PathConfig,
    RunConfig,
    RuntimeConfig,
    SelectionConfig,
)
from align.run_state import RunState
from align.runtime import RunArtifactStore
from align.runtime.loaders import SampleLoader
from align.runtime.runner import AlignmentRunner
from align.runtime.stages import CanonicalizeExecutor, MatchExecutor, StageResult
from align.runtime.worker import _build_stage_executors
from align.sample_manifest import SampleManifest
from align.samples import WeightSample


def _build_sample_tree(base: Path) -> tuple[Path, Path]:
    exp_root = base / "experiment"
    samples_dir = exp_root / "samples" / "0"
    samples_dir.mkdir(parents=True, exist_ok=True)

    params_tree = {
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
    leaves, treedef = jax.tree_util.tree_flatten(params_tree)
    sample_path = samples_dir / "sample_0.npz"
    sample_payload = {f"arr_{idx}": np.asarray(leaf) for idx, leaf in enumerate(leaves)}
    np.savez(sample_path, **sample_payload)

    tree_path = exp_root / "tree"
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    with tree_path.open("wb") as handle:
        pickle.dump(treedef, handle)
    return exp_root, tree_path


def _build_sample_tree_with_multiple_samples(
    base: Path,
    *,
    num_samples: int,
) -> tuple[Path, Path]:
    exp_root, tree_path = _build_sample_tree(base)
    samples_dir = exp_root / "samples" / "0"
    source = np.load(samples_dir / "sample_0.npz")
    payload = {key: np.asarray(source[key]) for key in source.files}
    source.close()
    for sample_id in range(1, num_samples):
        np.savez(samples_dir / f"sample_{sample_id}.npz", **payload)
    return exp_root, tree_path


def test_alignment_runner_writes_stage_output_and_aligned_sample(
    tmp_path, monkeypatch
) -> None:
    exp_root, tree_path = _build_sample_tree(tmp_path)
    samples_dir = exp_root / "samples"
    output_dir = exp_root / "align"

    selection = SelectionConfig(reference_chain=0, reference_sample=0)
    config = RunConfig(
        paths=PathConfig(
            experiment_root=exp_root,
            samples_dir=samples_dir,
            tree_path=tree_path,
            output_dir=output_dir,
        ),
        selection=selection,
        pipeline=("canonicalize", "match"),
        canonicalize=CanonicalizeConfig(),
        # Single-pass: this test fakes the executors to check artifact writing.
        match=MatchConfig(barycenter_passes=1),
        runtime=RuntimeConfig(save_intermediate=True, parallelism=1),
    )
    manifest_path = output_dir / "state" / "sample_manifest.json"
    manifest = SampleManifest.build(
        samples_dir=samples_dir,
        tree_path=tree_path,
        chain_indices=selection.chain_indices,
        samples_per_chain=selection.samples_per_chain,
        sample_step=selection.sample_step,
        max_total=selection.max_total,
        reference_chain=selection.reference_chain,
        reference_sample=selection.reference_sample,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(manifest_path)

    state_dir = output_dir / "state"
    run_state = RunState(
        path=state_dir / "run_state.json",
        experiment_root=exp_root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        config_digest="digest",
        manifest_digest=manifest.digest(),
        stages=config.active_stages(),
        reference={
            "chain": manifest.reference_chain,
            "sample": manifest.reference_sample,
            "index": manifest.reference_index,
        },
        filters=manifest.filters,
        total_samples=manifest.total,
        runtime_settings={},
    )
    run_state.save()

    # Patch heavy operations to cheap identities.
    def _fake_canonicalize(self, record, sample):
        return StageResult(
            sample=sample,
            scales={"mlp/h0": jnp.ones((1,), dtype=jnp.float32)},
            diagnostics={"plan": "dense_chain"},
        )

    def _fake_match(self, record, sample):
        return StageResult(
            sample=sample,
            transforms={},
            diagnostics={"reference": True, "transform_families": {}},
        )

    monkeypatch.setattr(
        CanonicalizeExecutor, "prepare", lambda self, manifest, reference_params: None
    )
    monkeypatch.setattr(
        MatchExecutor, "prepare", lambda self, manifest, reference_params: None
    )
    monkeypatch.setattr(
        CanonicalizeExecutor, "reference_output", lambda self, record, params: params
    )
    monkeypatch.setattr(
        MatchExecutor, "reference_output", lambda self, record, params: params
    )
    monkeypatch.setattr(
        CanonicalizeExecutor, "process_single", _fake_canonicalize, raising=True
    )
    monkeypatch.setattr(MatchExecutor, "process_single", _fake_match, raising=True)

    artifact_store = RunArtifactStore(
        manifest=manifest,
        output_dir=output_dir,
        stages=config.active_stages(),
        save_intermediate=True,
    )
    runner = AlignmentRunner(
        config=config,
        manifest=manifest,
        run_state=run_state,
        artifact_store=artifact_store,
        progress_logger=logging.getLogger("test_align_runner"),
    )

    runner.execute()

    aligned_sample = output_dir / "aligned_samples" / "0" / "sample_0.npz"
    stage_output = output_dir / "stage_outputs" / "canonicalize" / "0" / "sample_0.npz"
    assert aligned_sample.exists()
    assert stage_output.exists()
    assert (output_dir / "scales").exists()
    assert (output_dir / "transforms").exists()
    assert run_state.completed


def _build_runtime_state(tmp_path):
    exp_root, tree_path = _build_sample_tree(tmp_path)
    samples_dir = exp_root / "samples"
    output_dir = exp_root / "align"

    selection = SelectionConfig(reference_chain=0, reference_sample=0)
    config = RunConfig(
        paths=PathConfig(
            experiment_root=exp_root,
            samples_dir=samples_dir,
            tree_path=tree_path,
            output_dir=output_dir,
        ),
        selection=selection,
        pipeline=("match",),
        canonicalize=None,
        match=MatchConfig(),
        runtime=RuntimeConfig(parallelism=2, validate_artifacts_on_resume=True),
    )

    manifest_path = output_dir / "state" / "sample_manifest.json"
    manifest = SampleManifest.build(
        samples_dir=samples_dir,
        tree_path=tree_path,
        chain_indices=selection.chain_indices,
        samples_per_chain=selection.samples_per_chain,
        sample_step=selection.sample_step,
        max_total=selection.max_total,
        reference_chain=selection.reference_chain,
        reference_sample=selection.reference_sample,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(manifest_path)

    run_state = RunState(
        path=output_dir / "state" / "run_state.json",
        experiment_root=exp_root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        config_digest="digest",
        manifest_digest=manifest.digest(),
        stages=config.active_stages(),
        reference={
            "chain": manifest.reference_chain,
            "sample": manifest.reference_sample,
            "index": manifest.reference_index,
        },
        filters=manifest.filters,
        total_samples=manifest.total,
        runtime_settings={},
        metadata={"match_objective": config.match.objective.type},
    )
    run_state.save()

    artifact_store = RunArtifactStore(
        manifest=manifest,
        output_dir=output_dir,
        stages=config.active_stages(),
        save_intermediate=False,
    )

    return config, manifest, run_state, artifact_store


def _build_parallel_runtime_state(tmp_path):
    exp_root, tree_path = _build_sample_tree_with_multiple_samples(
        tmp_path,
        num_samples=2,
    )
    samples_dir = exp_root / "samples"
    output_dir = exp_root / "align_parallel"

    selection = SelectionConfig(reference_chain=0, reference_sample=0)
    config = RunConfig(
        paths=PathConfig(
            experiment_root=exp_root,
            samples_dir=samples_dir,
            tree_path=tree_path,
            output_dir=output_dir,
        ),
        selection=selection,
        pipeline=("match",),
        canonicalize=None,
        # Single-pass: this test fakes the worker pool to check respawn handling.
        match=MatchConfig(barycenter_passes=1),
        runtime=RuntimeConfig(parallelism=2, validate_artifacts_on_resume=False),
    )

    manifest_path = output_dir / "state" / "sample_manifest.json"
    manifest = SampleManifest.build(
        samples_dir=samples_dir,
        tree_path=tree_path,
        chain_indices=selection.chain_indices,
        samples_per_chain=selection.samples_per_chain,
        sample_step=selection.sample_step,
        max_total=selection.max_total,
        reference_chain=selection.reference_chain,
        reference_sample=selection.reference_sample,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(manifest_path)

    run_state = RunState(
        path=output_dir / "state" / "run_state.json",
        experiment_root=exp_root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        config_digest="digest",
        manifest_digest=manifest.digest(),
        stages=config.active_stages(),
        reference={
            "chain": manifest.reference_chain,
            "sample": manifest.reference_sample,
            "index": manifest.reference_index,
        },
        filters=manifest.filters,
        total_samples=manifest.total,
        runtime_settings={},
        metadata={"match_objective": config.match.objective.type},
    )
    run_state.save()

    artifact_store = RunArtifactStore(
        manifest=manifest,
        output_dir=output_dir,
        stages=config.active_stages(),
        save_intermediate=False,
    )
    return config, manifest, run_state, artifact_store


def test_resume_with_no_pending_samples_repairs_finalization(tmp_path) -> None:
    config, manifest, run_state, artifact_store = _build_parallel_runtime_state(
        tmp_path
    )
    for record in manifest.records:
        run_state.record_progress(record.index)
    run_state.save()

    runner = AlignmentRunner(
        config=config,
        manifest=manifest,
        run_state=run_state,
        artifact_store=artifact_store,
        progress_logger=logging.getLogger("test_resume_finalize"),
    )
    runner.execute()

    assert run_state.completed
    assert (config.paths.output_dir / "summary.json").exists()
    diagnostics = json.loads(
        (config.paths.output_dir / "sample_diagnostics.json").read_text()
    )
    assert diagnostics == [{}, {}]


def test_transient_worker_failure_respects_retry_budget(tmp_path) -> None:
    config, manifest, run_state, artifact_store = _build_parallel_runtime_state(
        tmp_path
    )
    config.runtime.max_worker_retries = 0
    runner = AlignmentRunner(
        config=config,
        manifest=manifest,
        run_state=run_state,
        artifact_store=artifact_store,
    )
    runner._worker_retry_counts = {}
    state = SimpleNamespace(
        stopping=False,
        worker_id=0,
        assigned=[manifest.records[0].index],
    )
    with pytest.raises(RuntimeError, match="retry budget exhausted"):
        runner._handle_worker_failure(
            SimpleNamespace(states={}),
            state,
            deque(),
            {},
            tmp_path,
            1,
            "synthetic process death",
            transient=True,
        )


def test_matching_solver_diagnostics_survive_aggregation(tmp_path) -> None:
    _, manifest, _, artifact_store = _build_parallel_runtime_state(tmp_path)
    record = manifest.records[0]
    artifact_store.write_diagnostics(
        record,
        {
            "match": {
                "objective": "euclidean",
                "solvers": [{"solver": "sinkhorn"}],
                "objective_final": 1.25,
                "steps": [
                    {
                        "solver": "sinkhorn",
                        "steps": 3,
                        "converged": True,
                        "ambiguity": {"mlp/h0": {"assignment_margin_min": 0.4}},
                    }
                ],
            }
        },
    )
    artifact_store.finalize(elapsed_seconds=0.0)

    aggregated = json.loads(
        (artifact_store.output_dir / "sample_diagnostics.json").read_text()
    )
    match = aggregated[record.index]["match"]
    assert match["objective_final"] == 1.25
    assert match["steps"][0]["converged"] is True
    assert match["steps"][0]["ambiguity"]["mlp/h0"]["assignment_margin_min"] == 0.4


def test_worker_rebuilds_matching_against_refined_reference(tmp_path) -> None:
    config, manifest, _, _ = _build_runtime_state(tmp_path)
    initial = SampleLoader(manifest).load_reference()
    refined = WeightSample(
        jax.tree_util.tree_map(lambda leaf: leaf + 0.25, initial.params)
    )
    job = {
        "stages": ["match"],
        "match_config": config.match,
        "match_reference_index": None,
        "rng_offset": manifest.total,
        "family": config.architecture.family,
        "recipe_kwargs": config.architecture.recipe_kwargs,
        "per_device_batch": 1,
    }

    stages = _build_stage_executors(
        job,
        manifest,
        initial,
        match_reference=refined,
    )

    assert len(stages) == 1
    executor = stages[0][1]
    assert executor.reference_index is None
    assert executor.rng_offset == manifest.total
    assert executor.reference_sample is refined


def test_runner_apply_commit_moves_scratch_and_updates_checksums(tmp_path) -> None:
    config, manifest, run_state, artifact_store = _build_runtime_state(tmp_path)
    runner = AlignmentRunner(
        config=config,
        manifest=manifest,
        run_state=run_state,
        artifact_store=artifact_store,
        progress_logger=logging.getLogger("test_apply_commit"),
    )

    record = manifest.records[0]
    scratch_dir = run_state.state_dir / "scratch" / "worker_0"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_file = scratch_dir / "aligned_sample.npz"
    scratch_file.write_bytes(b"payload")

    runner._apply_commit(
        record,
        artifacts={
            "aligned_sample_path": str(scratch_file),
            "scratch_dir": str(scratch_dir),
        },
        checksums={"aligned_sample": 123},
        diagnostics={},
    )

    dest_path = artifact_store.artifact_paths(record)["aligned_sample"]
    assert dest_path.exists()
    assert not scratch_dir.exists()

    entry = artifact_store._checksums.entry(record.index)
    assert entry["aligned_sample"] == 123
    assert entry.get("transforms", 0) == 0


def test_runner_requeues_on_checksum_mismatch(tmp_path) -> None:
    config, manifest, run_state, artifact_store = _build_runtime_state(tmp_path)
    loader = SampleLoader(manifest)
    record = manifest.records[0]

    sample = loader.load(record)
    artifact_store.write_aligned_sample(record, sample)
    artifact_store.maybe_flush(force=True)
    run_state.record_progress(record.index)

    aligned_path = artifact_store.artifact_paths(record)["aligned_sample"]
    aligned_path.write_bytes(b"corrupt")

    runner = AlignmentRunner(
        config=config,
        manifest=manifest,
        run_state=run_state,
        artifact_store=artifact_store,
        progress_logger=logging.getLogger("test_checksum_resume"),
    )

    pending = runner._pending_records()
    assert record in pending
    assert not run_state.is_processed(record.index)


def test_align_runner_parallel_does_not_retry_deterministic_worker_error(
    tmp_path,
    monkeypatch,
) -> None:
    config, manifest, run_state, artifact_store = _build_parallel_runtime_state(
        tmp_path
    )
    record_lookup = {record.index: record for record in manifest.records}
    fail_once_index = manifest.records[0].index

    class _FakeProcess:
        def __init__(self):
            self._alive = True
            self.exitcode = None

        def is_alive(self):
            return self._alive

        def terminate(self):
            self._alive = False
            self.exitcode = 1

        def join(self, timeout=None):
            del timeout
            return None

        def kill(self):
            self._alive = False
            self.exitcode = -9

    class _FakeQueue:
        def __init__(self, pool, worker_id):
            self._pool = pool
            self._worker_id = worker_id

        def put(self, message):
            record_indices = list(message["record_indices"])
            state = self._pool.states[self._worker_id]
            for sample_index in record_indices:
                attempt = self._pool.attempts.get(sample_index, 0) + 1
                self._pool.attempts[sample_index] = attempt
                if sample_index == fail_once_index and attempt == 1:
                    self._pool.messages.append(
                        {
                            "type": "error",
                            "worker_id": self._worker_id,
                            "generation": state.generation,
                            "error": "synthetic worker failure",
                        }
                    )
                    continue

                record = record_lookup[sample_index]
                scratch_dir = state.scratch_dir / f"sample_{sample_index}"
                scratch_dir.mkdir(parents=True, exist_ok=True)
                aligned_path = scratch_dir / "aligned_sample.npz"
                aligned_path.write_bytes(
                    record.absolute_path(manifest.samples_dir).read_bytes()
                )
                self._pool.messages.append(
                    {
                        "type": "commit",
                        "worker_id": self._worker_id,
                        "generation": state.generation,
                        "sample_index": sample_index,
                        "artifacts": {
                            "aligned_sample_path": str(aligned_path),
                            "scratch_dir": str(scratch_dir),
                        },
                        "checksums": None,
                        "diagnostics": {"worker": {"attempt": attempt}},
                    }
                )

        def close(self):
            return None

    class _FakeWorkerPool:
        respawn_count = 0

        def __init__(
            self,
            parallelism,
            device_ids,
            strategy_prefers_gpu,
            allow_device_sharing=False,
        ):
            del parallelism, device_ids, strategy_prefers_gpu, allow_device_sharing
            self.states = {}
            self.messages = []
            self.attempts = {}

        def _make_state(self, worker_id, scratch_root, *, generation, ready):
            return SimpleNamespace(
                worker_id=worker_id,
                generation=generation,
                ready=ready,
                stopping=False,
                assigned=[],
                last_heartbeat=time.time(),
                command_queue=_FakeQueue(self, worker_id),
                process=_FakeProcess(),
                scratch_dir=scratch_root / f"worker_{worker_id}_g{generation}",
            )

        def start(self, worker_count, job_template, scratch_root):
            del job_template
            for worker_id in range(worker_count):
                state = self._make_state(
                    worker_id,
                    scratch_root,
                    generation=0,
                    ready=False,
                )
                self.states[worker_id] = state
                self.messages.append(
                    {
                        "type": "ready",
                        "worker_id": worker_id,
                        "generation": state.generation,
                    }
                )

        def next_message(self, timeout=1.0):
            del timeout
            if self.messages:
                return self.messages.pop(0)
            return None

        def respawn(self, worker_id, job_template, scratch_root):
            # Mirror the real WorkerPool contract: _spawn_worker registers the
            # fresh state itself (ready=False) and the new worker process then
            # announces itself with a "ready" message.
            del job_template
            type(self).respawn_count += 1
            state = self._make_state(
                worker_id,
                scratch_root,
                generation=self.states[worker_id].generation + 1,
                ready=False,
            )
            self.states[worker_id] = state
            self.messages.append(
                {
                    "type": "ready",
                    "worker_id": worker_id,
                    "generation": state.generation,
                }
            )
            return state

        def shutdown(self, force=False):
            del force
            return None

    monkeypatch.setattr(
        "align.runtime.runner.SampleLoader.load_reference",
        lambda self: WeightSample(
            {"params": {"dummy": jnp.array([0.0], dtype=jnp.float32)}}
        ),
    )
    monkeypatch.setattr(
        AlignmentRunner,
        "_prepare_executors",
        lambda self, reference_params, **kwargs: [],
    )
    monkeypatch.setattr("align.runtime.runner.WorkerPool", _FakeWorkerPool)

    runner = AlignmentRunner(
        config=config,
        manifest=manifest,
        run_state=run_state,
        artifact_store=artifact_store,
        progress_logger=logging.getLogger("test_parallel_respawn"),
    )

    with pytest.raises(RuntimeError, match="will not be retried"):
        runner.execute()

    assert _FakeWorkerPool.respawn_count == 0
    assert not run_state.completed
