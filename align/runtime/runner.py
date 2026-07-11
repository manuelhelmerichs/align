"""AlignmentRunner with modular stage executors and parallel support."""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .._jax_platforms import is_gpu_platform
from ..config import RunConfig, resolve_recipe_defaults
from ..logging_utils import progress_bar
from ..run_state import RunState
from ..sample_manifest import SampleManifest, SampleRecord
from ..samples import WeightSample
from .artifacts import RunArtifactStore
from .diagnostics import reference_stability_diagnostic, tree_mean
from .loaders import SampleLoader
from .parallel import WorkerPool
from .pipeline import SampleAlignmentResult, StagePipeline
from .resources import (
    LOG_SAMPLE_INTERVAL,
    LOG_TIME_INTERVAL_SECONDS,
    STATE_SAVE_INTERVAL_SECONDS,
    WORKER_HEARTBEAT_INTERVAL,
    WORKER_HEARTBEAT_TIMEOUT,
    available_cpu_count,
)
from .stages import (
    CanonicalizeExecutor,
    CenterSoftmaxHeadExecutor,
    MatchExecutor,
    StageExecutor,
    prepare_stage_executors,
)

_LOG = logging.getLogger(__name__)


class _RunnerPipelineSink:
    """Write pipeline outputs through ``RunArtifactStore`` and update run progress."""

    def __init__(
        self,
        runner: AlignmentRunner,
        bar,
        *,
        output_artifact_store: RunArtifactStore,
        final_pass: bool,
    ) -> None:
        self.runner = runner
        self.bar = bar
        self.output_artifact_store = output_artifact_store
        self.final_pass = bool(final_pass)

    def write(self, output: SampleAlignmentResult) -> None:
        record = output.record
        artifact_store = self.output_artifact_store
        if output.scales is not None:
            artifact_store.write_scales(record, output.scales)
        if output.transforms is not None and self.runner.config.match:
            artifact_store.write_transforms(
                record,
                output.transforms,
                transform_families=output.transform_families,
            )
        if self.final_pass and self.runner.save_intermediate:
            for stage, sample in output.stage_outputs.items():
                artifact_store.write_stage_output(record, stage, sample)

        artifact_store.write_aligned_sample(record, output.sample)
        if self.final_pass:
            artifact_store.write_diagnostics(record, output.diagnostics)
            self.runner.run_state.record_progress(record=record)
            self.runner._maybe_persist_state()
        self.bar.update(label=record.label)
        if self.final_pass:
            self.runner._maybe_log_progress(record.label)


class AlignmentRunner:
    """Runner that composes stage executors and optional parallelism."""

    def __init__(
        self,
        config: RunConfig,
        manifest: SampleManifest,
        run_state: RunState,
        artifact_store: RunArtifactStore,
        *,
        progress_logger: logging.Logger | None = None,
    ) -> None:
        self.manifest = manifest
        self.run_state = run_state
        self.artifact_store = artifact_store
        self.progress_logger = progress_logger or _LOG
        self.config = config
        resolve_recipe_defaults(self.config)
        self.save_intermediate = bool(config.runtime.save_intermediate)
        self.per_device_batch = max(1, int(config.runtime.per_device_batch or 1))
        self.stage_order = config.active_stages()
        self.parallelism = self._compute_parallelism()
        self._stage_executors: list[tuple[str, StageExecutor]] = []
        now = time.time()
        processed_so_far = self.run_state.processed_count()
        self._last_logged_count = processed_so_far
        self._last_log_time = now
        self._last_save_time = now
        self._validate_artifacts_on_resume = bool(
            config.runtime.validate_artifacts_on_resume
        )
        self._refinement_pass_index = 0
        self._match_reference_path: Path | None = None
        self._match_reference_index: int | None = self.manifest.reference_index

    def _dry_run_summary_extra(self) -> dict[str, Any]:
        return {
            "stages": self.stage_order,
            "barycenter_passes": self._barycenter_passes(),
        }

    def _barycenter_passes(self) -> int:
        config = self.config.match
        if config is None:
            return 1
        return int(config.barycenter_passes)

    def _compute_parallelism(self) -> int:
        requested = self.config.runtime.parallelism
        if requested is not None:
            return max(1, int(requested))

        match_cfg = self.config.match
        if match_cfg and any(step.solver == "sinkhorn" for step in match_cfg.solvers):
            gpu_count = len(self._visible_gpu_ids(self.config.runtime.device_ids))
            if gpu_count > 0:
                return gpu_count
        return available_cpu_count()

    def execute(self, *, dry_run: bool = False) -> None:
        pending = self._pending_records()
        processed = self.run_state.processed_count()
        barycenter_passes = self._barycenter_passes()
        self.progress_logger.info(
            "Aligning %d/%d samples (resume=%d processed) | stages=%s | passes=%d",
            len(pending),
            self.manifest.total,
            self.manifest.total - len(pending),
            ",".join(self.stage_order),
            barycenter_passes,
        )
        if dry_run:
            self._write_dry_run_summary(pending)
            return

        if not pending:
            self.progress_logger.info("All samples already processed. Nothing to do.")
            return

        loader = SampleLoader(self.manifest)
        configured_reference = loader.load_reference()
        match_reference: WeightSample | None = None
        previous_artifact_store: RunArtifactStore | None = None
        stability_history: list[dict[str, Any]] = []
        refinement_root = self.run_state.state_dir / "refinement"
        if barycenter_passes > 1:
            shutil.rmtree(refinement_root, ignore_errors=True)
            refinement_root.mkdir(parents=True, exist_ok=True)

        elapsed = 0.0
        for pass_index in range(barycenter_passes):
            final_pass = pass_index == barycenter_passes - 1
            pass_number = pass_index + 1
            pass_records = pending if final_pass else list(self.manifest.records)
            output_artifact_store = (
                self.artifact_store
                if final_pass
                else RunArtifactStore(
                    manifest=self.manifest,
                    output_dir=refinement_root / f"pass_{pass_number}",
                    stages=[],
                    save_intermediate=False,
                )
            )

            self._refinement_pass_index = pass_index
            self._match_reference_index = (
                self.manifest.reference_index if pass_index == 0 else None
            )
            self._match_reference_path = None
            if match_reference is not None:
                reference_path = refinement_root / f"reference_{pass_number}.npz"
                self.artifact_store.sample_codec.save(reference_path, match_reference)
                self._match_reference_path = reference_path

            self._stage_executors = self._prepare_executors(
                configured_reference,
                match_reference=match_reference,
                pass_index=pass_index,
            )
            self.progress_logger.info(
                "Starting alignment pass %d/%d%s",
                pass_number,
                barycenter_passes,
                " (final artifacts)" if final_pass else "",
            )
            use_parallel = self.parallelism > 1 and len(pass_records) > 1
            if use_parallel:
                elapsed += self._run_parallel(
                    pass_records,
                    processed if final_pass else 0,
                    output_artifact_store=output_artifact_store,
                    final_pass=final_pass,
                )
            else:
                elapsed += self._run_local(
                    pass_records,
                    loader,
                    processed if final_pass else 0,
                    output_artifact_store=output_artifact_store,
                    final_pass=final_pass,
                )

            if previous_artifact_store is not None:
                diagnostic = self._reference_stability(
                    previous_artifact_store,
                    output_artifact_store,
                    previous_pass=pass_number - 1,
                    current_pass=pass_number,
                )
                stability_history.append(diagnostic)
                self.progress_logger.info(
                    "Reference stability after pass %d: convergence ratio=%s",
                    pass_number,
                    (
                        f"{diagnostic['convergence_ratio']:.6g}"
                        if diagnostic["convergence_ratio"] is not None
                        else "undefined (zero cloud spread)"
                    ),
                )

            if not final_pass:
                match_reference = self._mean_output_sample(output_artifact_store)
                previous_artifact_store = output_artifact_store

        total_elapsed = self.run_state.add_elapsed(elapsed)
        diagnostics = None
        if stability_history:
            latest = dict(stability_history[-1])
            latest["history"] = stability_history
            diagnostics = {"reference_stability": latest}
        self.artifact_store.finalize(
            elapsed_seconds=total_elapsed,
            diagnostics=diagnostics,
        )
        self.run_state.mark_complete()
        self.run_state.save()
        if barycenter_passes > 1:
            shutil.rmtree(refinement_root, ignore_errors=True)
        self.progress_logger.info("Align complete in %.2fs", elapsed)

    def _pending_records(self) -> list[SampleRecord]:
        pending: list[SampleRecord] = []
        for record in self.manifest.records:
            if not self.run_state.is_processed(record.index):
                pending.append(record)
                continue
            if (
                self._validate_artifacts_on_resume
                and not self.artifact_store.validate_artifacts(record)
            ):
                self.run_state.mark_unprocessed(record.index)
                pending.append(record)
                self.progress_logger.warning(
                    "Artifact checksum mismatch for %s. Sample will be reprocessed.",
                    record.label,
                )
        self.artifact_store.maybe_flush(force=True)
        return pending

    def _output_samples(self, artifact_store: RunArtifactStore):
        for record in self.manifest.records:
            path = artifact_store.artifact_paths(record)["aligned_sample"]
            yield artifact_store.sample_codec.load(path)

    def _mean_output_sample(self, artifact_store: RunArtifactStore) -> WeightSample:
        return tree_mean(self._output_samples(artifact_store))

    def _reference_stability(
        self,
        previous_artifact_store: RunArtifactStore,
        current_artifact_store: RunArtifactStore,
        *,
        previous_pass: int,
        current_pass: int,
    ) -> dict[str, Any]:
        match_executor = self._get_stage_executor("match")
        if match_executor is None or match_executor.graph is None:
            raise RuntimeError(
                "Reference stability requires a prepared match executor."
            )
        match_config = self.config.match
        if match_config is None:
            raise RuntimeError("Reference stability requires match configuration.")
        return reference_stability_diagnostic(
            previous_samples=lambda: self._output_samples(previous_artifact_store),
            current_samples=lambda: self._output_samples(current_artifact_store),
            graph=match_executor.graph,
            solvers=match_config.solvers,
            previous_pass=previous_pass,
            current_pass=current_pass,
        )

    def _write_dry_run_summary(self, pending: list[SampleRecord]) -> None:
        summary_path = self.run_state.state_dir / "dry_run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "manifest": self.manifest.summary(),
            "pending_samples": len(pending),
            "output_dir": str(self.run_state.output_dir),
        }
        summary.update(self._dry_run_summary_extra())
        summary_path.write_text(json.dumps(summary, indent=2))
        self.progress_logger.info("Dry run summary written to %s", summary_path)

    def _maybe_log_progress(self, label: str | None = None) -> None:
        # The progress bar already shows progress; log lines only add value in
        # debug mode.
        if self.progress_logger.getEffectiveLevel() >= logging.INFO:
            return
        processed = self.run_state.processed_count()
        now = time.time()
        if processed <= self._last_logged_count:
            return
        if (processed - self._last_logged_count) < LOG_SAMPLE_INTERVAL and (
            now - self._last_log_time
        ) < LOG_TIME_INTERVAL_SECONDS:
            return
        percent = (processed / max(1, self.manifest.total)) * 100.0
        suffix = f" | last={label}" if label else ""
        self.progress_logger.info(
            "Processed %d/%d samples (%.2f%%)%s",
            processed,
            self.manifest.total,
            percent,
            suffix,
        )
        self._last_logged_count = processed
        self._last_log_time = now

    def _maybe_persist_state(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_save_time) < STATE_SAVE_INTERVAL_SECONDS:
            return
        self.run_state.save()
        self.artifact_store.maybe_flush(force=force)
        self._last_save_time = now

    def _prepare_executors(
        self,
        reference_sample: WeightSample,
        *,
        match_reference: WeightSample | None = None,
        pass_index: int = 0,
    ) -> list[tuple[str, StageExecutor]]:
        executors: dict[str, StageExecutor] = {}
        family = self.config.architecture.family
        recipe_kwargs = dict(self.config.architecture.recipe_kwargs)
        if self.config.canonicalize is not None:
            executors["canonicalize"] = CanonicalizeExecutor(
                self.config.canonicalize,
                family=family,
                recipe_kwargs=recipe_kwargs,
            )
        if self.config.center_softmax_head is not None:
            executors["center_softmax_head"] = CenterSoftmaxHeadExecutor(
                self.config.center_softmax_head,
                family=family,
                recipe_kwargs=recipe_kwargs,
            )
        if self.config.match is not None:
            executors["match"] = MatchExecutor(
                self.config.match,
                reference_index=(
                    self.manifest.reference_index if pass_index == 0 else None
                ),
                seed=self.config.match.seed,
                rng_offset=pass_index * self.manifest.total,
                batch_size=self.per_device_batch,
                family=family,
                recipe_kwargs=recipe_kwargs,
            )

        stage_list = [
            (name, executors[name]) for name in self.stage_order if name in executors
        ]
        return prepare_stage_executors(
            stage_list,
            self.manifest,
            reference_sample,
            match_reference=match_reference,
        )

    def _get_stage_executor(self, name: str) -> StageExecutor | None:
        for stage_name, executor in self._stage_executors:
            if stage_name == name:
                return executor
        return None

    def _visible_gpu_ids(self, device_ids: Sequence[int] | None) -> list[int]:
        try:
            import jax

            if device_ids:
                available = [
                    int(dev.id)
                    for dev in jax.devices()
                    if is_gpu_platform(dev.platform)
                ]
                return [dev_id for dev_id in device_ids if dev_id in available]
            return [
                int(device.id)
                for device in jax.devices()
                if is_gpu_platform(device.platform)
            ]
        except Exception:  # pragma: no cover - backend specific
            return []

    def _run_local(
        self,
        pending: list[SampleRecord],
        loader: SampleLoader,
        processed_initial: int,
        *,
        output_artifact_store: RunArtifactStore,
        final_pass: bool,
    ) -> float:
        start = time.time()
        pipeline = StagePipeline(
            self._stage_executors,
            save_intermediate=self.save_intermediate,
        )
        use_batched = pipeline.can_batch_match(self.per_device_batch)
        batch_size = self.per_device_batch if use_batched else 1

        with progress_bar(
            total=self.manifest.total,
            initial=min(processed_initial, self.manifest.total),
        ) as bar:
            if use_batched:
                self.progress_logger.info(
                    "Using batched matching with batch_size=%d", batch_size
                )
            pipeline.run_records(
                pending,
                loader,
                _RunnerPipelineSink(
                    self,
                    bar,
                    output_artifact_store=output_artifact_store,
                    final_pass=final_pass,
                ),
                batch_size=batch_size,
            )

        self._maybe_persist_state(force=True)
        return time.time() - start

    def _run_parallel(
        self,
        pending: list[SampleRecord],
        processed_initial: int,
        *,
        output_artifact_store: RunArtifactStore,
        final_pass: bool,
    ) -> float:
        start = time.time()
        scratch_root = self.run_state.state_dir / "scratch"
        scratch_root.mkdir(parents=True, exist_ok=True)

        match_executor = self._get_stage_executor("match")
        prefers_gpu = match_executor.prefers_gpu if match_executor else False
        pool = WorkerPool(
            parallelism=self.parallelism,
            device_ids=self.config.runtime.device_ids,
            strategy_prefers_gpu=prefers_gpu,
        )

        record_lookup = {record.index: record for record in pending}
        pending_indices = deque(record.index for record in pending)
        total_samples = len(pending_indices)
        worker_count = min(self.parallelism, total_samples)
        chunk_size = self._worker_chunk_size()
        job_template = self._worker_job_template()

        pool.start(
            worker_count=worker_count,
            job_template=job_template,
            scratch_root=scratch_root,
        )

        completed = 0
        try:
            with progress_bar(
                total=self.manifest.total,
                initial=min(processed_initial, self.manifest.total),
            ) as bar:
                while completed < total_samples:
                    message = pool.next_message(timeout=1.0)
                    if message is None:
                        self._check_workers(
                            pool,
                            pending_indices,
                            job_template,
                            scratch_root,
                            chunk_size,
                        )
                        continue

                    worker_id = int(message.get("worker_id", -1))
                    state = pool.states.get(worker_id)
                    if state is None or message.get("generation") != state.generation:
                        continue
                    state.last_heartbeat = time.time()
                    msg_type = message.get("type")

                    if msg_type == "ready":
                        state.ready = True
                        self._assign_work(state, pending_indices, chunk_size)
                    elif msg_type == "heartbeat":
                        pass
                    elif msg_type == "commit":
                        sample_index = int(message["sample_index"])
                        record = record_lookup[sample_index]
                        self._apply_commit(
                            record,
                            artifacts=message.get("artifacts", {}),
                            checksums=message.get("checksums"),
                            diagnostics=message.get("diagnostics") or {},
                            output_artifact_store=output_artifact_store,
                            final_pass=final_pass,
                        )
                        try:
                            state.assigned.remove(sample_index)
                        except ValueError:
                            pass
                        if final_pass:
                            self.run_state.record_progress(sample_index=sample_index)
                            self._maybe_persist_state()
                        bar.update(label=record.label)
                        if final_pass:
                            self._maybe_log_progress(record.label)
                        completed += 1
                    elif msg_type == "error":
                        reason = message.get("error", "worker reported error")
                        self._handle_worker_failure(
                            pool,
                            state,
                            pending_indices,
                            job_template,
                            scratch_root,
                            chunk_size,
                            reason,
                        )
                    elif msg_type == "stopped":
                        state.stopping = True
                        state.ready = False
                    else:
                        self.progress_logger.debug(
                            "Ignoring worker message: %s", msg_type
                        )

                    self._check_workers(
                        pool, pending_indices, job_template, scratch_root, chunk_size
                    )
        except KeyboardInterrupt:
            self.progress_logger.warning("Interrupt received, stopping workers...")
            pool.shutdown(force=False)
            self.run_state.save()
            self.artifact_store.maybe_flush(force=True)
            raise
        finally:
            pool.shutdown(force=False)

        self._maybe_persist_state(force=True)
        return time.time() - start

    def _assign_work(self, state, pending_indices: deque[int], chunk_size: int) -> None:
        if state.stopping or not state.ready:
            return
        if not pending_indices:
            return
        chunk: list[int] = []
        while pending_indices and len(chunk) < chunk_size:
            chunk.append(pending_indices.popleft())
        if not chunk:
            return
        state.assigned.extend(chunk)
        state.ready = False
        state.command_queue.put({"type": "process", "record_indices": chunk})

    def _requeue_assigned(self, state, pending_indices: deque[int]) -> None:
        if not state.assigned:
            return
        pending_indices.extendleft(reversed(state.assigned))
        state.assigned.clear()

    def _handle_worker_failure(
        self,
        pool: WorkerPool,
        state,
        pending_indices: deque[int],
        job_template: dict[str, Any],
        scratch_root: Path,
        chunk_size: int,
        reason: str,
    ) -> None:
        if state.stopping:
            return
        self.progress_logger.warning(
            "Worker %d failed (%s). Respawning.", state.worker_id, reason
        )
        self._requeue_assigned(state, pending_indices)
        try:
            state.command_queue.close()
        except Exception:
            pass
        if state.process.is_alive():
            state.process.terminate()
            state.process.join(timeout=5)
            if state.process.is_alive():
                try:
                    state.process.kill()
                except Exception:
                    pass
                state.process.join(timeout=2)
        shutil.rmtree(state.scratch_dir, ignore_errors=True)
        remaining = len(pending_indices) + sum(
            len(item.assigned) for item in pool.states.values()
        )
        if remaining > 0:
            pool.respawn(state.worker_id, job_template, scratch_root)
        else:
            state.stopping = True

    def _check_workers(
        self,
        pool: WorkerPool,
        pending_indices: deque[int],
        job_template: dict[str, Any],
        scratch_root: Path,
        chunk_size: int,
    ) -> None:
        now = time.time()
        for _worker_id, state in list(pool.states.items()):
            if state.stopping:
                continue
            if not state.process.is_alive():
                self._handle_worker_failure(
                    pool,
                    state,
                    pending_indices,
                    job_template,
                    scratch_root,
                    chunk_size,
                    f"exitcode={state.process.exitcode}",
                )
                continue
            timeout_seconds = WORKER_HEARTBEAT_TIMEOUT * max(1, len(state.assigned))
            if state.assigned and (now - state.last_heartbeat) > timeout_seconds:
                self._handle_worker_failure(
                    pool,
                    state,
                    pending_indices,
                    job_template,
                    scratch_root,
                    chunk_size,
                    "heartbeat timeout",
                )

    def _worker_chunk_size(self) -> int:
        match_executor = self._get_stage_executor("match")
        if match_executor and match_executor.supports_batching:
            return max(1, self.per_device_batch)
        return 1

    def _worker_job_template(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.run_state.manifest_path),
            "stages": [name for name, _ in self._stage_executors],
            "canonicalize_config": self.config.canonicalize,
            "center_softmax_head_config": self.config.center_softmax_head,
            "match_config": self.config.match,
            "seed": self.config.match.seed if self.config.match else None,
            "rng_offset": self._refinement_pass_index * self.manifest.total,
            "match_reference_path": str(self._match_reference_path)
            if self._match_reference_path is not None
            else None,
            "match_reference_index": self._match_reference_index,
            "per_device_batch": self.per_device_batch,
            "save_intermediate": self.save_intermediate,
            "family": self.config.architecture.family,
            "recipe_kwargs": dict(self.config.architecture.recipe_kwargs),
            "heartbeat_interval": WORKER_HEARTBEAT_INTERVAL,
        }

    def _apply_commit(
        self,
        record: SampleRecord,
        *,
        artifacts: Mapping[str, Any],
        checksums: Mapping[str, int] | None,
        diagnostics: Mapping[str, Any],
        output_artifact_store: RunArtifactStore | None = None,
        final_pass: bool = True,
    ) -> None:
        artifact_store = output_artifact_store or self.artifact_store
        artifact_store.commit_from_scratch(
            record, artifacts=artifacts, checksums=checksums
        )
        if final_pass and diagnostics:
            artifact_store.write_diagnostics(record, diagnostics)
        sample_scratch = artifacts.get("scratch_dir")
        if sample_scratch:
            shutil.rmtree(Path(sample_scratch), ignore_errors=True)


__all__ = ["AlignmentRunner"]
