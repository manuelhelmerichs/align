"""AlignRunner with modular stage executors and parallel support."""

import json
import logging
import shutil
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import AlignConfig
from ..logging_utils import progress_bar
from ..state import RunManifest, SampleManifest, SampleRecord
from .common import (
    _LOG_SAMPLE_INTERVAL,
    _LOG_TIME_INTERVAL_SECONDS,
    _STATE_SAVE_INTERVAL_SECONDS,
    _WORKER_HEARTBEAT_INTERVAL,
    _WORKER_HEARTBEAT_TIMEOUT,
    available_cpu_count,
)
from .loaders import PrefetchingLoader, SampleLoader
from .loggers import AlignLogger
from .parallel import WorkerPool
from .stages import NormalizeExecutor, RebasinExecutor, StageExecutor

_LOG = logging.getLogger(__name__)


class AlignRunner:
    """Runner that composes stage executors and optional parallelism."""

    def __init__(
        self,
        config: AlignConfig,
        manifest: SampleManifest,
        run_manifest: RunManifest,
        logger: AlignLogger,
        *,
        progress_logger: logging.Logger | None = None,
    ) -> None:
        self.manifest = manifest
        self.run_manifest = run_manifest
        self.logger = logger
        self.progress_logger = progress_logger or _LOG
        self.config = config
        self.save_intermediate = bool(config.runtime.save_intermediate)
        self.per_device_batch = max(
            1, int(getattr(config.runtime, "per_device_batch", None) or 1)
        )
        self.stage_order = config.active_stages()
        self.parallelism = self._compute_parallelism()
        self._stage_executors: list[tuple[str, StageExecutor]] = []
        now = time.time()
        processed_so_far = self.run_manifest.processed_count()
        self._log_sample_interval = _LOG_SAMPLE_INTERVAL
        self._log_time_interval = _LOG_TIME_INTERVAL_SECONDS
        self._state_save_interval = _STATE_SAVE_INTERVAL_SECONDS
        self._last_logged_count = processed_so_far
        self._last_log_time = now
        self._last_save_time = now
        self._progress_bar_uses_tqdm = False
        self._validate_artifacts_on_resume = bool(
            config.runtime.validate_artifacts_on_resume
        )

    def _dry_run_summary_extra(self) -> dict[str, Any]:
        return {"stages": self.stage_order}

    def _compute_parallelism(self) -> int:
        requested = self.config.runtime.parallelism
        if requested is not None:
            return max(1, int(requested))

        rebasin_cfg = self.config.rebasin
        if (
            rebasin_cfg
            and rebasin_cfg.enabled
            and any(step.solver == "sinkhorn" for step in rebasin_cfg.schedule)
        ):
            gpu_count = len(self._visible_gpu_ids(self.config.runtime.device_ids))
            if gpu_count > 0:
                return gpu_count
        return available_cpu_count()

    def execute(self, *, dry_run: bool = False) -> None:
        pending = self._pending_records()
        processed = self.run_manifest.processed_count()
        self.progress_logger.info(
            "Aligning %d/%d samples (resume=%d processed) | stages=%s",
            len(pending),
            self.manifest.total,
            self.manifest.total - len(pending),
            ",".join(self.stage_order),
        )
        if dry_run:
            self._write_dry_run_summary(pending)
            return

        if not pending:
            self.progress_logger.info("All samples already processed. Nothing to do.")
            return

        loader = SampleLoader(self.manifest)
        ref_sample = loader.load_reference()
        self._stage_executors = self._prepare_executors(ref_sample)

        use_parallel = self.parallelism > 1 and len(pending) > 1
        if use_parallel:
            elapsed = self._run_parallel(pending, processed)
        else:
            elapsed = self._run_local(pending, loader, processed)

        total_elapsed = self.run_manifest.add_elapsed(elapsed)
        self.logger.finalize(elapsed_seconds=total_elapsed)
        self.run_manifest.mark_complete()
        self.run_manifest.save()
        self.progress_logger.info("Align complete in %.2fs", elapsed)

    def _pending_records(self) -> list[SampleRecord]:
        pending: list[SampleRecord] = []
        for record in self.manifest.records:
            if not self.run_manifest.is_processed(record.index):
                pending.append(record)
                continue
            if (
                self._validate_artifacts_on_resume
                and not self.logger.validate_artifacts(record)
            ):
                self.run_manifest.mark_unprocessed(record.index)
                pending.append(record)
                self.progress_logger.warning(
                    "Artifact checksum mismatch for %s. Sample will be reprocessed.",
                    record.label,
                )
        self.logger.maybe_flush(force=True)
        return pending

    def _write_dry_run_summary(self, pending: list[SampleRecord]) -> None:
        summary_path = self.run_manifest.state_dir / "dry_run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "manifest": self.manifest.summary(),
            "pending_samples": len(pending),
            "output_dir": str(self.run_manifest.output_dir),
        }
        summary.update(self._dry_run_summary_extra())
        summary_path.write_text(json.dumps(summary, indent=2))
        self.progress_logger.info("Dry run summary written to %s", summary_path)

    def _maybe_log_progress(self, label: str | None = None) -> None:
        if (
            self._progress_bar_uses_tqdm
            and self.progress_logger.getEffectiveLevel() >= logging.INFO
        ):
            return
        processed = self.run_manifest.processed_count()
        now = time.time()
        if processed <= self._last_logged_count:
            return
        if (processed - self._last_logged_count) < self._log_sample_interval and (
            now - self._last_log_time
        ) < self._log_time_interval:
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
        if not force and (now - self._last_save_time) < self._state_save_interval:
            return
        self.run_manifest.save()
        self.logger.maybe_flush(force=force)
        self._last_save_time = now

    def _prepare_executors(self, ref_sample) -> list[tuple[str, StageExecutor]]:
        executors: dict[str, StageExecutor] = {}
        adapter_kwargs = dict(getattr(self.config, "adapter", {}) or {})
        if self.config.normalize and self.config.normalize.enabled:
            executors["normalize"] = NormalizeExecutor(
                self.config.normalize,
                architecture=self.config.architecture,
                adapter_kwargs=adapter_kwargs,
            )
        if self.config.rebasin and self.config.rebasin.enabled:
            executors["rebasin"] = RebasinExecutor(
                self.config.rebasin,
                reference_index=self.manifest.reference_index,
                seed=self.config.rebasin.seed,
                batch_size=self.per_device_batch,
                architecture=self.config.architecture,
                adapter_kwargs=adapter_kwargs,
            )

        stage_list: list[tuple[str, StageExecutor]] = []
        ref_current = ref_sample
        for name in self.stage_order:
            executor = executors.get(name)
            if executor is None:
                continue
            executor.prepare(self.manifest, ref_current)
            ref_current = executor.reference_output(
                self.manifest.reference_record, ref_current
            )
            stage_list.append((name, executor))
        return stage_list

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
                    int(dev.id) for dev in jax.devices() if dev.platform == "gpu"
                ]
                return [dev_id for dev_id in device_ids if dev_id in available]
            return [
                int(device.id) for device in jax.devices() if device.platform == "gpu"
            ]
        except Exception:  # pragma: no cover - backend specific
            return []

    def _run_local(
        self, pending: list[SampleRecord], loader: SampleLoader, processed_initial: int
    ) -> float:
        start = time.time()
        use_batched = self._can_batch_rebasin()
        batch_size = self.per_device_batch if use_batched else 1

        try:
            with progress_bar(
                total=self.manifest.total,
                initial=min(processed_initial, self.manifest.total),
            ) as bar:
                self._progress_bar_uses_tqdm = getattr(bar, "uses_tqdm", False)
                if use_batched:
                    self.progress_logger.info(
                        "Using batched rebasin with batch_size=%d", batch_size
                    )
                    self._execute_batched(pending, loader, batch_size, bar)
                else:
                    self._execute_sequential(pending, loader, bar)
        finally:
            self._progress_bar_uses_tqdm = False

        self._maybe_persist_state(force=True)
        return time.time() - start

    def _execute_sequential(
        self, pending: list[SampleRecord], loader: SampleLoader, bar
    ) -> None:
        for record in pending:
            sample = loader.load(record)
            aux_payload: dict[str, Any] = {}
            current = sample

            for idx, (stage, executor) in enumerate(self._stage_executors):
                last_stage = idx == len(self._stage_executors) - 1
                result = executor.process_single(record, current)
                current = result.sample

                if result.aux:
                    aux_payload[stage] = result.aux
                if stage == "normalize":
                    scale_factors = result.artifacts.get("scale_factors")
                    if scale_factors is not None:
                        self.logger.write_scale_factors(record, scale_factors)
                elif stage == "rebasin":
                    perms = result.artifacts.get("permutations")
                    if perms is not None and self.config.rebasin:
                        self.logger.write_permutations(record, perms)

                if self.save_intermediate and not last_stage:
                    self.logger.write_intermediate(record, current)

            self.logger.write_sample(record, current)
            self.logger.write_aux(record, aux_payload)
            self.run_manifest.record_progress(record=record)
            self._maybe_persist_state()
            bar.update(label=record.label)
            self._maybe_log_progress(record.label)

    def _execute_batched(
        self,
        pending: list[SampleRecord],
        loader: SampleLoader,
        batch_size: int,
        bar,
    ) -> None:
        rebasin_executor = self._get_stage_executor("rebasin")
        if rebasin_executor is None:
            raise ValueError(
                "Batched execution requested but no rebasin stage configured."
            )

        prefetch_loader = PrefetchingLoader(loader, prefetch_count=batch_size)

        for batch_start in range(0, len(pending), batch_size):
            batch_records = pending[batch_start : batch_start + batch_size]

            next_start = batch_start + batch_size
            if next_start < len(pending):
                prefetch_loader.prefetch(pending[next_start : next_start + batch_size])

            sample_batch = []
            stage_meta: list[dict[str, Any]] = []
            for record in batch_records:
                sample = prefetch_loader.get(record)
                current = sample
                aux_payload: dict[str, Any] = {}
                scale_factors = None
                intermediate_params = None

                for idx, (stage, executor) in enumerate(self._stage_executors):
                    last_stage = idx == len(self._stage_executors) - 1
                    if stage == "rebasin" and last_stage:
                        break
                    result = executor.process_single(record, current)
                    current = result.sample
                    if result.aux:
                        aux_payload[stage] = result.aux
                    if stage == "normalize":
                        scale_factors = result.artifacts.get("scale_factors")
                    if self.save_intermediate and not last_stage:
                        intermediate_params = current

                sample_batch.append(current)
                stage_meta.append(
                    {
                        "record": record,
                        "aux": aux_payload,
                        "scale_factors": scale_factors,
                        "intermediate": intermediate_params,
                    }
                )

            rebasin_results = rebasin_executor.process_batch(
                batch_records, sample_batch
            )
            for meta_entry, reb_result in zip(stage_meta, rebasin_results, strict=True):
                aux_payload = dict(meta_entry["aux"])
                if reb_result.aux:
                    aux_payload["rebasin"] = reb_result.aux
                if (
                    reb_result.artifacts.get("permutations") is not None
                    and self.config.rebasin
                ):
                    self.logger.write_permutations(
                        meta_entry["record"],
                        reb_result.artifacts["permutations"],
                    )
                if meta_entry["scale_factors"] is not None:
                    self.logger.write_scale_factors(
                        meta_entry["record"], meta_entry["scale_factors"]
                    )
                if self.save_intermediate and meta_entry["intermediate"] is not None:
                    self.logger.write_intermediate(
                        meta_entry["record"], meta_entry["intermediate"]
                    )

                self.logger.write_sample(
                    meta_entry["record"],
                    reb_result.sample,
                )
                self.logger.write_aux(meta_entry["record"], aux_payload)
                self.run_manifest.record_progress(record=meta_entry["record"])
                bar.update(label=meta_entry["record"].label)
                self._maybe_log_progress(meta_entry["record"].label)

            self._maybe_persist_state()

        prefetch_loader.clear()

    def _can_batch_rebasin(self) -> bool:
        if not self._stage_executors or self._stage_executors[-1][0] != "rebasin":
            return False
        rebasin_executor = self._stage_executors[-1][1]
        return rebasin_executor.supports_batching and self.per_device_batch > 1

    def _run_parallel(
        self, pending: list[SampleRecord], processed_initial: int
    ) -> float:
        start = time.time()
        scratch_root = self.run_manifest.state_dir / "scratch"
        scratch_root.mkdir(parents=True, exist_ok=True)

        rebasin_executor = self._get_stage_executor("rebasin")
        prefers_gpu = rebasin_executor.prefers_gpu if rebasin_executor else False
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
                self._progress_bar_uses_tqdm = getattr(bar, "uses_tqdm", False)
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
                            aux=message.get("aux") or {},
                        )
                        try:
                            state.assigned.remove(sample_index)
                        except ValueError:
                            pass
                        self.run_manifest.record_progress(sample_index=sample_index)
                        self._maybe_persist_state()
                        bar.update(label=record.label)
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
            self.run_manifest.save()
            self.logger.maybe_flush(force=True)
            raise
        finally:
            pool.shutdown(force=False)
            self._progress_bar_uses_tqdm = False

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
            new_state = pool.respawn(state.worker_id, job_template, scratch_root)
            if new_state:
                pool.states[state.worker_id] = new_state
                if new_state.ready:
                    self._assign_work(new_state, pending_indices, chunk_size)
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
            timeout_seconds = _WORKER_HEARTBEAT_TIMEOUT * max(1, len(state.assigned))
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
        rebasin_executor = self._get_stage_executor("rebasin")
        if rebasin_executor and rebasin_executor.supports_batching:
            return max(1, self.per_device_batch)
        return 1

    def _worker_job_template(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.run_manifest.manifest_path),
            "stages": [name for name, _ in self._stage_executors],
            "normalize_config": self.config.normalize
            if self.config.normalize and self.config.normalize.enabled
            else None,
            "rebasin_config": self.config.rebasin
            if self.config.rebasin and self.config.rebasin.enabled
            else None,
            "seed": self.config.rebasin.seed if self.config.rebasin else None,
            "per_device_batch": self.per_device_batch,
            "save_intermediate": self.save_intermediate,
            "architecture": self.config.architecture,
            "adapter_kwargs": dict(getattr(self.config, "adapter", {}) or {}),
            "heartbeat_interval": _WORKER_HEARTBEAT_INTERVAL,
        }

    def _apply_commit(
        self,
        record: SampleRecord,
        *,
        artifacts: Mapping[str, Any],
        checksums: Mapping[str, int] | None,
        aux: Mapping[str, Any],
    ) -> None:
        self.logger.commit_from_scratch(
            record, artifacts=artifacts, checksums=checksums
        )
        if aux:
            self.logger.write_aux(record, aux)
        sample_scratch = artifacts.get("scratch_dir")
        if sample_scratch:
            shutil.rmtree(Path(sample_scratch), ignore_errors=True)


__all__ = ["AlignRunner"]
