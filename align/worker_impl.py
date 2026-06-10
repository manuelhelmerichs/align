"""Worker entrypoint used by the align runner."""

from __future__ import annotations

import os
import queue
import shutil
import threading
import traceback
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifacts import (
    write_aux_artifact,
    write_permutations_artifact,
    write_scale_factors_artifact,
)
from .state import SampleManifest, SampleRecord, _file_checksum

if TYPE_CHECKING:
    from .rebasin import ParamTree


class _ArtifactWriter:
    """Persists stage outputs to a scratch directory for the parent to commit."""

    def __init__(
        self, scratch_root: Path, *, rebasin_method: str | None, save_intermediate: bool
    ) -> None:
        self.root = Path(scratch_root)
        self.rebasin_method = (rebasin_method or "").lower() if rebasin_method else None
        self.save_intermediate = bool(save_intermediate)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        record: SampleRecord,
        *,
        final_params: ParamTree,
        permutations,
        scale_factors,
        intermediate_params: ParamTree | None = None,
        aux: dict[str, Any] | None = None,
    ) -> tuple[dict[str, str | None], dict[str, int]]:
        from .rebasin import (
            save_pytree_npz,
        )  # Imported lazily to honor worker device settings

        sample_dir = self.root / f"sample_{record.index}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, str | None] = {"scratch_dir": str(sample_dir)}
        checksums: dict[str, int] = {}

        final_path = sample_dir / "final.npz"
        save_pytree_npz(final_path, final_params)
        artifacts["final_path"] = str(final_path)
        checksums["final"] = _file_checksum(final_path)

        perm_path = None
        if permutations is not None and self.rebasin_method:
            perm_path = write_permutations_artifact(
                sample_dir / "permutations.npz",
                permutations,
                method=self.rebasin_method,
            )
        artifacts["permutations_path"] = str(perm_path) if perm_path else None
        checksums["permutations"] = _file_checksum(perm_path) if perm_path else 0

        scale_path = None
        if scale_factors is not None:
            scale_path = write_scale_factors_artifact(
                sample_dir / "scale_factors.npz", scale_factors
            )
        artifacts["scale_factors_path"] = str(scale_path) if scale_path else None
        checksums["scale_factors"] = _file_checksum(scale_path) if scale_path else 0

        if self.save_intermediate:
            intermediate_path = None
            if intermediate_params is not None:
                intermediate_path = sample_dir / "intermediate.npz"
                save_pytree_npz(intermediate_path, intermediate_params)
            artifacts["intermediate_path"] = (
                str(intermediate_path) if intermediate_path else None
            )
            checksums["intermediate"] = (
                _file_checksum(intermediate_path) if intermediate_path else 0
            )

        aux_path = write_aux_artifact(sample_dir / "aux.json", aux)
        artifacts["aux_path"] = str(aux_path) if aux_path else None

        return artifacts, checksums


class _WorkerLoop:
    def __init__(
        self,
        job: dict[str, Any],
        manifest: SampleManifest,
        loader,
        stages: list[tuple[str, Any]],
        *,
        save_intermediate: bool,
        per_device_batch: int,
        command_queue,
        progress_queue,
    ) -> None:
        self.worker_id = int(job["worker_id"])
        self.generation = int(job.get("generation", 0))
        self.command_queue = command_queue
        self.progress_queue = progress_queue
        self.manifest = manifest
        self.loader = loader
        self.stages = stages
        self.save_intermediate = bool(save_intermediate)
        self.per_device_batch = max(1, int(per_device_batch))

        self.rebasin_method = None
        for name, executor in self.stages:
            if name == "rebasin":
                self.rebasin_method = executor.config.method
                break

        self._artifact_writer = _ArtifactWriter(
            Path(job["scratch_dir"]),
            rebasin_method=self.rebasin_method,
            save_intermediate=self.save_intermediate,
        )

        self._heartbeat_interval = float(job.get("heartbeat_interval") or 5.0)
        self._heartbeat_stop = threading.Event()
        self._heartbeat_lock = threading.Lock()
        self._heartbeat_active = False
        self._heartbeat_sample: int | None = None
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()
        self._seq = 0

    def run(self) -> None:
        try:
            self._send_ready()
            while True:
                try:
                    command = self.command_queue.get()
                except (EOFError, OSError):
                    break
                if not command:
                    continue
                cmd_type = command.get("type")
                if cmd_type == "process":
                    indices = command.get("record_indices") or []
                    records = [self.manifest.records[idx] for idx in indices]
                    if records:
                        try:
                            self._handle_records(records)
                        except Exception as exc:  # pragma: no cover - worker crash path
                            self._send_message(
                                {
                                    "type": "error",
                                    "error": str(exc),
                                    "traceback": traceback.format_exc(),
                                    "sample_indices": [rec.index for rec in records],
                                }
                            )
                            raise
                    self._send_ready()
                elif cmd_type == "stop":
                    break
        finally:
            self._shutdown()

    def _handle_records(self, records: list[SampleRecord]) -> None:
        if not records:
            return
        pending = deque(records)
        self._set_active_sample(pending[0].index if pending else None)
        try:
            if self._can_batch_rebasin():
                self._process_batched(records)
            else:
                self._process_sequential(records)
        finally:
            self._set_active_sample(None)

    def _can_batch_rebasin(self) -> bool:
        if not self.stages or self.stages[-1][0] != "rebasin":
            return False
        executor = self.stages[-1][1]
        return executor.supports_batching and self.per_device_batch > 1

    def _process_sequential(self, records: list[SampleRecord]) -> None:
        for record in records:
            self._set_active_sample(record.index)
            params = self.loader.load(record)
            aux_payload: dict[str, Any] = {}
            intermediate_params = None
            permutations = None
            scale_factors = None

            for idx, (name, executor) in enumerate(self.stages):
                last_stage = idx == len(self.stages) - 1
                result = executor.process_single(record, params)
                params = result.params
                if result.aux:
                    aux_payload[name] = result.aux
                if name == "normalize":
                    scale_factors = result.artifacts.get("scale_factors")
                if name == "rebasin":
                    permutations = result.artifacts.get("permutations")
                if self.save_intermediate and not last_stage:
                    intermediate_params = params

            self._emit_commit(
                record,
                params,
                permutations=permutations,
                scale_factors=scale_factors,
                aux=aux_payload,
                intermediate_params=intermediate_params,
            )

    def _process_batched(self, records: list[SampleRecord]) -> None:
        from .runtime.loaders import (
            PrefetchingLoader,
        )  # Imported lazily to respect device visibility

        rebasin_executor = self.stages[-1][1]
        batch_size = self.per_device_batch
        prefetch_loader = PrefetchingLoader(self.loader, prefetch_count=batch_size)

        for batch_start in range(0, len(records), batch_size):
            batch_records = records[batch_start : batch_start + batch_size]
            next_start = batch_start + batch_size
            if next_start < len(records):
                prefetch_loader.prefetch(records[next_start : next_start + batch_size])

            params_batch = []
            meta: list[dict[str, Any]] = []
            for record in batch_records:
                self._set_active_sample(record.index)
                params = prefetch_loader.get(record)
                current = params
                aux_payload: dict[str, Any] = {}
                scale_factors = None
                intermediate_params = None

                for idx, (name, executor) in enumerate(self.stages):
                    last_stage = idx == len(self.stages) - 1
                    if name == "rebasin" and last_stage:
                        break
                    result = executor.process_single(record, current)
                    current = result.params
                    if result.aux:
                        aux_payload[name] = result.aux
                    if name == "normalize":
                        scale_factors = result.artifacts.get("scale_factors")
                    if self.save_intermediate and not last_stage:
                        intermediate_params = current

                params_batch.append(current)
                meta.append(
                    {
                        "record": record,
                        "aux": aux_payload,
                        "scale_factors": scale_factors,
                        "intermediate": intermediate_params,
                    }
                )

            batch_results: list[Any] = rebasin_executor.process_batch(
                batch_records, params_batch
            )
            for meta_entry, reb_result in zip(meta, batch_results, strict=True):
                aux_payload = dict(meta_entry["aux"])
                if reb_result.aux:
                    aux_payload["rebasin"] = reb_result.aux
                self._emit_commit(
                    meta_entry["record"],
                    reb_result.params,
                    permutations=reb_result.artifacts.get("permutations"),
                    scale_factors=meta_entry["scale_factors"],
                    aux=aux_payload,
                    intermediate_params=meta_entry["intermediate"],
                )

        prefetch_loader.clear()

    def _emit_commit(
        self,
        record: SampleRecord,
        params: ParamTree,
        *,
        permutations,
        scale_factors,
        aux: dict[str, Any] | None,
        intermediate_params: ParamTree | None,
    ) -> None:
        artifacts, checksums = self._artifact_writer.write(
            record,
            final_params=params,
            permutations=permutations,
            scale_factors=scale_factors,
            intermediate_params=intermediate_params,
            aux=aux,
        )
        payload = {
            "type": "commit",
            "sample_index": record.index,
            "sample_label": record.label,
            "artifacts": artifacts,
            "checksums": checksums,
            "aux": aux or {},
        }
        self._send_message(payload)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            with self._heartbeat_lock:
                if not self._heartbeat_active:
                    continue
                sample_index = self._heartbeat_sample
            self._send_message(
                {"type": "heartbeat", "sample_index": sample_index}, block=False
            )

    def _set_active_sample(self, sample_index: int | None) -> None:
        with self._heartbeat_lock:
            if sample_index is None:
                self._heartbeat_active = False
                self._heartbeat_sample = None
            else:
                self._heartbeat_active = True
                self._heartbeat_sample = int(sample_index)

    def _send_ready(self) -> None:
        self._send_message({"type": "ready"})

    def _send_message(
        self,
        payload: dict[str, Any],
        *,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        self._seq += 1
        payload.update(
            {
                "worker_id": self.worker_id,
                "generation": self.generation,
                "seq": self._seq,
            }
        )
        try:
            self.progress_queue.put(payload, block=block, timeout=timeout)
        except queue.Full:
            if block:
                raise

    def _shutdown(self) -> None:
        self._set_active_sample(None)
        self._send_message({"type": "heartbeat", "sample_index": None}, block=False)
        self._heartbeat_stop.set()
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2)
        self._send_message({"type": "stopped"})


def _build_stage_executors(
    job: dict[str, Any], manifest: SampleManifest, ref_params: ParamTree
):
    from .runtime.stages import NormalizeExecutor, RebasinExecutor

    stages: list[tuple[str, Any]] = []
    normalize_cfg = job.get("normalize_config")
    rebasin_cfg = job.get("rebasin_config")
    stage_order = job.get("stages") or []
    architecture = job.get("architecture", "dense_mlp")
    adapter_kwargs = job.get("adapter_kwargs") or {}

    if normalize_cfg:
        stages.append(
            (
                "normalize",
                NormalizeExecutor(
                    normalize_cfg,
                    architecture=architecture,
                    adapter_kwargs=adapter_kwargs,
                ),
            )
        )
    if rebasin_cfg:
        stages.append(
            (
                "rebasin",
                RebasinExecutor(
                    rebasin_cfg,
                    reference_index=manifest.reference_index,
                    seed=job.get("seed"),
                    batch_size=int(job.get("per_device_batch") or 1),
                    architecture=architecture,
                    adapter_kwargs=adapter_kwargs,
                ),
            )
        )

    if stage_order:
        name_to_exec = {name: exec for name, exec in stages}
        stages = [
            (name, name_to_exec[name]) for name in stage_order if name in name_to_exec
        ]

    ref_current = ref_params
    for _, executor in stages:
        executor.prepare(manifest, ref_current)
        ref_current = executor.reference_output(manifest.reference_record, ref_current)

    return stages


def run_worker(job: dict[str, Any], command_queue, progress_queue) -> None:
    device_id = job.get("device_id")
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        os.environ["JAX_VISIBLE_DEVICES"] = str(device_id)

    manifest = SampleManifest.load(Path(job["manifest_path"]))

    from .runtime.loaders import SampleLoader  # Imported after device visibility is set

    loader = SampleLoader(manifest)
    ref_params = loader.load_reference()
    stages = _build_stage_executors(job, manifest, ref_params)

    loop = _WorkerLoop(
        job,
        manifest,
        loader,
        stages,
        save_intermediate=bool(job.get("save_intermediate", False)),
        per_device_batch=int(job.get("per_device_batch") or 1),
        command_queue=command_queue,
        progress_queue=progress_queue,
    )
    loop.run()
