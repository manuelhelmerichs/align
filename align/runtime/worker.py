"""Worker entrypoint used by the align runner."""

from __future__ import annotations

import dataclasses
import queue
import shutil
import threading
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..sample_manifest import SampleManifest, SampleRecord
from .artifacts import (
    write_scales_artifact_with_checksum,
    write_transforms_artifact_with_checksum,
)
from .pipeline import SampleAlignmentResult, StagePipeline

if TYPE_CHECKING:
    from ..samples import WeightSample, WeightSampleCodec


class _ArtifactWriter:
    """Persists stage outputs to a scratch directory for the parent to commit."""

    def __init__(
        self,
        scratch_root: Path,
        *,
        sample_codec: WeightSampleCodec,
        stage_output_stages: tuple[str, ...],
    ) -> None:
        self.root = Path(scratch_root)
        self.sample_codec = sample_codec
        self.stage_output_stages = stage_output_stages
        self.root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        record: SampleRecord,
        *,
        final_sample: WeightSample,
        transforms,
        scales,
        transform_families: Mapping[str, str] | None = None,
        stage_outputs: Mapping[str, WeightSample] | None = None,
    ) -> tuple[dict[str, str | None], dict[str, int]]:
        sample_dir = self.root / f"sample_{record.index}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, str | None] = {"scratch_dir": str(sample_dir)}
        checksums: dict[str, int] = {}

        aligned_path = sample_dir / "aligned_sample.npz"
        aligned_checksum = self.sample_codec.save(aligned_path, final_sample)
        artifacts["aligned_sample_path"] = str(aligned_path)
        checksums["aligned_sample"] = aligned_checksum

        transform_path = None
        if transforms is not None:
            transform_path, transform_checksum = (
                write_transforms_artifact_with_checksum(
                    sample_dir / "transforms.npz",
                    transforms,
                    transform_families=transform_families,
                )
            )
        artifacts["transforms_path"] = str(transform_path) if transform_path else None
        checksums["transforms"] = transform_checksum if transform_path else 0

        scale_path = None
        if scales is not None:
            scale_path, scale_checksum = write_scales_artifact_with_checksum(
                sample_dir / "scales.npz", scales
            )
        artifacts["scales_path"] = str(scale_path) if scale_path else None
        checksums["scales"] = scale_checksum if scale_path else 0

        outputs = dict(stage_outputs or {})
        for stage in self.stage_output_stages:
            kind = f"stage_output_{stage}"
            stage_path = None
            if stage in outputs:
                stage_path = sample_dir / f"{kind}.npz"
                checksums[kind] = self.sample_codec.save(stage_path, outputs[stage])
            artifacts[f"{kind}_path"] = str(stage_path) if stage_path else None
            checksums.setdefault(kind, 0)

        return artifacts, checksums


class _WorkerPipelineSink:
    """Emit completed pipeline outputs as worker commit messages."""

    def __init__(self, worker: _WorkerLoop) -> None:
        self.worker = worker

    def write(self, output: SampleAlignmentResult) -> None:
        self.worker._emit_commit(
            output.record,
            output.sample,
            transforms=output.transforms,
            scales=output.scales,
            transform_families=output.transform_families,
            diagnostics=output.diagnostics,
            stage_outputs=output.stage_outputs,
        )


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
        self._pipeline = StagePipeline(
            self.stages,
            save_intermediate=self.save_intermediate,
        )

        from ..samples import create_sample_codec

        self._artifact_writer = _ArtifactWriter(
            Path(job["scratch_dir"]),
            sample_codec=create_sample_codec(
                manifest.sample_format,
                tree_path=manifest.tree_path,
            ),
            stage_output_stages=(
                tuple(name for name, _ in self.stages[:-1])
                if self.save_intermediate
                else ()
            ),
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
        self._set_active_sample(records[0].index)
        try:
            batch_size = (
                self.per_device_batch
                if self._pipeline.can_batch_match(self.per_device_batch)
                else 1
            )
            self._pipeline.run_records(
                records,
                self.loader,
                _WorkerPipelineSink(self),
                batch_size=batch_size,
                on_record_start=lambda record: self._set_active_sample(record.index),
            )
        finally:
            self._set_active_sample(None)

    def _emit_commit(
        self,
        record: SampleRecord,
        sample: WeightSample,
        *,
        transforms,
        scales,
        transform_families: Mapping[str, str] | None,
        diagnostics: dict[str, Any] | None,
        stage_outputs: Mapping[str, WeightSample],
    ) -> None:
        artifacts, checksums = self._artifact_writer.write(
            record,
            final_sample=sample,
            transforms=transforms,
            scales=scales,
            transform_families=transform_families,
            stage_outputs=stage_outputs,
        )
        payload = {
            "type": "commit",
            "sample_index": record.index,
            "sample_label": record.label,
            "artifacts": artifacts,
            "checksums": checksums,
            "diagnostics": diagnostics or {},
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
        message = {
            **payload,
            "worker_id": self.worker_id,
            "generation": self.generation,
        }
        try:
            self.progress_queue.put(message, block=block, timeout=timeout)
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
    job: dict[str, Any],
    manifest: SampleManifest,
    reference_sample: WeightSample,
    match_reference: WeightSample | None = None,
):
    from .stages import build_stage_executors

    canonicalize_cfg = job.get("canonicalize_config")
    center_softmax_head_cfg = job.get("center_softmax_head_config")
    match_cfg = job.get("match_config")
    stage_order = job.get("stages") or []
    family = job.get("family", "mlp")
    recipe_kwargs = job.get("recipe_kwargs") or {}

    return build_stage_executors(
        stage_order=stage_order,
        manifest=manifest,
        reference_sample=reference_sample,
        family=family,
        recipe_kwargs=recipe_kwargs,
        canonicalize_config=canonicalize_cfg,
        center_softmax_head_config=center_softmax_head_cfg,
        match_config=match_cfg,
        match_reference=match_reference,
        match_reference_index=(
            job["match_reference_index"]
            if "match_reference_index" in job
            else manifest.reference_index
        ),
        seed=job.get("seed"),
        rng_offset=int(job.get("rng_offset") or 0),
        batch_size=int(job.get("per_device_batch") or 1),
    )


def run_worker(job: dict[str, Any], command_queue, progress_queue) -> None:
    manifest = SampleManifest.load(Path(job["manifest_path"]))
    input_samples_dir = job.get("input_samples_dir")
    if input_samples_dir:
        input_dir = Path(input_samples_dir)
        manifest = dataclasses.replace(
            manifest,
            samples_dir=input_dir,
            tree_path=input_dir / "tree",
        )

    from .loaders import SampleLoader  # Imported after device visibility is set

    loader = SampleLoader(manifest)
    reference_sample = loader.load_reference()
    match_reference = None
    reference_path = job.get("match_reference_path")
    if reference_path:
        match_reference = loader.codec.load(Path(reference_path))
    stages = _build_stage_executors(
        job,
        manifest,
        reference_sample,
        match_reference=match_reference,
    )

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
