"""Parallel worker helpers for align execution."""

import logging
import multiprocessing as mp
import os
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

from .._jax_platforms import is_gpu_platform
from .resources import WORKER_HEARTBEAT_INTERVAL

_LOG = logging.getLogger(__name__)


def worker_process_main(job: dict[str, object], command_queue, progress_queue) -> None:
    device_id = job.get("device_id")
    device_type = str(job.get("device_type") or "cpu")
    if device_id is not None and device_type in {"cuda", "gpu", "rocm"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        os.environ["JAX_VISIBLE_DEVICES"] = "0"
    if device_type == "mps" and job.get("mps_async_dispatch") is not None:
        os.environ["JAX_MPS_ASYNC_DISPATCH"] = "1" if job["mps_async_dispatch"] else "0"
    worker_threads = str(int(job.get("worker_threads") or 1))
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = worker_threads
    from .worker import run_worker

    run_worker(job, command_queue, progress_queue)


@dataclass
class WorkerConfig:
    worker_id: int
    device_id: int | None
    device_type: str


@dataclass
class WorkerState:
    worker_id: int
    process: mp.Process
    command_queue: mp.Queue
    scratch_dir: Path
    device_id: int | None
    generation: int
    device_type: str | None = None
    assigned: deque[int] = field(default_factory=deque)
    ready: bool = False
    stopping: bool = False
    last_heartbeat: float = field(default_factory=lambda: time.time())


class WorkerPool:
    """Manage spawning and lifecycle for align worker processes."""

    def __init__(
        self,
        *,
        parallelism: int,
        device_ids: list[int] | None,
        strategy_prefers_gpu: bool,
        allow_device_sharing: bool = False,
    ) -> None:
        self.parallelism = max(1, int(parallelism))
        self.device_ids = list(device_ids) if device_ids else None
        self.strategy_prefers_gpu = bool(strategy_prefers_gpu)
        self.allow_device_sharing = bool(allow_device_sharing)

        self._ctx = mp.get_context("spawn")
        self._progress_queue = None
        self._states: dict[int, WorkerState] = {}
        self._generations: dict[int, int] = {}
        self._device_plan: list[WorkerConfig] | None = None

    @property
    def states(self) -> dict[int, WorkerState]:
        return self._states

    @property
    def progress_queue(self):
        return self._ensure_progress_queue()

    def device_plan(self, requested: int | None = None) -> list[WorkerConfig]:
        count = requested if requested is not None else self.parallelism
        if count <= 0:
            return []

        if self.strategy_prefers_gpu:
            visible = self._visible_gpu_devices()
            if self.device_ids:
                requested_ids = set(self.device_ids)
                base = [device for device in visible if device[0] in requested_ids]
                known_ids = {device[0] for device in base}
                base.extend(
                    (device_id, "gpu")
                    for device_id in self.device_ids
                    if device_id not in known_ids
                )
            else:
                base = visible
        else:
            base = [(None, "cpu")]

        if not base:
            base = [(None, "cpu")]
        elif (
            self.strategy_prefers_gpu
            and count > len(base)
            and not self.allow_device_sharing
        ):
            raise ValueError(
                f"Requested {count} accelerator workers for {len(base)} visible "
                "device(s). Set runtime.allow_device_sharing=true only if the "
                "resulting compilation, memory, and contention risk is intentional."
            )

        plan: list[WorkerConfig] = []
        for idx in range(count):
            device_id, device_type = base[idx % len(base)]
            plan.append(
                WorkerConfig(
                    worker_id=idx, device_id=device_id, device_type=device_type
                )
            )
        self._device_plan = plan
        return plan

    def start(
        self,
        *,
        worker_count: int,
        job_template: dict[str, Any],
        scratch_root: Path,
    ) -> None:
        plan = self._device_plan or self.device_plan(worker_count)
        if not plan:
            return
        requested = min(worker_count, len(plan))
        self._ensure_progress_queue()
        for worker_id in range(requested):
            config = plan[worker_id % len(plan)]
            self._spawn_worker(worker_id, config, job_template, scratch_root)

    def respawn(
        self,
        worker_id: int,
        job_template: dict[str, Any],
        scratch_root: Path,
    ) -> WorkerState:
        plan = self._device_plan or self.device_plan()
        config = plan[worker_id % len(plan)]
        return self._spawn_worker(worker_id, config, job_template, scratch_root)

    def next_message(self, timeout: float = 1.0):
        if self._progress_queue is None:
            return None
        try:
            return self._progress_queue.get(timeout=timeout)
        except Empty:
            return None

    def shutdown(self, *, force: bool = False) -> None:
        for state in self._states.values():
            if force:
                if state.process.is_alive():
                    state.process.terminate()
                continue
            if not state.stopping:
                try:
                    state.command_queue.put({"type": "stop"})
                    state.stopping = True
                except Exception:
                    pass

        for state in self._states.values():
            try:
                state.process.join(timeout=10)
            except Exception:
                pass
            if state.process.is_alive():
                state.process.terminate()
                state.process.join(timeout=5)
            shutil.rmtree(state.scratch_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _spawn_worker(
        self,
        worker_id: int,
        config: WorkerConfig,
        job_template: dict[str, Any],
        scratch_root: Path,
    ) -> WorkerState:
        queue = self._ensure_progress_queue()
        command_queue = self._ctx.Queue()
        scratch_dir = Path(scratch_root) / f"worker_{worker_id}"
        shutil.rmtree(scratch_dir, ignore_errors=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        generation = self._generations.get(worker_id, 0) + 1
        self._generations[worker_id] = generation
        job = dict(job_template)
        job.update(
            {
                "worker_id": worker_id,
                "generation": generation,
                "device_id": config.device_id,
                "device_type": config.device_type,
                "scratch_dir": str(scratch_dir),
                "heartbeat_interval": WORKER_HEARTBEAT_INTERVAL,
            }
        )
        proc = self._ctx.Process(
            target=worker_process_main, args=(job, command_queue, queue)
        )
        proc.start()
        state = WorkerState(
            worker_id=worker_id,
            process=proc,
            command_queue=command_queue,
            scratch_dir=scratch_dir,
            device_id=config.device_id,
            generation=generation,
            device_type=config.device_type,
        )
        self._states[worker_id] = state
        return state

    def _ensure_progress_queue(self):
        if self._progress_queue is None:
            self._progress_queue = self._ctx.Queue()
        return self._progress_queue

    @staticmethod
    def _visible_gpu_devices() -> list[tuple[int, str]]:
        try:
            import jax

            return [
                (int(device.id), str(device.platform))
                for device in jax.devices()
                if is_gpu_platform(device.platform)
            ]
        except Exception as exc:  # pragma: no cover - backend specific
            _LOG.warning("GPU device discovery failed: %s", exc)
            return []


__all__ = [
    "WorkerConfig",
    "WorkerState",
    "WorkerPool",
    "worker_process_main",
]
