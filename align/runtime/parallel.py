"""Parallel worker helpers for align execution."""

import multiprocessing as mp
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

from .common import _WORKER_HEARTBEAT_INTERVAL


def worker_process_main(job: dict[str, object], command_queue, progress_queue) -> None:
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
    ) -> None:
        self.parallelism = max(1, int(parallelism))
        self.device_ids = list(device_ids) if device_ids else None
        self.strategy_prefers_gpu = bool(strategy_prefers_gpu)

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
            base = self.device_ids or self._visible_gpu_ids()
        else:
            base = [None]

        if not base:
            base = [None]

        plan: list[WorkerConfig] = []
        for idx in range(count):
            device_id = base[idx % len(base)]
            device_type = "gpu" if device_id is not None else "cpu"
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
    ) -> WorkerState | None:
        plan = self._device_plan or self.device_plan()
        if not plan:
            return None
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
                "heartbeat_interval": _WORKER_HEARTBEAT_INTERVAL,
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
    def _visible_gpu_ids() -> list[int]:
        try:
            import jax

            return [
                int(device.id) for device in jax.devices() if device.platform == "gpu"
            ]
        except Exception:  # pragma: no cover - backend specific
            return []


__all__ = [
    "WorkerConfig",
    "WorkerState",
    "WorkerPool",
    "worker_process_main",
    "_WORKER_HEARTBEAT_INTERVAL",
]
