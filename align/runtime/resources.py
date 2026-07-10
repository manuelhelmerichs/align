"""Runtime resource limits, progress intervals, and worker timeouts."""

import os

_LOG_SAMPLE_INTERVAL = 100
_LOG_TIME_INTERVAL_SECONDS = 5.0
_STATE_SAVE_INTERVAL_SECONDS = 30.0
_WORKER_HEARTBEAT_INTERVAL = 5.0
_WORKER_HEARTBEAT_TIMEOUT = 60.0


def available_cpu_count() -> int:
    """Return CPUs visible to the current process (respects cgroup pinning)."""

    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 1
    except Exception:  # pragma: no cover - defensive
        return os.cpu_count() or 1


__all__ = [
    "_LOG_SAMPLE_INTERVAL",
    "_LOG_TIME_INTERVAL_SECONDS",
    "_STATE_SAVE_INTERVAL_SECONDS",
    "_WORKER_HEARTBEAT_INTERVAL",
    "_WORKER_HEARTBEAT_TIMEOUT",
    "available_cpu_count",
]
