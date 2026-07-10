"""Runtime resource limits, progress intervals, and worker timeouts."""

import os

LOG_SAMPLE_INTERVAL = 100
LOG_TIME_INTERVAL_SECONDS = 5.0
STATE_SAVE_INTERVAL_SECONDS = 30.0
WORKER_HEARTBEAT_INTERVAL = 5.0
WORKER_HEARTBEAT_TIMEOUT = 60.0


def available_cpu_count() -> int:
    """Return CPUs visible to the current process (respects cgroup pinning)."""

    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 1
    except Exception:  # pragma: no cover - defensive
        return os.cpu_count() or 1


__all__ = [
    "LOG_SAMPLE_INTERVAL",
    "LOG_TIME_INTERVAL_SECONDS",
    "STATE_SAVE_INTERVAL_SECONDS",
    "WORKER_HEARTBEAT_INTERVAL",
    "WORKER_HEARTBEAT_TIMEOUT",
    "available_cpu_count",
]
