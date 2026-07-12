"""Tests for worker-pool device planning and GPU assignment policy."""

import pytest

from align.runtime.parallel import WorkerPool


def test_device_plan_cpu_strategy():
    pool = WorkerPool(parallelism=3, device_ids=None, strategy_prefers_gpu=False)
    plan = pool.device_plan(2)
    assert len(plan) == 2
    assert all(cfg.device_id is None for cfg in plan)
    assert all(cfg.device_type == "cpu" for cfg in plan)


def test_device_plan_gpu_with_explicit_ids():
    pool = WorkerPool(
        parallelism=3,
        device_ids=[0, 2],
        strategy_prefers_gpu=True,
        allow_device_sharing=True,
    )
    plan = pool.device_plan(3)
    assert [cfg.device_id for cfg in plan] == [0, 2, 0]
    assert all(cfg.device_type == "gpu" for cfg in plan)


def test_device_plan_gpu_autodetect(monkeypatch):
    pool = WorkerPool(
        parallelism=2,
        device_ids=None,
        strategy_prefers_gpu=True,
        allow_device_sharing=True,
    )
    monkeypatch.setattr(pool, "_visible_gpu_devices", lambda: [(5, "gpu")])
    plan = pool.device_plan(2)
    assert [cfg.device_id for cfg in plan] == [5, 5]
    assert all(cfg.device_type == "gpu" for cfg in plan)


def test_device_plan_rejects_implicit_accelerator_sharing(monkeypatch):
    pool = WorkerPool(parallelism=2, device_ids=None, strategy_prefers_gpu=True)
    monkeypatch.setattr(pool, "_visible_gpu_devices", lambda: [(0, "gpu")])
    with pytest.raises(ValueError, match="allow_device_sharing"):
        pool.device_plan(2)
