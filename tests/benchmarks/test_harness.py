"""Tests for the benchmark harness: regression suite, thresholds, reports."""

import json

import pytest

from align.benchmarks.harness import (
    DEFAULT_THRESHOLDS,
    BenchmarkRecord,
    build_report,
    check_records,
    format_comparison_markdown,
    load_thresholds,
    run_objective_comparison,
    run_regression_suite,
    write_report,
)
from align.benchmarks.posterior import make_synthetic_mlp_posterior_case
from align.benchmarks.synthetic import default_schedule_grid


def test_fast_regression_suite_stays_within_thresholds():
    records = run_regression_suite(fast=True)

    assert not check_records(records)
    names = {record.name for record in records}
    assert "orbit/mlp/lap/euclidean/seed0" in names
    assert "orbit/residual_conv/lap/euclidean/seed0" in names
    assert "orbit/split_concat/sinkhorn/euclidean/seed0" in names
    assert "orbit/layernorm_mha_transformer/lap/euclidean/seed0" in names
    assert "posterior/synthetic_mlp/lap/euclidean" in names
    assert "perf/mlp/lap/euclidean" in names


def _record(name, kind="orbit", params=None, metrics=None, skipped=None):
    return BenchmarkRecord(
        name=name,
        kind=kind,
        params=params or {},
        metrics=metrics or {},
        skipped=skipped,
    )


def test_check_records_flags_bounds_missing_metrics_and_skips():
    thresholds = {"orbit/*": {"gap": {"max": 1.0}, "reduction": {"min": 0.5}}}
    records = [
        _record("orbit/pass", metrics={"gap": 0.5, "reduction": 0.9}),
        _record("orbit/too_high", metrics={"gap": 1.5, "reduction": 0.9}),
        _record("orbit/too_low", metrics={"gap": 0.5, "reduction": 0.1}),
        _record("orbit/missing", metrics={"gap": 0.5}),
        _record("orbit/skipped", skipped="unsupported"),
        _record("perf/unmatched", kind="performance"),
    ]

    violations = check_records(records, thresholds)

    assert {(v.record, v.metric, v.bound) for v in violations} == {
        ("orbit/too_high", "gap", "max"),
        ("orbit/too_low", "reduction", "min"),
        ("orbit/missing", "reduction", "present"),
        ("orbit/skipped", "<skipped>", "run"),
    }


def test_objective_comparison_runs_a_minimal_grid():
    records = run_objective_comparison(
        objectives={"euclidean": {}},
        schedules={"lap": default_schedule_grid()["lap"]},
        cases=("mlp",),
        seeds=(0,),
        include_posterior=False,
        include_performance=False,
    )

    assert [record.name for record in records] == ["orbit/mlp/lap/euclidean/seed0"]
    (record,) = records
    assert record.skipped is None
    assert record.metrics["optimality_gap"] <= 1e-4


def test_inverse_fisher_posterior_grid_includes_relative_fisher_sinkhorn():
    case = make_synthetic_mlp_posterior_case(
        seed=0,
        n_chains=2,
        n_samples=4,
        noise_mode="inverse_fisher",
        noise_scale=0.4,
    )
    records = run_objective_comparison(
        objectives={
            "euclidean": {},
            "diagonal_fisher": {"calibration": "fisher"},
            "relative_fisher": {"calibration": "relative_fisher"},
        },
        schedules={
            "sinkhorn": [
                {
                    "solver": "sinkhorn",
                    "max_steps": 2,
                    "sinkhorn_iterations": 10,
                    "init_scale": 0.0,
                }
            ]
        },
        cases=(),
        include_posterior=True,
        posterior_case=case,
        include_performance=False,
    )
    assert len(records) == 3
    assert {record.params["objective"] for record in records} == {
        "euclidean",
        "diagonal_fisher",
        "relative_fisher",
    }
    assert all(record.skipped is None for record in records)


def test_format_comparison_markdown_renders_all_kinds_and_skips():
    params = {"case": "mlp", "schedule": "lap", "seed": 0, "objective": "l2"}
    records = [
        _record(
            "orbit/a",
            params=params,
            metrics={
                "function_drift_max": 0.0,
                "distance_after": 1234.5,
                "optimality_gap": 2e-06,
                "recovered_transform_error": 0.0,
            },
        ),
        _record("posterior/b", kind="posterior", params=params, skipped="nope"),
        _record(
            "perf/c",
            kind="performance",
            params=params,
            metrics={
                "cold_time_s": 0.25,
                "warm_throughput_samples_s": 12.0,
                "peak_memory_bytes": 1024,
            },
        ),
    ]

    markdown = format_comparison_markdown(records)

    assert "## orbit" in markdown
    assert "## posterior" in markdown
    assert "## performance" in markdown
    assert "| mlp/lap/seed0 | l2 | 0 | 1.23e+03 | 2.00e-06 | 0 |" in markdown
    assert "| mlp/lap | l2 | skipped | skipped |" in markdown
    assert "| mlp/lap | l2 | 0.25 | 12 | 1024 |" in markdown


def test_report_write_and_threshold_load_roundtrip(tmp_path):
    report = build_report([_record("orbit/a", metrics={"gap": 0.5})], label="unit")
    path = tmp_path / "nested" / "report.json"
    write_report(report, path)

    loaded = json.loads(path.read_text())
    assert loaded["label"] == "unit"
    assert loaded["records"][0]["name"] == "orbit/a"
    assert "jax" in loaded["environment"]

    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(json.dumps(DEFAULT_THRESHOLDS))
    assert load_thresholds(thresholds_path) == DEFAULT_THRESHOLDS

    thresholds_path.write_text("[]")
    with pytest.raises(ValueError):
        load_thresholds(thresholds_path)
