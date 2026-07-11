"""Tests for synthetic alignment benchmarks and exact-orbit cases."""

import numpy as np

from align.matching import match_sample
from benchmarks.synthetic import (
    default_schedule_grid,
    make_mlp_orbit_case,
    make_residual_conv_orbit_case,
    make_split_concat_conv_orbit_case,
    measure_matching_performance,
    run_alignment_benchmark,
    run_robustness_sweep,
)


def test_mlp_exact_orbit_benchmark_recovers_permutations_and_scales():
    case = make_mlp_orbit_case(seed=0)

    result = run_alignment_benchmark(
        case,
        schedule=[{"solver": "lap", "max_sweeps": 50, "tolerance": 0.0}],
        canonicalize=True,
    )

    metrics = result.metrics
    assert metrics.function_drift_max < 1e-5
    assert metrics.distance_after < 1e-5
    assert metrics.distance_reduction > 1.0
    assert metrics.optimality_gap is not None
    assert metrics.optimality_gap < 1e-5
    assert metrics.transform_validity_error == 0.0
    assert metrics.recovered_transform_error == 0.0
    assert metrics.residual_constraint_violations == 0


def test_residual_conv_exact_orbit_benchmark_tracks_channel_ties():
    case = make_residual_conv_orbit_case(seed=1)

    result = run_alignment_benchmark(
        case,
        schedule=[{"solver": "lap", "max_sweeps": 50, "tolerance": 0.0}],
    )

    group_order = case.graph.metadata["group_order"]
    assert len(group_order) == 2
    assert case.graph.metadata["residual_groups"] == [["core/Conv_0", "core/Conv_2"]]
    np.testing.assert_allclose(
        np.asarray(result.transforms[group_order[0]]),
        np.asarray(next(iter(case.expected_transforms.values()))),
    )

    metrics = result.metrics
    assert metrics.function_drift_max < 1e-5
    assert metrics.distance_after < metrics.distance_before
    assert metrics.optimality_gap is not None
    assert metrics.transform_validity_error == 0.0
    assert metrics.recovered_transform_error == 0.0
    assert metrics.residual_constraint_violations == 0


def test_split_concat_conv_exact_orbit_benchmark_covers_fanout_and_concat():
    case = make_split_concat_conv_orbit_case(seed=4)

    result = run_alignment_benchmark(
        case,
        schedule=[{"solver": "lap", "max_sweeps": 50, "tolerance": 0.0}],
    )

    assert case.graph.repeated_group_terms() == {}
    metrics = result.metrics
    assert metrics.function_drift_max < 1e-5
    assert metrics.distance_after < 1e-5
    assert metrics.optimality_gap is not None
    assert metrics.optimality_gap < 1e-5
    assert metrics.transform_validity_error == 0.0
    assert metrics.recovered_transform_error == 0.0
    assert metrics.residual_constraint_violations == 0


def test_split_concat_conv_sinkhorn_handles_sliced_bindings():
    case = make_split_concat_conv_orbit_case(seed=5)

    _, perms, aux = match_sample(
        case.graph,
        case.reference,
        case.target,
        schedule=[
            {
                "solver": "sinkhorn",
                "max_steps": 2,
                "sinkhorn_iterations": 10,
                "init_scale": 0.0,
            }
        ],
    )

    assert set(perms) == {"stem", "branch_a", "branch_b"}
    assert aux is not None
    assert aux["steps"][0]["solver"] == "sinkhorn"


def test_robustness_sweep_covers_all_schedule_modes():
    grid = default_schedule_grid()

    records = run_robustness_sweep(seeds=(0, 1))

    assert len(records) == 2 * len(grid)
    assert {record.schedule_name for record in records} == set(grid)
    for record in records:
        assert record.seed in {0, 1}
        assert record.case_name == f"mlp_exact_orbit_seed_{record.seed}"
        metrics = record.metrics
        assert metrics.function_drift_max < 1e-5
        assert metrics.distance_after < 1e-5
        assert metrics.optimality_gap is not None
        assert metrics.optimality_gap < 1e-5
        assert metrics.recovered_transform_error == 0.0


def test_performance_probe_reports_basic_measurements():
    case = make_mlp_orbit_case(seed=2)

    measurement = measure_matching_performance(
        case,
        schedule=[{"solver": "lap", "max_sweeps": 20, "tolerance": 0.0}],
        canonicalize=True,
        warm_repetitions=2,
    )

    assert measurement.cold_time_s >= 0.0
    assert measurement.warm_throughput_samples_s > 0.0
    assert measurement.peak_memory_bytes > 0
    assert measurement.compilation_time_s >= 0.0
