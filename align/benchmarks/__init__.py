"""Reusable benchmark helpers for synthetic alignment problems."""

from .synthetic import (
    AlignmentBenchmarkMetrics,
    AlignmentBenchmarkResult,
    PerformanceMeasurement,
    SyntheticOrbitCase,
    dense_mlp_apply,
    make_dense_mlp_orbit_case,
    make_residual_conv_orbit_case,
    measure_rebasin_performance,
    permutation_matrix,
    run_alignment_benchmark,
    run_lap_robustness_sweep,
)

__all__ = [
    "AlignmentBenchmarkMetrics",
    "AlignmentBenchmarkResult",
    "PerformanceMeasurement",
    "SyntheticOrbitCase",
    "dense_mlp_apply",
    "make_dense_mlp_orbit_case",
    "make_residual_conv_orbit_case",
    "measure_rebasin_performance",
    "permutation_matrix",
    "run_alignment_benchmark",
    "run_lap_robustness_sweep",
]
