"""Repeatable benchmark harness: suites, reports, and regression thresholds.

The harness turns individual benchmark runs into named, JSON-serializable
records so results can be compared across code revisions, solver schedules,
and objectives. Three layers build on each other:

- ``run_regression_suite`` produces a fixed set of records with known-good
  expectations; ``check_records`` compares them against (fnmatch-keyed)
  thresholds, which is what the CI regression test asserts.
- ``run_objective_comparison`` sweeps objectives x schedules x cases and is
  the data source for cost-function comparisons (issue #4). Objectives that
  decline exact LAP linearization are recorded as skipped, not errors.
- ``python -m align.benchmarks`` exposes both plus real-experiment posterior
  runs on SMILE/MILE directories.
"""

from __future__ import annotations

import fnmatch
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from align.rebasin.objectives import UnsupportedGroupLinearization

from .posterior import (
    PosteriorBenchmarkCase,
    make_synthetic_mlp_posterior_case,
    make_synthetic_transformer_posterior_case,
    run_posterior_benchmark,
)
from .synthetic import (
    SyntheticOrbitCase,
    default_schedule_grid,
    make_dense_mlp_orbit_case,
    make_frn_residual_conv_orbit_case,
    make_residual_conv_orbit_case,
    make_split_concat_conv_orbit_case,
    make_transformer_orbit_case,
    make_transformer_scaled_orbit_case,
    measure_rebasin_performance,
    run_alignment_benchmark,
)

Schedule = Sequence[Mapping[str, Any]]
CaseFactory = Callable[..., SyntheticOrbitCase]

ORBIT_CASE_FACTORIES: dict[str, CaseFactory] = {
    "dense_mlp": make_dense_mlp_orbit_case,
    "residual_conv": make_residual_conv_orbit_case,
    "frn_residual_conv": make_frn_residual_conv_orbit_case,
    "split_concat": make_split_concat_conv_orbit_case,
    "transformer": make_transformer_orbit_case,
    "transformer_scaled": make_transformer_scaled_orbit_case,
}

# Cases whose scale symmetry the normalizer can remove before rebasin.
_NORMALIZABLE_CASES = frozenset(
    {"dense_mlp", "frn_residual_conv", "transformer_scaled"}
)


@dataclass
class BenchmarkRecord:
    """One named benchmark result with JSON-friendly params and metrics."""

    name: str
    kind: str  # "orbit" | "posterior" | "performance"
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    skipped: str | None = None


@dataclass(frozen=True)
class ThresholdViolation:
    """A metric that crossed its regression bound."""

    record: str
    metric: str
    bound: str
    limit: float
    value: float | None

    def __str__(self) -> str:
        return (
            f"{self.record}: {self.metric}={self.value} violates "
            f"{self.bound}={self.limit}"
        )


def _orbit_record(
    *,
    case_name: str,
    schedule_name: str,
    schedule: Schedule,
    objective: str,
    objective_kwargs: Mapping[str, Any] | None,
    normalize: bool,
    seed: int,
) -> BenchmarkRecord:
    record = BenchmarkRecord(
        name=f"orbit/{case_name}/{schedule_name}/{objective}/seed{seed}",
        kind="orbit",
        params={
            "case": case_name,
            "schedule": schedule_name,
            "schedule_steps": [dict(step) for step in schedule],
            "objective": objective,
            "objective_kwargs": dict(objective_kwargs or {}),
            "normalize": normalize,
            "seed": seed,
        },
    )
    case = ORBIT_CASE_FACTORIES[case_name](seed=seed)
    try:
        result = run_alignment_benchmark(
            case,
            objective=objective,
            objective_kwargs=objective_kwargs,
            schedule=schedule,
            normalize=normalize,
            rng_seed=seed,
        )
    except (UnsupportedGroupLinearization, NotImplementedError) as exc:
        record.skipped = str(exc)
        return record
    record.metrics = asdict(result.metrics)
    return record


def _posterior_record(
    *,
    case: PosteriorBenchmarkCase,
    case_label: str,
    schedule_name: str,
    schedule: Schedule,
    objective: str,
    objective_kwargs: Mapping[str, Any] | None,
    normalize: bool,
    rng_seed: int = 0,
) -> BenchmarkRecord:
    record = BenchmarkRecord(
        name=f"posterior/{case_label}/{schedule_name}/{objective}",
        kind="posterior",
        params={
            "case": case_label,
            "schedule": schedule_name,
            "schedule_steps": [dict(step) for step in schedule],
            "objective": objective,
            "objective_kwargs": dict(objective_kwargs or {}),
            "normalize": normalize,
            "seed": rng_seed,
        },
    )
    try:
        result = run_posterior_benchmark(
            case,
            objective=objective,
            objective_kwargs=objective_kwargs,
            schedule=schedule,
            normalize=normalize,
            rng_seed=rng_seed,
        )
    except (UnsupportedGroupLinearization, NotImplementedError) as exc:
        record.skipped = str(exc)
        return record
    metrics: dict[str, float | int | None] = {}
    for prefix, snapshot in (
        ("before", result.metrics_before),
        ("after", result.metrics_after),
    ):
        for key, value in asdict(snapshot).items():
            metrics[f"{prefix}_{key}"] = value
    metrics.update(
        {
            "function_drift_max": result.function_drift_max,
            "objective_final_mean": result.objective_final_mean,
            "wall_time_s": result.wall_time_s,
            "samples_per_s": result.samples_per_s,
        }
    )
    record.metrics = metrics
    return record


def _performance_record(
    *,
    case_name: str,
    schedule_name: str,
    schedule: Schedule,
    objective: str,
    objective_kwargs: Mapping[str, Any] | None,
    normalize: bool,
    seed: int = 0,
) -> BenchmarkRecord:
    record = BenchmarkRecord(
        name=f"perf/{case_name}/{schedule_name}/{objective}",
        kind="performance",
        params={
            "case": case_name,
            "schedule": schedule_name,
            "schedule_steps": [dict(step) for step in schedule],
            "objective": objective,
            "objective_kwargs": dict(objective_kwargs or {}),
            "normalize": normalize,
            "seed": seed,
        },
    )
    case = ORBIT_CASE_FACTORIES[case_name](seed=seed)
    try:
        measurement = measure_rebasin_performance(
            case,
            objective=objective,
            objective_kwargs=objective_kwargs,
            schedule=schedule,
            normalize=normalize,
            rng_seed=seed,
        )
    except (UnsupportedGroupLinearization, NotImplementedError) as exc:
        record.skipped = str(exc)
        return record
    record.metrics = asdict(measurement)
    return record


def run_regression_suite(*, fast: bool = False) -> list[BenchmarkRecord]:
    """Run the fixed regression benchmark suite.

    ``fast`` trims seeds so the suite stays test-suite friendly; the
    schedules and thresholds are identical in both modes.
    """

    schedules = default_schedule_grid()
    orbit_seeds = (0,) if fast else (0, 1)
    records: list[BenchmarkRecord] = []

    for seed in orbit_seeds:
        for schedule_name in ("lap", "sinkhorn", "lap_sinkhorn_lap"):
            records.append(
                _orbit_record(
                    case_name="dense_mlp",
                    schedule_name=schedule_name,
                    schedule=schedules[schedule_name],
                    objective="l2_weight",
                    objective_kwargs=None,
                    normalize=True,
                    seed=seed,
                )
            )
    records.append(
        _orbit_record(
            case_name="residual_conv",
            schedule_name="lap",
            schedule=schedules["lap"],
            objective="l2_weight",
            objective_kwargs=None,
            normalize=False,
            seed=0,
        )
    )
    # The FRN case's residual cycle couples its groups; plain LAP coordinate
    # descent can stall on some seeds, so the regression pin uses the mixed
    # schedule.
    for seed in orbit_seeds:
        records.append(
            _orbit_record(
                case_name="frn_residual_conv",
                schedule_name="lap_sinkhorn_lap",
                schedule=schedules["lap_sinkhorn_lap"],
                objective="l2_weight",
                objective_kwargs=None,
                normalize=True,
                seed=seed,
            )
        )
    for schedule_name in ("lap", "sinkhorn"):
        records.append(
            _orbit_record(
                case_name="split_concat",
                schedule_name=schedule_name,
                schedule=schedules[schedule_name],
                objective="l2_weight",
                objective_kwargs=None,
                normalize=False,
                seed=0,
            )
        )
    for seed in orbit_seeds:
        records.append(
            _orbit_record(
                case_name="transformer",
                schedule_name="lap",
                schedule=schedules["lap"],
                objective="l2_weight",
                objective_kwargs=None,
                normalize=False,
                seed=seed,
            )
        )
    for seed in orbit_seeds:
        records.append(
            _orbit_record(
                case_name="transformer_scaled",
                schedule_name="lap",
                schedule=schedules["lap"],
                objective="l2_weight",
                objective_kwargs=None,
                normalize=True,
                seed=seed,
            )
        )

    posterior_case = make_synthetic_mlp_posterior_case(
        seed=0, n_chains=3 if fast else 4, n_samples=6 if fast else 8
    )
    records.append(
        _posterior_record(
            case=posterior_case,
            case_label="synthetic_mlp",
            schedule_name="lap",
            schedule=schedules["lap"],
            objective="l2_weight",
            objective_kwargs=None,
            normalize=True,
        )
    )
    # Laplace-shaped (inverse-Fisher) noise: the regime where the Fisher
    # metric beats plain L2. The after-gap threshold of 0.2 encodes that
    # advantage: l2_weight scores ~0.36 on the full-mode cell.
    anisotropic_case = make_synthetic_mlp_posterior_case(
        seed=0,
        noise_mode="inverse_fisher",
        noise_scale=0.3,
        n_chains=3 if fast else 4,
        n_samples=6 if fast else 8,
    )
    records.append(
        _posterior_record(
            case=anisotropic_case,
            case_label="synthetic_mlp_anisotropic",
            schedule_name="lap",
            schedule=schedules["lap"],
            objective="fisher_l2",
            objective_kwargs={"calibration": "fisher"},
            normalize=True,
        )
    )
    transformer_posterior_case = make_synthetic_transformer_posterior_case(
        seed=0, n_chains=2 if fast else 3, n_samples=6 if fast else 8
    )
    records.append(
        _posterior_record(
            case=transformer_posterior_case,
            case_label="synthetic_transformer",
            schedule_name="lap",
            schedule=schedules["lap"],
            objective="l2_weight",
            objective_kwargs=None,
            normalize=True,
        )
    )

    records.append(
        _performance_record(
            case_name="dense_mlp",
            schedule_name="lap",
            schedule=schedules["lap"],
            objective="l2_weight",
            objective_kwargs=None,
            normalize=True,
        )
    )
    return records


# Regression bounds for the suite above. Keys are fnmatch patterns over record
# names; each metric maps to {"max": ...} and/or {"min": ...}. Performance
# metrics are reported but unbounded because they are machine-dependent.
DEFAULT_THRESHOLDS: dict[str, dict[str, dict[str, float]]] = {
    "orbit/*": {
        "function_drift_max": {"max": 1e-4},
        "permutation_validity_error": {"max": 0.0},
        "residual_constraint_violations": {"max": 0.0},
    },
    "orbit/dense_mlp/*": {
        "recovered_permutation_error": {"max": 0.0},
        "optimality_gap": {"max": 1e-4},
    },
    "orbit/residual_conv/*": {
        "recovered_permutation_error": {"max": 0.0},
        "optimality_gap": {"max": 1e-4},
    },
    "orbit/frn_residual_conv/*": {
        "recovered_permutation_error": {"max": 0.0},
        "optimality_gap": {"max": 1e-4},
    },
    "orbit/split_concat/lap/*": {
        "recovered_permutation_error": {"max": 0.0},
        "optimality_gap": {"max": 1e-4},
    },
    "orbit/split_concat/sinkhorn/*": {
        "optimality_gap": {"max": 1e-4},
    },
    "orbit/transformer/*": {
        "recovered_permutation_error": {"max": 0.0},
        "optimality_gap": {"max": 1e-4},
    },
    "orbit/transformer_scaled/*": {
        "recovered_permutation_error": {"max": 0.0},
        "optimality_gap": {"max": 1e-4},
    },
    "posterior/synthetic_mlp/*": {
        "before_split_rhat_max": {"min": 3.0},
        "after_split_rhat_max": {"max": 2.0},
        "before_weight_averaging_gap": {"min": 0.2},
        "after_weight_averaging_gap": {"max": 0.05},
        "function_drift_max": {"max": 1e-4},
    },
    "posterior/synthetic_mlp_anisotropic/*": {
        "before_split_rhat_max": {"min": 10.0},
        "after_split_rhat_max": {"max": 2.0},
        "before_weight_averaging_gap": {"min": 0.5},
        "after_weight_averaging_gap": {"max": 0.2},
        "function_drift_max": {"max": 1e-4},
    },
    # The transformer case has ~1.2k parameters, so the *max* split R-hat is
    # dominated by order statistics of the noise floor; the mean is the
    # collapse signal (goes from >10 to ~1 only when circuit scales are
    # normalized before rebasin).
    "posterior/synthetic_transformer/*": {
        "before_split_rhat_mean": {"min": 10.0},
        "after_split_rhat_mean": {"max": 1.2},
        "before_weight_averaging_gap": {"min": 0.2},
        "after_weight_averaging_gap": {"max": 0.15},
        "function_drift_max": {"max": 1e-3},
    },
}


def check_records(
    records: Sequence[BenchmarkRecord],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
) -> list[ThresholdViolation]:
    """Return every threshold violation in ``records`` (empty means pass)."""

    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    violations: list[ThresholdViolation] = []
    for record in records:
        matching = [
            bounds
            for pattern, bounds in thresholds.items()
            if fnmatch.fnmatch(record.name, pattern)
        ]
        if not matching:
            continue
        if record.skipped is not None:
            violations.append(
                ThresholdViolation(
                    record=record.name,
                    metric="<skipped>",
                    bound="run",
                    limit=float("nan"),
                    value=None,
                )
            )
            continue
        for bounds in matching:
            for metric, bound in bounds.items():
                value = record.metrics.get(metric)
                if value is None:
                    violations.append(
                        ThresholdViolation(
                            record=record.name,
                            metric=metric,
                            bound="present",
                            limit=float("nan"),
                            value=None,
                        )
                    )
                    continue
                if "max" in bound and value > bound["max"]:
                    violations.append(
                        ThresholdViolation(
                            record=record.name,
                            metric=metric,
                            bound="max",
                            limit=bound["max"],
                            value=float(value),
                        )
                    )
                if "min" in bound and value < bound["min"]:
                    violations.append(
                        ThresholdViolation(
                            record=record.name,
                            metric=metric,
                            bound="min",
                            limit=bound["min"],
                            value=float(value),
                        )
                    )
    return violations


def run_objective_comparison(
    *,
    objectives: Mapping[str, Mapping[str, Any]],
    schedules: Mapping[str, Schedule] | None = None,
    cases: Sequence[str] = ("dense_mlp", "residual_conv", "split_concat"),
    seeds: Sequence[int] = (0,),
    include_posterior: bool = True,
    posterior_case: PosteriorBenchmarkCase | None = None,
    include_performance: bool = True,
) -> list[BenchmarkRecord]:
    """Sweep objectives x schedules x cases for cost-function comparisons.

    ``objectives`` maps objective names to their kwargs. Objectives that
    cannot linearize a group for LAP are recorded as skipped for that cell so
    Sinkhorn-only objectives still produce a complete comparison grid.
    """

    schedules = schedules or default_schedule_grid()
    records: list[BenchmarkRecord] = []
    for objective, objective_kwargs in objectives.items():
        for schedule_name, schedule in schedules.items():
            for case_name in cases:
                normalize = case_name in _NORMALIZABLE_CASES
                for seed in seeds:
                    records.append(
                        _orbit_record(
                            case_name=case_name,
                            schedule_name=schedule_name,
                            schedule=schedule,
                            objective=objective,
                            objective_kwargs=objective_kwargs,
                            normalize=normalize,
                            seed=seed,
                        )
                    )
                if include_performance:
                    records.append(
                        _performance_record(
                            case_name=case_name,
                            schedule_name=schedule_name,
                            schedule=schedule,
                            objective=objective,
                            objective_kwargs=objective_kwargs,
                            normalize=normalize,
                        )
                    )
            if include_posterior:
                case = posterior_case or make_synthetic_mlp_posterior_case(seed=0)
                records.append(
                    _posterior_record(
                        case=case,
                        case_label=case.name
                        if posterior_case is not None
                        else "synthetic_mlp",
                        schedule_name=schedule_name,
                        schedule=schedule,
                        objective=objective,
                        objective_kwargs=objective_kwargs,
                        normalize=posterior_case is None,
                    )
                )
    return records


_MARKDOWN_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "orbit": (
        ("function_drift_max", "drift"),
        ("distance_after", "dist after"),
        ("optimality_gap", "gap"),
        ("recovered_permutation_error", "perm err"),
    ),
    "posterior": (
        ("before_split_rhat_max", "rhat before"),
        ("after_split_rhat_max", "rhat after"),
        ("before_weight_averaging_gap", "avg gap before"),
        ("after_weight_averaging_gap", "avg gap after"),
        ("samples_per_s", "samples/s"),
    ),
    "performance": (
        ("cold_time_s", "cold s"),
        ("warm_throughput_samples_s", "warm samples/s"),
        ("peak_memory_bytes", "peak bytes"),
    ),
}


def _format_cell(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4g}"


def format_comparison_markdown(records: Sequence[BenchmarkRecord]) -> str:
    """Render comparison records as one markdown table per benchmark kind."""

    sections: list[str] = []
    for kind, columns in _MARKDOWN_COLUMNS.items():
        rows = [record for record in records if record.kind == kind]
        if not rows:
            continue
        header = (
            "| benchmark | objective | "
            + " | ".join(label for _, label in columns)
            + " |"
        )
        divider = "|" + " --- |" * (len(columns) + 2)
        lines = [f"## {kind}", "", header, divider]
        for record in rows:
            label = "/".join(
                str(record.params.get(key, "?")) for key in ("case", "schedule")
            )
            if record.params.get("seed") is not None and kind == "orbit":
                label += f"/seed{record.params['seed']}"
            if record.skipped is not None:
                cells = ["skipped"] * len(columns)
            else:
                cells = [_format_cell(record.metrics.get(key)) for key, _ in columns]
            objective = str(record.params.get("objective", "?"))
            lines.append(f"| {label} | {objective} | " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def build_report(
    records: Sequence[BenchmarkRecord], *, label: str | None = None
) -> dict[str, Any]:
    """Assemble a JSON-serializable report with environment provenance."""

    import jax

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "label": label,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
        },
        "records": [asdict(record) for record in records],
    }


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def load_thresholds(path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Thresholds file {path} must contain a JSON object.")
    return payload


__all__ = [
    "BenchmarkRecord",
    "DEFAULT_THRESHOLDS",
    "ORBIT_CASE_FACTORIES",
    "ThresholdViolation",
    "build_report",
    "check_records",
    "format_comparison_markdown",
    "load_thresholds",
    "run_objective_comparison",
    "run_regression_suite",
    "write_report",
]
