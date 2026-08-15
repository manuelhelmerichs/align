"""CLI for the repository benchmark harness: ``python -m benchmarks``."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from .harness import (
    _posterior_record,
    build_report,
    check_records,
    format_comparison_markdown,
    load_thresholds,
    run_objective_comparison,
    run_regression_suite,
    write_report,
)
from .posterior import (
    load_experiment_posterior_case,
    make_synthetic_layernorm_mha_transformer_posterior_case,
    make_synthetic_mlp_posterior_case,
    make_synthetic_rmsnorm_gqa_rope_transformer_posterior_case,
)
from .synthetic import default_schedule_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Run alignment benchmarks: regression suite, objective "
        "comparisons, and real-experiment posterior benchmarks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    regression = sub.add_parser(
        "regression", help="Run the fixed regression suite and check thresholds."
    )
    regression.add_argument("--output", type=Path, help="Write the JSON report here.")
    regression.add_argument(
        "--thresholds",
        type=Path,
        help="JSON thresholds file overriding the built-in defaults.",
    )
    regression.add_argument(
        "--fast", action="store_true", help="Trim seeds for a quicker run."
    )

    compare = sub.add_parser(
        "compare", help="Sweep objectives x schedules x cases (issue #4 grids)."
    )
    compare.add_argument(
        "--objective",
        action="append",
        required=True,
        metavar="NAME[=JSON]",
        help="Objective to include; repeat for several. Optional '=JSON' "
        "suffix passes objective kwargs, e.g. "
        '--objective \'diagonal_fisher={"calibration": "fisher"}\'.',
    )
    compare.add_argument(
        "--schedules",
        default="lap,sinkhorn,lap_sinkhorn_lap",
        help="Comma-separated schedule names from the default grid.",
    )
    compare.add_argument(
        "--cases",
        default="mlp,residual_conv,split_concat",
        help="Comma-separated synthetic orbit case names.",
    )
    compare.add_argument(
        "--seeds", default="0", help="Comma-separated orbit case seeds."
    )
    compare.add_argument(
        "--no-posterior",
        action="store_true",
        help="Skip the posterior benchmark rows.",
    )
    compare.add_argument(
        "--no-performance",
        action="store_true",
        help="Skip the performance benchmark rows.",
    )
    compare.add_argument(
        "--synthetic-posterior",
        choices=("mlp", "layernorm_mha_transformer", "rmsnorm_gqa_rope_transformer"),
        help="Use a named synthetic posterior instead of the default isotropic MLP.",
    )
    compare.add_argument(
        "--posterior-noise-mode",
        choices=("isotropic", "inverse_fisher"),
        default="isotropic",
    )
    compare.add_argument("--posterior-noise-scale", type=float, default=0.02)
    compare.add_argument("--posterior-seed", type=int, default=0)
    compare.add_argument("--posterior-chains", type=int, default=3)
    compare.add_argument("--posterior-samples", type=int, default=6)
    compare.add_argument("--posterior-barycenter-passes", type=int, default=2)
    compare.add_argument("--output", type=Path, help="Write the JSON report here.")
    compare.add_argument(
        "--markdown", type=Path, help="Write a markdown comparison table here."
    )
    _add_experiment_arguments(compare, required=False)

    posterior = sub.add_parser(
        "posterior",
        help="Benchmark alignment on a real SMILE/MILE experiment directory.",
    )
    _add_experiment_arguments(posterior, required=True)
    posterior.add_argument(
        "--objective", default="euclidean", help="Objective name (default euclidean)."
    )
    posterior.add_argument(
        "--objective-kwargs",
        default="{}",
        help="JSON dict of objective kwargs.",
    )
    posterior.add_argument(
        "--schedule",
        default="lap",
        help="Schedule name from the default grid, or a JSON list of steps.",
    )
    posterior.add_argument(
        "--canonicalize",
        action="store_true",
        help="Scale-canonicalize samples before matching (dense MLPs only).",
    )
    posterior.add_argument(
        "--barycenter-passes",
        type=int,
        default=2,
        help="Barycenter refinement passes: re-align against the mean of the "
        "previous pass's aligned samples (1 = plain single-reference "
        "matching; default 2 -- single-reference canonicalization is "
        "reference-dependent at moderate posterior noise).",
    )
    posterior.add_argument("--output", type=Path, help="Write the JSON report here.")

    return parser


def _add_experiment_arguments(
    parser: argparse.ArgumentParser, *, required: bool
) -> None:
    group = parser.add_argument_group("experiment posterior")
    group.add_argument("--experiment-root", type=Path, required=required)
    group.add_argument(
        "--architecture", help="Recipe name, e.g. mlp or residual_convnet."
    )
    group.add_argument(
        "--recipe-kwargs", default="{}", help="JSON dict of recipe kwargs."
    )
    group.add_argument("--samples-dir", type=Path)
    group.add_argument("--tree-path", type=Path)
    group.add_argument("--chain-indices", help="Comma-separated chain ids to include.")
    group.add_argument("--samples-per-chain", type=int)
    group.add_argument("--sample-step", type=int, default=1)
    group.add_argument("--max-total", type=int)
    group.add_argument("--reference-chain", type=int, default=0)
    group.add_argument("--reference-sample", type=int, default=0)
    group.add_argument(
        "--evaluator",
        metavar="MODULE:FUNCTION",
        help="Factory supplying function-space metrics, e.g. "
        "'benchmarks.campaigns:make_experiment_evaluator'. Reconstructing the "
        "sampled model needs the sampler that wrote the run, which lives "
        "outside align; without this flag only weight-space metrics run.",
    )
    group.add_argument(
        "--functional-split",
        choices=("train", "valid", "test"),
        default="test",
        help="Saved data split used for invariant function-space metrics.",
    )
    group.add_argument(
        "--functional-inputs",
        type=int,
        default=128,
        help="Maximum examples (or text context windows) used functionally.",
    )
    group.add_argument(
        "--functional-batch-size",
        type=int,
        help="Optional forward-pass batch size for functional metrics.",
    )


def _parse_json_dict(text: str, *, flag: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SystemExit(f"{flag} must be a JSON object, got: {text}")
    return payload


def _parse_objectives(values: list[str]) -> dict[str, dict[str, Any]]:
    objectives: dict[str, dict[str, Any]] = {}
    for value in values:
        name, _, kwargs_text = value.partition("=")
        kwargs = (
            _parse_json_dict(kwargs_text, flag="--objective") if kwargs_text else {}
        )
        objectives[name.strip()] = kwargs
    return objectives


def _parse_int_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(part) for part in text.split(",") if part.strip()]


def _resolve_evaluator(spec: str):
    """Import the ``module:function`` evaluator factory named on the CLI.

    Function-space metrics need the model and data behind a run, which only
    the sampler that wrote it can rebuild. The factory takes the experiment
    root plus ``split``/``n_inputs``/``batch_size`` and returns loader kwargs
    for :func:`load_experiment_posterior_case`.
    """

    module_name, separator, attribute = spec.partition(":")
    if not (module_name and separator and attribute):
        raise SystemExit(f"--evaluator must be 'module:function', got: {spec}")
    try:
        module = import_module(module_name)
    except ImportError as error:
        raise SystemExit(f"--evaluator module is not importable: {module_name}") from (
            error
        )
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise SystemExit(f"--evaluator {module_name} has no {attribute!r}") from error


def _load_experiment_case(args: argparse.Namespace):
    if args.experiment_root is None:
        return None
    if not args.architecture:
        raise SystemExit("--architecture is required with --experiment-root.")
    functional_kwargs = {}
    if args.evaluator:
        functional_kwargs = _resolve_evaluator(args.evaluator)(
            args.experiment_root,
            split=args.functional_split,
            n_inputs=args.functional_inputs,
            batch_size=args.functional_batch_size,
        )
    return load_experiment_posterior_case(
        args.experiment_root,
        architecture=args.architecture,
        recipe_kwargs=_parse_json_dict(args.recipe_kwargs, flag="--recipe-kwargs"),
        samples_dir=args.samples_dir,
        tree_path=args.tree_path,
        chain_indices=_parse_int_list(args.chain_indices),
        samples_per_chain=args.samples_per_chain,
        sample_step=args.sample_step,
        max_total=args.max_total,
        reference_chain=args.reference_chain,
        reference_sample=args.reference_sample,
        **functional_kwargs,
    )


def _emit(records, *, label: str, output: Path | None) -> None:
    report = build_report(records, label=label)
    if output is not None:
        write_report(report, output)
        print(f"Report written to {output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


def _run_regression(args: argparse.Namespace) -> int:
    records = run_regression_suite(fast=args.fast)
    thresholds = load_thresholds(args.thresholds) if args.thresholds else None
    violations = check_records(records, thresholds)
    _emit(records, label="regression", output=args.output)
    if violations:
        print(f"\n{len(violations)} threshold violation(s):", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print(f"\nAll {len(records)} benchmarks within thresholds.")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    grid = default_schedule_grid()
    schedule_names = [name.strip() for name in args.schedules.split(",") if name]
    unknown = sorted(set(schedule_names) - set(grid))
    if unknown:
        raise SystemExit(
            f"Unknown schedule(s) {', '.join(unknown)}; available: "
            + ", ".join(sorted(grid))
        )
    if args.synthetic_posterior is not None and args.experiment_root is not None:
        raise SystemExit(
            "--synthetic-posterior and --experiment-root are mutually exclusive."
        )
    posterior_case = _load_experiment_case(args)
    if args.synthetic_posterior is not None and not args.no_posterior:
        factories = {
            "mlp": make_synthetic_mlp_posterior_case,
            "layernorm_mha_transformer": make_synthetic_layernorm_mha_transformer_posterior_case,
            "rmsnorm_gqa_rope_transformer": make_synthetic_rmsnorm_gqa_rope_transformer_posterior_case,
        }
        posterior_case = factories[args.synthetic_posterior](
            seed=args.posterior_seed,
            n_chains=args.posterior_chains,
            n_samples=args.posterior_samples,
            noise_mode=args.posterior_noise_mode,
            noise_scale=args.posterior_noise_scale,
        )
    records = run_objective_comparison(
        objectives=_parse_objectives(args.objective),
        schedules={name: grid[name] for name in schedule_names},
        cases=[name.strip() for name in args.cases.split(",") if name],
        seeds=_parse_int_list(args.seeds) or [0],
        include_posterior=not args.no_posterior,
        posterior_case=posterior_case,
        posterior_canonicalize=args.synthetic_posterior is not None,
        posterior_barycenter_passes=args.posterior_barycenter_passes,
        include_performance=not args.no_performance,
    )
    _emit(records, label="objective-comparison", output=args.output)
    markdown = format_comparison_markdown(records)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown)
        print(f"Markdown table written to {args.markdown}")
    else:
        print("\n" + markdown)
    return 0


def _run_posterior(args: argparse.Namespace) -> int:
    case = _load_experiment_case(args)
    assert case is not None  # --experiment-root is required for this command.
    grid = default_schedule_grid()
    if args.schedule in grid:
        schedule_name, schedule = args.schedule, grid[args.schedule]
    else:
        schedule_name, schedule = "custom", json.loads(args.schedule)
    record = _posterior_record(
        case=case,
        case_label=case.name,
        schedule_name=schedule_name,
        schedule=schedule,
        objective=args.objective,
        objective_kwargs=_parse_json_dict(
            args.objective_kwargs, flag="--objective-kwargs"
        ),
        canonicalize=args.canonicalize,
        barycenter_passes=args.barycenter_passes,
    )
    _emit([record], label="experiment-posterior", output=args.output)
    return 0 if record.skipped is None else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "regression":
        return _run_regression(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "posterior":
        return _run_posterior(args)
    raise SystemExit(f"Unknown command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
