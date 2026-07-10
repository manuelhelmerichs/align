"""Align CLI."""

import argparse
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ._jax_platforms import configure_jax_platforms, is_gpu_platform
from .config import (
    RunConfig,
    load_align_config,
    resolve_recipe_defaults,
    validate_method,
    validate_paths,
    validate_ref_sample,
)
from .config._utils import _parse_int_list
from .logging_utils import configure_logging
from .state import RunState, SampleManifest, compute_config_digest

_STATE_MANIFEST = "sample_manifest.json"
_STATE_RUN = "align_run_state.json"
_LOG = logging.getLogger("align")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="align",
        description="Align posterior samples via canonicalization and/or matching.",
    )
    parser.add_argument("config", type=Path, help="Path to YAML configuration file.")
    parser.add_argument(
        "--experiment-root", type=Path, help="Override experiment root."
    )
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument(
        "--architecture",
        help="Override the architecture recipe family "
        "(e.g., mlp, convnet, residual_convnet).",
    )

    parser.add_argument(
        "--canonicalize-only",
        action="store_true",
        help="Run only the canonicalize stage, skip matching.",
    )
    parser.add_argument(
        "--match-only",
        action="store_true",
        help="Run only the match stage, skip canonicalization.",
    )
    parser.add_argument(
        "--pipeline",
        help="Override the pipeline stages, e.g. 'canonicalize,match'.",
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-samples", action="store_true")
    parser.add_argument(
        "--describe-symmetry",
        action="store_true",
        help="Describe the weight-space symmetry derived from the reference sample.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results between stages.",
    )
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument("--parallelism", type=int)
    parser.add_argument(
        "--device-ids", help="Comma-separated device IDs for rebasin batching."
    )
    parser.add_argument("--per-device-batch", type=int)
    parser.add_argument("--verbosity", choices=["quiet", "info", "debug"])
    parser.add_argument(
        "--trust-artifacts-on-resume",
        action="store_true",
        help="Skip checksum validation when resuming (faster, less safe).",
    )
    parser.add_argument(
        "--ignore-config-mismatch",
        action="store_true",
        help="Bypass config digest checks when resuming (use with caution).",
    )
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    if args.experiment_root or args.output_dir:
        overrides["paths"] = {}
        if args.experiment_root:
            overrides["paths"]["experiment_root"] = args.experiment_root
        if args.output_dir:
            overrides["paths"]["output_dir"] = args.output_dir

    if args.architecture:
        overrides["architecture"] = {"family": args.architecture}

    if args.pipeline:
        overrides["pipeline"] = [
            stage.strip() for stage in args.pipeline.split(",") if stage.strip()
        ]
    if args.canonicalize_only:
        overrides["pipeline"] = ["canonicalize"]
    if args.match_only:
        overrides["pipeline"] = ["match"]

    runtime = overrides.setdefault("runtime", {})
    if args.resume:
        runtime["resume"] = True
    if args.dry_run:
        runtime["dry_run"] = True
    if args.list_samples:
        runtime["list_samples"] = True
    if args.describe_symmetry:
        runtime["describe_symmetry"] = True
    if args.validate_only:
        runtime["validate_only"] = True
    if args.save_intermediate:
        runtime["save_intermediate"] = True
    if args.force_cpu:
        runtime["force_cpu"] = True
    if args.force_gpu:
        runtime["force_gpu"] = True
    if args.parallelism is not None:
        runtime["parallelism"] = args.parallelism
    if args.device_ids:
        runtime["device_ids"] = _parse_int_list(args.device_ids)
    if args.per_device_batch is not None:
        runtime["per_device_batch"] = args.per_device_batch
    if args.verbosity:
        runtime["verbosity"] = args.verbosity
    if args.trust_artifacts_on_resume:
        runtime["validate_artifacts_on_resume"] = False

    return overrides


def run(args: argparse.Namespace) -> None:
    config = load_align_config(args.config, _cli_overrides(args))
    stages = config.active_stages()
    _configure_platform_preferences(config)

    logger = configure_logging(config.runtime.verbosity)
    validate_paths(config)
    exp_root = config.paths.experiment_root
    if exp_root is None:
        raise ValueError("experiment_root must be provided via CLI or config file.")
    exp_root = exp_root.resolve()
    config.paths.experiment_root = exp_root
    samples_dir = (config.paths.samples_dir or (exp_root / "samples")).resolve()
    sample_format = config.paths.sample_format
    explicit_tree_path = config.paths.tree_path is not None
    tree_path = (config.paths.tree_path or _detect_tree_path(exp_root)).resolve()
    if explicit_tree_path:
        if not tree_path.exists():
            raise FileNotFoundError(f"Tree path not found: {tree_path}")
        if not tree_path.is_file():
            raise FileNotFoundError(f"Tree path must be a file: {tree_path}")
    output_dir = (config.paths.output_dir or _default_output_dir(exp_root)).resolve()
    config.paths.output_dir = output_dir
    config.paths.samples_dir = samples_dir
    config.paths.tree_path = tree_path
    resolve_recipe_defaults(config)

    if output_dir.exists() and not config.runtime.resume:
        raise FileExistsError(
            "Output directory already exists. Pass --resume to continue the run."
        )

    state_dir = output_dir / "state"
    manifest_path = state_dir / _STATE_MANIFEST
    manifest = _load_or_build_manifest(
        manifest_path,
        samples_dir,
        tree_path,
        selection=config.selection,
        sample_format=sample_format,
        resume=config.runtime.resume,
    )
    validate_ref_sample(manifest, config.selection)

    validate_method(config)

    config_payload = config.as_dict()
    config_payload["resolved_paths"] = {
        "experiment_root": str(exp_root),
        "samples_dir": str(samples_dir),
        "tree_path": str(tree_path),
        "output_dir": str(output_dir),
        "sample_format": sample_format,
    }
    config_payload["stages"] = stages
    if args.print_config:
        print(json.dumps(config_payload, indent=2))
        if not (
            config.runtime.dry_run
            or config.runtime.list_samples
            or config.runtime.describe_symmetry
            or config.runtime.validate_only
        ):
            return

    digest_payload = _digest_payload(config_payload)
    digest = compute_config_digest(digest_payload)
    runtime_settings = _runtime_settings_snapshot(config.runtime)

    run_state_path = state_dir / _STATE_RUN
    run_manifest = _load_or_create_run_manifest(
        run_state_path,
        run_exists=config.runtime.resume,
        manifest=manifest,
        output_dir=output_dir,
        exp_root=exp_root,
        digest=digest,
        config=config,
        manifest_path=manifest_path,
        runtime_settings=runtime_settings,
        ignore_digest=args.ignore_config_mismatch,
    )

    logger.info("Stages: %s", ", ".join(stages))
    logger.info("%s", _format_manifest_summary(manifest))

    if config.runtime.list_samples:
        for record in manifest.records:
            print(f"{record.index:05d} | {record.label} | {record.relative_path}")
        return

    if config.runtime.describe_symmetry:
        print(_describe_symmetry(config, manifest))
        return

    if config.runtime.validate_only:
        print("Validation successful. No alignment executed.")
        return

    from .runtime import AlignmentRunner, RunArtifactStore

    align_logger = RunArtifactStore(
        manifest=manifest,
        output_dir=output_dir,
        stages=stages,
        save_intermediate=config.runtime.save_intermediate,
    )
    runner = AlignmentRunner(
        config=config,
        manifest=manifest,
        run_manifest=run_manifest,
        logger=align_logger,
        progress_logger=logger,
    )

    if config.runtime.dry_run:
        runner.execute(dry_run=True)
        print("Dry run complete. Inspect state directory for manifest summary.")
        return

    runner.execute()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


def _describe_symmetry(config: RunConfig, manifest: SampleManifest) -> str:
    """Describe the weight-space symmetry for the reference sample."""

    from .architectures import get_recipe
    from .runtime.loaders import SampleLoader
    from .symmetry import format_symmetry_description

    recipe = get_recipe(config.architecture.family, **config.architecture.recipe_kwargs)
    ref_sample = SampleLoader(manifest).load_reference()
    problem = recipe.build_problem(ref_sample.params)
    return format_symmetry_description(problem)


def _configure_platform_preferences(config: RunConfig) -> None:
    runtime_cfg = config.runtime
    force_cpu = bool(runtime_cfg.force_cpu)
    force_gpu = bool(runtime_cfg.force_gpu)
    if force_cpu and force_gpu:
        raise ValueError("--force-cpu and --force-gpu are mutually exclusive.")

    preference = "cpu"
    if force_gpu:
        preference = "gpu"
    elif config.match is not None:
        if any(step.solver == "sinkhorn" for step in config.match.solvers):
            preference = "gpu"
    applied_preference = configure_jax_platforms(
        preference=preference, allow_preallocation=False
    )
    import jax

    log = logging.getLogger("align")
    if applied_preference == "cpu":
        jax.config.update("jax_platform_name", "cpu")
        return

    try:
        gpus = [dev for dev in jax.devices() if is_gpu_platform(dev.platform)]
        if not gpus:
            raise RuntimeError("No GPU devices visible")
    except Exception as exc:
        log.warning("GPU backend unavailable (%s). Falling back to CPU.", exc)
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_VISIBLE_DEVICES"] = ""
        jax.config.update("jax_platform_name", "cpu")


def _format_manifest_summary(manifest) -> str:
    summary = manifest.summary()
    chains = summary.get("chains") or []
    chains_text = _format_chain_list(chains)
    reference = summary.get("reference", {})
    ref_chain = reference.get("chain", "-")
    ref_sample = reference.get("sample", "-")
    filters = summary.get("filters") or {}
    filter_text = _format_key_values(filters)
    total = summary.get("total_samples", "-")
    sample_format = summary.get("sample_format", "-")
    return (
        "Manifest | "
        f"chains={chains_text} | "
        f"total_samples={total} | "
        f"sample_format={sample_format} | "
        f"reference=chain{ref_chain}/sample{ref_sample} | "
        f"filters[{filter_text}]"
    )


def _format_chain_list(chains) -> str:
    values = list(chains)
    if not values:
        return "-"
    runs: list[tuple[int, int]] = []
    start = prev = int(values[0])
    for value in values[1:]:
        current = int(value)
        if current == prev + 1:
            prev = current
            continue
        runs.append((start, prev))
        start = prev = current
    runs.append((start, prev))

    parts: list[str] = []
    for start, end in runs:
        span = end - start + 1
        if span == 1:
            parts.append(str(start))
        elif span == 2:
            parts.append(f"{start},{end}")
        else:
            parts.append(f"{start},...,{end}")
    return ",".join(parts)


def _format_key_values(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "-"
    parts = []
    for key in sorted(payload):
        value = payload[key]
        parts.append(f"{key}={_format_scalar(value)}")
    return " ".join(parts)


def _format_scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, tuple, set)):
        if not value:
            return "-"
        return ",".join(str(item) for item in value)
    return str(value)


def _detect_tree_path(exp_root: Path) -> Path:
    for candidate in ("tree_sampling", "tree"):
        path = exp_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate 'tree_sampling' or 'tree' under the experiment root."
    )


def _default_output_dir(exp_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return exp_root / "align" / timestamp


def _digest_payload(config_payload: Mapping[str, Any]) -> dict[str, Any]:
    resolved = config_payload.get("resolved_paths", {})
    selection = dict(config_payload.get("selection", {}))
    payload: dict[str, Any] = {
        "architecture": config_payload.get("architecture"),
        "paths": {
            "experiment_root": resolved.get("experiment_root"),
            "samples_dir": resolved.get("samples_dir"),
            "tree_path": resolved.get("tree_path"),
            "sample_format": resolved.get("sample_format"),
        },
        "selection": selection,
        "pipeline": config_payload.get("pipeline"),
    }
    canonicalize = config_payload.get("canonicalize")
    if canonicalize is not None:
        payload["canonicalize"] = canonicalize
    match = config_payload.get("match")
    if match is not None:
        payload["match"] = match
    return payload


def _runtime_settings_snapshot(runtime_cfg) -> dict[str, Any]:
    payload = asdict(runtime_cfg)
    return {key: value for key, value in payload.items() if value is not None}


def _load_or_build_manifest(
    manifest_path: Path,
    samples_dir: Path,
    tree_path: Path,
    *,
    selection,
    sample_format: str,
    resume: bool,
) -> SampleManifest:
    if resume and manifest_path.exists():
        manifest = SampleManifest.load(manifest_path)
    else:
        manifest = SampleManifest.build(
            samples_dir,
            tree_path,
            chain_indices=selection.chain_indices,
            samples_per_chain=selection.samples_per_chain,
            sample_step=selection.sample_step,
            max_total=selection.max_total,
            ref_chain=selection.reference_chain,
            ref_sample=selection.reference_sample,
            sample_format=sample_format,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.save(manifest_path)
    manifest.ensure_shared_records(manifest_path)
    manifest.ensure_shared_tree(manifest_path)
    return manifest


def _load_or_create_run_manifest(
    path: Path,
    *,
    run_exists: bool,
    manifest: SampleManifest,
    output_dir: Path,
    exp_root: Path,
    digest: str,
    config: RunConfig,
    manifest_path: Path,
    runtime_settings: Mapping[str, Any],
    ignore_digest: bool = False,
) -> RunState:
    if path.exists():
        run_manifest = RunState.load(path)
        if run_manifest.config_digest != digest:
            if ignore_digest:
                _LOG.warning(
                    "Config digest mismatch ignored. Resume run behaviour may be undefined."
                )
            else:
                raise ValueError("Configuration mismatch detected for resume run.")
        manifest.validate_digest(run_manifest.manifest_digest)
        return run_manifest

    if run_exists:
        raise FileNotFoundError(
            "--resume specified but no run state found under the output directory."
        )

    run_manifest = RunState(
        path=path,
        experiment_root=exp_root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        config_digest=digest,
        manifest_digest=manifest.digest(),
        stages=config.active_stages(),
        reference={
            "chain": manifest.ref_chain,
            "sample": manifest.ref_sample,
            "index": manifest.reference_index,
        },
        filters=manifest.filters,
        total_samples=manifest.total,
        runtime_settings=runtime_settings,
        metadata={
            "architecture": config.architecture.family,
            "canonicalize_strategy": config.canonicalize.strategy
            if config.canonicalize
            else None,
            "match_objective": config.match.objective.type if config.match else None,
            "match_solvers": config.match.solvers_payload() if config.match else None,
            "match_barycenter_passes": config.match.barycenter_passes
            if config.match
            else None,
        },
    )
    run_manifest.save()
    return run_manifest


if __name__ == "__main__":  # pragma: no cover
    main()
