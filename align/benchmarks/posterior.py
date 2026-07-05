"""Posterior-level benchmarks: multi-chain quality metrics and real samples.

While :mod:`align.benchmarks.synthetic` scores single reference/target pairs on
exact orbits, this module benchmarks alignment across a whole posterior: many
samples organized into chains. It provides

- a synthetic multi-chain posterior with a known ground truth (one mode whose
  chains live in different symmetry copies, plus small within-chain noise),
- a loader that turns a real SMILE/MILE experiment directory into the same
  case representation, and
- chain-aware quality metrics (split R-hat, pairwise distances, the
  weight-averaging gap) evaluated before and after alignment.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import numpy as np

from align.alignment import AlignmentProblem
from align.architectures import get_adapter
from align.normalization import ScaleNormalizer, ScaleState
from align.rebasin import PermutationState, build_scheduler, rebasin_single_sample

from .synthetic import (
    ParamTree,
    _make_dense_params,
    dense_mlp_apply,
    permutation_matrix,
)

ApplyFn = Callable[[ParamTree, jax.Array], jax.Array]

ChainList = tuple[tuple[ParamTree, ...], ...]


@dataclass
class PosteriorBenchmarkCase:
    """A multi-chain posterior of parameter trees sharing one alignment problem."""

    name: str
    problem: AlignmentProblem
    chains: ChainList
    ref_chain: int = 0
    ref_sample: int = 0
    apply_fn: ApplyFn | None = None
    inputs: jax.Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference(self) -> ParamTree:
        return self.chains[self.ref_chain][self.ref_sample]

    def iter_samples(self) -> list[tuple[int, int, ParamTree]]:
        """Return ``(chain_index, sample_index, params)`` in stable order."""

        return [
            (chain_idx, sample_idx, params)
            for chain_idx, chain in enumerate(self.chains)
            for sample_idx, params in enumerate(chain)
        ]


@dataclass(frozen=True)
class PosteriorMetrics:
    """Chain-aware quality metrics for one set of posterior samples."""

    split_rhat_max: float
    split_rhat_mean: float
    pairwise_distance_mean: float
    pairwise_distance_max: float
    cross_chain_distance_mean: float
    weight_variance_total: float
    weight_averaging_gap: float | None


@dataclass(frozen=True)
class PosteriorBenchmarkResult:
    """Before/after metrics and runtime statistics for one posterior run."""

    metrics_before: PosteriorMetrics
    metrics_after: PosteriorMetrics
    function_drift_max: float | None
    objective_final_mean: float | None
    wall_time_s: float
    samples_per_s: float
    aligned_chains: ChainList


def _flatten_sample(params: ParamTree) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(params)
    return np.concatenate(
        [np.ravel(np.asarray(leaf, dtype=np.float64)) for leaf in leaves]
    )


def _stack_chains(chains: ChainList) -> np.ndarray:
    """Return draws with shape ``(n_chains, n_samples, n_params)``.

    Unequal chains are truncated to the shortest length, the standard
    convention for chain diagnostics.
    """

    rows = [[_flatten_sample(sample) for sample in chain] for chain in chains]
    shortest = min(len(chain) for chain in rows)
    return np.stack([np.stack(chain[:shortest]) for chain in rows])


def split_rhat(draws: np.ndarray) -> np.ndarray:
    """Return the split Gelman-Rubin R-hat per parameter.

    ``draws`` has shape ``(n_chains, n_samples, n_params)``. Each chain is
    split in half, so at least four samples per chain are required. Parameters
    that are constant within and across chains report R-hat 1.
    """

    if draws.ndim != 3:
        raise ValueError(f"Expected (chains, samples, params), got {draws.shape}.")
    n_chains, n_samples, _ = draws.shape
    if n_chains < 2:
        raise ValueError("R-hat requires at least two chains.")
    if n_samples < 4:
        raise ValueError("Split R-hat requires at least four samples per chain.")

    half = n_samples // 2
    split = np.concatenate(
        [draws[:, :half, :], draws[:, half : 2 * half, :]], axis=0
    ).astype(np.float64)
    m, n, _ = split.shape

    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    within = chain_vars.mean(axis=0)
    between = n * chain_means.var(axis=0, ddof=1)
    var_plus = (n - 1) / n * within + between / n

    rhat = np.ones_like(within)
    active = within > 1e-30
    rhat[active] = np.sqrt(var_plus[active] / within[active])
    rhat[~active & (between > 1e-30)] = np.inf
    return rhat


def evaluate_posterior_metrics(
    case: PosteriorBenchmarkCase, chains: ChainList
) -> PosteriorMetrics:
    """Compute chain-aware quality metrics for one set of posterior samples."""

    draws = _stack_chains(chains)
    rhat = split_rhat(draws)

    flat = draws.reshape(-1, draws.shape[-1])
    chain_ids = np.repeat(np.arange(draws.shape[0]), draws.shape[1])
    distances: list[float] = []
    cross: list[float] = []
    for i in range(flat.shape[0]):
        for j in range(i + 1, flat.shape[0]):
            dist = float(np.linalg.norm(flat[i] - flat[j]))
            distances.append(dist)
            if chain_ids[i] != chain_ids[j]:
                cross.append(dist)

    averaging_gap = None
    if case.apply_fn is not None and case.inputs is not None:
        sample_list = [sample for chain in chains for sample in chain]
        predictions = np.stack(
            [np.asarray(case.apply_fn(sample, case.inputs)) for sample in sample_list]
        )
        mean_params = jax.tree_util.tree_map(
            lambda *leaves: np.mean(np.stack([np.asarray(x) for x in leaves]), axis=0),
            *sample_list,
        )
        mean_predictions = np.asarray(case.apply_fn(mean_params, case.inputs))
        averaging_gap = float(
            np.sqrt(np.mean(np.square(mean_predictions - predictions.mean(axis=0))))
        )

    return PosteriorMetrics(
        split_rhat_max=float(np.max(rhat)),
        split_rhat_mean=float(np.mean(rhat)),
        pairwise_distance_mean=float(np.mean(distances)),
        pairwise_distance_max=float(np.max(distances)),
        cross_chain_distance_mean=float(np.mean(cross)) if cross else 0.0,
        weight_variance_total=float(np.sum(flat.var(axis=0))),
        weight_averaging_gap=averaging_gap,
    )


def make_synthetic_mlp_posterior_case(
    *,
    seed: int = 0,
    n_chains: int = 4,
    n_samples: int = 8,
    sizes: Sequence[int] = (3, 5, 4, 2),
    noise_scale: float = 0.02,
    scale_jitter: float = 0.3,
    n_inputs: int = 16,
) -> PosteriorBenchmarkCase:
    """Build a single-mode posterior whose chains sit in symmetry copies.

    Every chain applies one fixed random permutation and positive scale
    transformation of a shared base mode (function-preserving), then adds
    i.i.d. Gaussian weight noise per sample (scale ``noise_scale``). Before
    alignment, cross-chain R-hat is far above 1 because chain means differ by
    symmetry actions; after normalization plus rebasin all chains should
    collapse onto one basin and R-hat should approach 1.
    """

    if n_chains < 2:
        raise ValueError("Posterior cases require at least two chains.")
    if n_samples < 4:
        raise ValueError("Posterior cases require at least four samples per chain.")

    key = jax.random.PRNGKey(seed)
    param_key, input_key = jax.random.split(key)
    base = _make_dense_params(key=param_key, sizes=tuple(sizes))
    adapter = get_adapter("dense_mlp")
    problem = adapter.build_problem(base)

    rng = np.random.default_rng(seed)
    chains: list[tuple[ParamTree, ...]] = []
    chain_transforms: list[dict[str, Any]] = []
    for chain_idx in range(n_chains):
        if chain_idx == 0:
            transformed = base
            perms: dict[str, np.ndarray] = {}
            scales: dict[str, np.ndarray] = {}
        else:
            perms = {
                gid: permutation_matrix(rng.permutation(group.size))
                for gid, group in problem.groups.items()
            }
            scales = {
                gid: np.exp(
                    rng.uniform(-scale_jitter, scale_jitter, size=group.size)
                ).astype(np.float32)
                for gid, group in problem.groups.items()
            }
            scaled = problem.apply_scales(base, ScaleState.from_scales(problem, scales))
            transformed = problem.apply(
                scaled, PermutationState.from_perms(problem, perms)
            )
        chain_transforms.append({"perms": perms, "scales": scales})

        samples = []
        for _ in range(n_samples):
            noisy = jax.tree_util.tree_map(
                lambda leaf: (
                    np.asarray(leaf)
                    + noise_scale
                    * rng.standard_normal(np.shape(leaf)).astype(np.asarray(leaf).dtype)
                ),
                transformed,
            )
            samples.append(noisy)
        chains.append(tuple(samples))

    inputs = jax.random.normal(input_key, (n_inputs, int(sizes[0])))
    return PosteriorBenchmarkCase(
        name=f"synthetic_mlp_posterior_seed_{seed}",
        problem=problem,
        chains=tuple(chains),
        apply_fn=dense_mlp_apply,
        inputs=inputs,
        metadata={
            "seed": seed,
            "n_chains": n_chains,
            "n_samples": n_samples,
            "noise_scale": noise_scale,
            "scale_jitter": scale_jitter,
            "chain_transforms": chain_transforms,
        },
    )


def _detect_tree_path(experiment_root: Path) -> Path:
    for candidate in ("tree_sampling", "tree"):
        path = experiment_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate 'tree_sampling' or 'tree' under the experiment root."
    )


def load_experiment_posterior_case(
    experiment_root: str | Path,
    *,
    architecture: str,
    adapter_kwargs: Mapping[str, Any] | None = None,
    samples_dir: str | Path | None = None,
    tree_path: str | Path | None = None,
    chain_indices: Sequence[int] | None = None,
    samples_per_chain: int | None = None,
    sample_step: int = 1,
    max_total: int | None = None,
    ref_chain: int = 0,
    ref_sample: int = 0,
    sample_format: str = "pytree_npz",
    name: str | None = None,
) -> PosteriorBenchmarkCase:
    """Load a real SMILE/MILE experiment directory as a posterior case.

    All selected samples are materialized in memory, so restrict the selection
    (``samples_per_chain``, ``max_total``) for large posteriors. Function-space
    metrics are unavailable because real experiments carry no ``apply_fn``.
    """

    from align.runtime.loaders import SampleLoader
    from align.state import SampleManifest

    root = Path(experiment_root).resolve()
    resolved_samples_dir = Path(samples_dir) if samples_dir else root / "samples"
    resolved_tree_path = Path(tree_path) if tree_path else _detect_tree_path(root)

    manifest = SampleManifest.build(
        resolved_samples_dir,
        resolved_tree_path,
        chain_indices=chain_indices,
        samples_per_chain=samples_per_chain,
        sample_step=sample_step,
        max_total=max_total,
        ref_chain=ref_chain,
        ref_sample=ref_sample,
        sample_format=sample_format,
    )
    loader = SampleLoader(manifest)

    by_chain: dict[int, list[tuple[int, ParamTree]]] = {}
    for record in manifest.records:
        params = loader.load(record).params
        by_chain.setdefault(record.chain_id, []).append((record.sample_id, params))

    chain_ids = sorted(by_chain)
    chains = tuple(
        tuple(params for _, params in sorted(by_chain[cid], key=lambda x: x[0]))
        for cid in chain_ids
    )
    ref_chain_index = chain_ids.index(manifest.reference_record.chain_id)
    ref_sample_ids = [sid for sid, _ in sorted(by_chain[chain_ids[ref_chain_index]])]
    ref_sample_index = ref_sample_ids.index(manifest.reference_record.sample_id)

    reference = chains[ref_chain_index][ref_sample_index]
    adapter = get_adapter(architecture, **dict(adapter_kwargs or {}))
    problem = adapter.build_problem(reference)

    return PosteriorBenchmarkCase(
        name=name or f"experiment_{root.name}",
        problem=problem,
        chains=chains,
        ref_chain=ref_chain_index,
        ref_sample=ref_sample_index,
        metadata={
            "experiment_root": str(root),
            "architecture": architecture,
            "chain_ids": chain_ids,
            "total_samples": manifest.total,
        },
    )


def _normalize_tree(problem: AlignmentProblem, params: ParamTree) -> ParamTree:
    normalized, _, _ = ScaleNormalizer().normalize(
        problem,
        params,
        task_type="regression",
        normalize_biases=True,
    )
    return normalized


def run_posterior_benchmark(
    case: PosteriorBenchmarkCase,
    *,
    objective: str = "l2_weight",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[Mapping[str, Any]] | None = None,
    normalize: bool = False,
    rng_seed: int = 0,
) -> PosteriorBenchmarkResult:
    """Align every sample of a posterior case to its reference and score it.

    ``metrics_before`` is computed on the raw input chains, ``metrics_after``
    on the normalized (optional) and rebasined chains, so the comparison
    reflects the complete alignment treatment.
    """

    metrics_before = evaluate_posterior_metrics(case, case.chains)

    scheduler = build_scheduler(
        objective=objective,
        objective_kwargs=dict(objective_kwargs or {}),
        schedule=schedule,
    )
    reference = case.reference
    if normalize:
        reference = _normalize_tree(case.problem, reference)
    ref_data = case.problem.materialize(
        reference, backend=scheduler.backend, cache=True
    )

    flat_samples = case.iter_samples()
    aligned_flat: list[ParamTree] = []
    objective_values: list[float] = []
    start = time.perf_counter()
    for index, (_, _, params) in enumerate(flat_samples):
        target = _normalize_tree(case.problem, params) if normalize else params
        aligned, _, aux = rebasin_single_sample(
            case.problem,
            reference,
            target,
            scheduler=scheduler,
            ref_data=ref_data,
            rng_key=jax.random.fold_in(jax.random.PRNGKey(rng_seed), index),
        )
        aligned_flat.append(aligned)
        if aux is not None and "objective_final" in aux:
            objective_values.append(float(aux["objective_final"]))
    wall_time = time.perf_counter() - start

    aligned_chains: list[list[ParamTree]] = [[] for _ in case.chains]
    for (chain_idx, _, _), aligned in zip(flat_samples, aligned_flat, strict=True):
        aligned_chains[chain_idx].append(aligned)
    aligned_tuple: ChainList = tuple(tuple(chain) for chain in aligned_chains)

    metrics_after = evaluate_posterior_metrics(case, aligned_tuple)

    function_drift = None
    if case.apply_fn is not None and case.inputs is not None:
        drift = 0.0
        for (_, _, params), aligned in zip(flat_samples, aligned_flat, strict=True):
            original = np.asarray(case.apply_fn(params, case.inputs))
            after = np.asarray(case.apply_fn(aligned, case.inputs))
            drift = max(drift, float(np.max(np.abs(original - after))))
        function_drift = drift

    n_samples = len(flat_samples)
    return PosteriorBenchmarkResult(
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        function_drift_max=function_drift,
        objective_final_mean=(
            float(np.mean(objective_values)) if objective_values else None
        ),
        wall_time_s=float(wall_time),
        samples_per_s=float(n_samples / wall_time) if wall_time > 0 else float("inf"),
        aligned_chains=aligned_tuple,
    )


__all__ = [
    "PosteriorBenchmarkCase",
    "PosteriorBenchmarkResult",
    "PosteriorMetrics",
    "evaluate_posterior_metrics",
    "load_experiment_posterior_case",
    "make_synthetic_mlp_posterior_case",
    "run_posterior_benchmark",
    "split_rhat",
]
