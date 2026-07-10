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
import jax.numpy as jnp
import numpy as np

from align.architectures import get_recipe
from align.canonicalization import ScaleCanonicalizer, ScaleState
from align.matching import (
    TransformState,
    build_solver_sequence,
    estimate_diag_fisher_tree,
    match_sample,
    resolve_calibration_kwargs,
)
from align.symmetry import MHACircuitConstraint, SymmetryGraph

from .synthetic import (
    ParamTree,
    _make_dense_params,
    layernorm_mha_transformer_apply,
    make_layernorm_mha_transformer_params,
    make_rmsnorm_gqa_rope_transformer_params,
    mlp_apply,
    permutation_matrix,
    rmsnorm_gqa_rope_transformer_apply,
)

ApplyFn = Callable[[ParamTree, jax.Array], jax.Array]

ChainList = tuple[tuple[ParamTree, ...], ...]


@dataclass
class PosteriorBenchmarkCase:
    """A multi-chain posterior of parameter trees sharing one symmetry graph."""

    name: str
    graph: SymmetryGraph
    chains: ChainList
    reference_chain: int = 0
    reference_sample: int = 0
    apply_fn: ApplyFn | None = None
    inputs: jax.Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference(self) -> ParamTree:
        return self.chains[self.reference_chain][self.reference_sample]

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
    that are constant within and across chains report R-hat 1. Constancy is
    judged against the parameter's float32 quantization scale, not exact
    equality: canonicalized coordinates (e.g. RMSNorm scales folded to unit
    energy) land within one ulp of their canonical value, and ulp-level
    variation must not read as chain separation. Materially separated
    constant chains still report ``inf``.
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

    scale = np.mean(np.abs(split), axis=(0, 1))
    quantum = np.maximum((np.finfo(np.float32).eps * scale) ** 2, 1e-30)
    rhat = np.ones_like(within)
    active = within > quantum
    rhat[active] = np.sqrt(var_plus[active] / within[active])
    rhat[~active & (between > quantum)] = np.inf
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
        mean_params = _tree_mean(sample_list)
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


def _noise_std_tree(
    transformed: ParamTree,
    *,
    noise_mode: str,
    noise_scale: float,
    apply_fn: ApplyFn,
    inputs: jax.Array,
    fisher_damping: float = 1e-3,
) -> ParamTree:
    """Per-leaf noise standard deviations for one chain mode.

    ``isotropic`` reproduces i.i.d. noise. ``inverse_fisher`` shapes the noise
    like a Laplace-approximate posterior: std proportional to
    ``(F + damping)^(-1/2)`` with the diagonal Fisher ``F`` at the chain mode,
    rescaled so the total noise energy matches the isotropic case. Flat
    (function-irrelevant) directions then dominate weight-space distances,
    which is the realistic failure regime for unweighted L2 matching.
    """

    if noise_mode == "isotropic":
        return jax.tree_util.tree_map(
            lambda leaf: np.full(np.shape(leaf), noise_scale, dtype=np.float64),
            transformed,
        )
    if noise_mode != "inverse_fisher":
        raise ValueError(
            f"Unknown noise_mode {noise_mode!r}; expected 'isotropic' or "
            "'inverse_fisher'."
        )
    fisher = estimate_diag_fisher_tree(transformed, apply_fn, inputs)
    fisher_flat = np.concatenate(
        [
            np.ravel(np.asarray(leaf, dtype=np.float64))
            for leaf in jax.tree_util.tree_leaves(fisher)
        ]
    )
    floor = fisher_damping * float(np.mean(fisher_flat))
    raw = jax.tree_util.tree_map(
        lambda leaf: 1.0 / np.sqrt(np.asarray(leaf, dtype=np.float64) + floor),
        fisher,
    )
    raw_flat = np.concatenate(
        [np.ravel(leaf) for leaf in jax.tree_util.tree_leaves(raw)]
    )
    rms = float(np.sqrt(np.mean(np.square(raw_flat))))
    return jax.tree_util.tree_map(lambda leaf: noise_scale * leaf / rms, raw)


def make_synthetic_mlp_posterior_case(
    *,
    seed: int = 0,
    n_chains: int = 4,
    n_samples: int = 8,
    sizes: Sequence[int] = (3, 5, 4, 2),
    noise_scale: float = 0.02,
    scale_jitter: float = 0.3,
    n_inputs: int = 16,
    noise_mode: str = "isotropic",
) -> PosteriorBenchmarkCase:
    """Build a single-mode posterior whose chains sit in symmetry copies.

    Every chain applies one fixed random permutation and positive scale
    transformation of a shared base mode (function-preserving), then adds
    Gaussian weight noise per sample (scale ``noise_scale``). ``noise_mode``
    selects the within-chain noise shape: ``isotropic`` (i.i.d.) or
    ``inverse_fisher`` (Laplace-like, std proportional to ``F^(-1/2)`` at each
    chain's mode; see :func:`_noise_std_tree`). Before alignment, cross-chain
    R-hat is far above 1 because chain means differ by symmetry actions; after
    scale canonicalization plus matching should collapse all chains onto one basin and
    R-hat should approach 1.
    """

    if n_chains < 2:
        raise ValueError("Posterior cases require at least two chains.")
    if n_samples < 4:
        raise ValueError("Posterior cases require at least four samples per chain.")

    key = jax.random.PRNGKey(seed)
    param_key, input_key = jax.random.split(key)
    base = _make_dense_params(key=param_key, sizes=tuple(sizes))
    recipe = get_recipe("mlp")
    graph = recipe.build_graph(base)
    inputs = jax.random.normal(input_key, (n_inputs, int(sizes[0])))

    rng = np.random.default_rng(seed)
    chains: list[tuple[ParamTree, ...]] = []
    chain_transforms: list[dict[str, Any]] = []
    for chain_idx in range(n_chains):
        if chain_idx == 0:
            transformed = base
            transforms: dict[str, np.ndarray] = {}
            scales: dict[str, np.ndarray] = {}
        else:
            transforms = {
                gid: permutation_matrix(rng.permutation(group.size))
                for gid, group in graph.groups.items()
            }
            scales = {
                gid: np.exp(
                    rng.uniform(-scale_jitter, scale_jitter, size=group.size)
                ).astype(np.float32)
                for gid, group in graph.groups.items()
            }
            scaled = graph.apply_scales(base, ScaleState.from_scales(graph, scales))
            transformed = graph.apply_transforms(
                scaled, TransformState.from_transforms(graph, transforms)
            )
        chain_transforms.append({"transforms": transforms, "scales": scales})

        noise_std = _noise_std_tree(
            transformed,
            noise_mode=noise_mode,
            noise_scale=noise_scale,
            apply_fn=mlp_apply,
            inputs=inputs,
        )
        samples = []
        for _ in range(n_samples):
            noisy = jax.tree_util.tree_map(
                lambda leaf, std: (
                    np.asarray(leaf)
                    + (std * rng.standard_normal(np.shape(leaf))).astype(
                        np.asarray(leaf).dtype
                    )
                ),
                transformed,
                noise_std,
            )
            samples.append(noisy)
        chains.append(tuple(samples))

    return PosteriorBenchmarkCase(
        name=f"synthetic_mlp_posterior_{noise_mode}_seed_{seed}"
        if noise_mode != "isotropic"
        else f"synthetic_mlp_posterior_seed_{seed}",
        graph=graph,
        chains=tuple(chains),
        apply_fn=mlp_apply,
        inputs=inputs,
        metadata={
            "seed": seed,
            "n_chains": n_chains,
            "n_samples": n_samples,
            "noise_scale": noise_scale,
            "scale_jitter": scale_jitter,
            "noise_mode": noise_mode,
            "chain_transforms": chain_transforms,
        },
    )


def make_synthetic_layernorm_mha_transformer_posterior_case(
    *,
    seed: int = 0,
    n_chains: int = 3,
    n_samples: int = 6,
    noise_scale: float = 0.01,
    scale_jitter: float = 0.5,
    noise_mode: str = "isotropic",
) -> PosteriorBenchmarkCase:
    """Build a single-mode GPT posterior whose chains sit in symmetry copies.

    Every non-reference chain applies one fixed random symmetry of a shared
    base mode: permutations on all groups (stream, heads, intra-head, FFN)
    composed with positive diagonal scales on the qk/vo intra-head groups (the
    exact attention circuit symmetry), then Gaussian weight noise per sample
    (``noise_mode`` as in :func:`make_synthetic_mlp_posterior_case`). Matching
    alone cannot remove the circuit scales, so full collapse requires
    balancing canonicalization first.
    """

    if n_chains < 2:
        raise ValueError("Posterior cases require at least two chains.")
    if n_samples < 4:
        raise ValueError("Posterior cases require at least four samples per chain.")

    key = jax.random.PRNGKey(seed)
    param_key, input_key = jax.random.split(key)
    base = make_layernorm_mha_transformer_params(key=param_key)
    recipe = get_recipe("layernorm_mha_transformer")
    graph = recipe.build_graph(base)
    tokens = jax.random.randint(input_key, (4, 5), 0, 7, dtype=jnp.int32)
    intra_groups = [
        group_id
        for constraint in graph.constraints
        if isinstance(constraint, MHACircuitConstraint)
        for group_id in (
            *constraint.qk_groups,
            *constraint.vo_groups,
        )
    ]

    rng = np.random.default_rng(seed)
    chains: list[tuple[ParamTree, ...]] = []
    chain_transforms: list[dict[str, Any]] = []
    for chain_idx in range(n_chains):
        if chain_idx == 0:
            transformed = base
            transforms: dict[str, np.ndarray] = {}
            scales: dict[str, np.ndarray] = {}
        else:
            transforms = {
                gid: permutation_matrix(rng.permutation(group.size))
                for gid, group in graph.groups.items()
            }
            scales = {
                gid: np.exp(
                    rng.uniform(
                        -scale_jitter, scale_jitter, size=graph.groups[gid].size
                    )
                ).astype(np.float32)
                for gid in intra_groups
            }
            scaled = graph.apply_scales(base, ScaleState.from_scales(graph, scales))
            transformed = graph.apply_transforms(
                scaled, TransformState.from_transforms(graph, transforms)
            )
        chain_transforms.append({"transforms": transforms, "scales": scales})

        noise_std = _noise_std_tree(
            transformed,
            noise_mode=noise_mode,
            noise_scale=noise_scale,
            apply_fn=layernorm_mha_transformer_apply,
            inputs=tokens,
        )
        samples = []
        for _ in range(n_samples):
            noisy = jax.tree_util.tree_map(
                lambda leaf, std: (
                    np.asarray(leaf)
                    + (std * rng.standard_normal(np.shape(leaf))).astype(
                        np.asarray(leaf).dtype
                    )
                ),
                transformed,
                noise_std,
            )
            samples.append(noisy)
        chains.append(tuple(samples))

    return PosteriorBenchmarkCase(
        name=f"synthetic_layernorm_mha_transformer_posterior_{noise_mode}_seed_{seed}"
        if noise_mode != "isotropic"
        else f"synthetic_layernorm_mha_transformer_posterior_seed_{seed}",
        graph=graph,
        chains=tuple(chains),
        apply_fn=layernorm_mha_transformer_apply,
        inputs=tokens,
        metadata={
            "seed": seed,
            "n_chains": n_chains,
            "n_samples": n_samples,
            "noise_scale": noise_scale,
            "scale_jitter": scale_jitter,
            "noise_mode": noise_mode,
            "chain_transforms": chain_transforms,
        },
    )


def make_synthetic_rmsnorm_gqa_rope_transformer_posterior_case(
    *,
    seed: int = 0,
    n_chains: int = 3,
    n_samples: int = 6,
    noise_scale: float = 0.01,
    scale_jitter: float = 0.5,
    noise_mode: str = "isotropic",
) -> PosteriorBenchmarkCase:
    """Single-mode RMSNorm/RoPE/GQA posterior whose chains sit in symmetry copies.

    Every non-reference chain applies one fixed random symmetry of a shared
    base mode — the family's full implemented group: signed permutations on
    the stream and vo groups, plain permutations on kv/query-head/FFN groups,
    per-pair qk rotations, composed with the exact diagonal scales (RMSNorm
    gamma, pair-tied qk, vo) — then Gaussian weight noise per sample. Matching
    alone cannot remove the scales, so full collapse requires the
    RMSNorm/RoPE/GQA canonicalization plan first.
    """

    from align.canonicalization.rmsnorm_gqa_rope import (
        apply_rms_gamma_scales,
        gqa_rope_circuit_constraints,
        rmsnorm_scale_constraints,
    )

    from .synthetic import rotation_pairs_matrix, signed_permutation_matrix

    if n_chains < 2:
        raise ValueError("Posterior cases require at least two chains.")
    if n_samples < 4:
        raise ValueError("Posterior cases require at least four samples per chain.")

    key = jax.random.PRNGKey(seed)
    param_key, input_key = jax.random.split(key)
    base = make_rmsnorm_gqa_rope_transformer_params(key=param_key)
    recipe = get_recipe("rmsnorm_gqa_rope_transformer")
    graph = recipe.build_graph(base)
    tokens = jax.random.randint(input_key, (4, 5), 0, 7, dtype=jnp.int32)

    rng = np.random.default_rng(seed)
    chains: list[tuple[ParamTree, ...]] = []
    chain_transforms: list[dict[str, Any]] = []
    for chain_idx in range(n_chains):
        if chain_idx == 0:
            transformed = base
            transforms: dict[str, np.ndarray] = {}
        else:
            gamma_scales = {
                constraint.scale: np.exp(
                    rng.uniform(
                        -scale_jitter,
                        scale_jitter,
                        size=graph.tensors[constraint.scale].shape[0],
                    )
                ).astype(np.float32)
                for constraint in rmsnorm_scale_constraints(graph)
            }
            transformed = apply_rms_gamma_scales(graph, base, gamma_scales)
            circuit_scales = {}
            for constraint in gqa_rope_circuit_constraints(graph):
                head_dim = constraint.head_dim
                for group_id in constraint.qk_groups:
                    pair = np.exp(
                        rng.uniform(-scale_jitter, scale_jitter, size=head_dim // 2)
                    )
                    circuit_scales[group_id] = np.concatenate([pair, pair]).astype(
                        np.float32
                    )
                for group_id in constraint.vo_groups:
                    circuit_scales[group_id] = np.exp(
                        rng.uniform(-scale_jitter, scale_jitter, size=head_dim)
                    ).astype(np.float32)
            transformed = graph.apply_scales(
                transformed, ScaleState.from_scales(graph, circuit_scales)
            )
            transforms = {}
            for gid, group in graph.groups.items():
                if group.transform_family == "rotation_pairs":
                    transforms[gid] = rotation_pairs_matrix(rng, group.size)
                elif group.transform_family in ("signed_permutation", "orthogonal"):
                    transforms[gid] = signed_permutation_matrix(rng, group.size)
                else:
                    transforms[gid] = permutation_matrix(rng.permutation(group.size))
            transformed = graph.apply_transforms(
                transformed, TransformState.from_transforms(graph, transforms)
            )
        chain_transforms.append({"transforms": transforms})

        noise_std = _noise_std_tree(
            transformed,
            noise_mode=noise_mode,
            noise_scale=noise_scale,
            apply_fn=rmsnorm_gqa_rope_transformer_apply,
            inputs=tokens,
        )
        samples = []
        for _ in range(n_samples):
            noisy = jax.tree_util.tree_map(
                lambda leaf, std: (
                    np.asarray(leaf)
                    + (std * rng.standard_normal(np.shape(leaf))).astype(
                        np.asarray(leaf).dtype
                    )
                ),
                transformed,
                noise_std,
            )
            samples.append(noisy)
        chains.append(tuple(samples))

    return PosteriorBenchmarkCase(
        name=f"synthetic_rmsnorm_gqa_rope_transformer_posterior_{noise_mode}_seed_{seed}"
        if noise_mode != "isotropic"
        else f"synthetic_rmsnorm_gqa_rope_transformer_posterior_seed_{seed}",
        graph=graph,
        chains=tuple(chains),
        apply_fn=rmsnorm_gqa_rope_transformer_apply,
        inputs=tokens,
        metadata={
            "seed": seed,
            "n_chains": n_chains,
            "n_samples": n_samples,
            "noise_scale": noise_scale,
            "scale_jitter": scale_jitter,
            "noise_mode": noise_mode,
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
    recipe_kwargs: Mapping[str, Any] | None = None,
    samples_dir: str | Path | None = None,
    tree_path: str | Path | None = None,
    chain_indices: Sequence[int] | None = None,
    samples_per_chain: int | None = None,
    sample_step: int = 1,
    max_total: int | None = None,
    reference_chain: int = 0,
    reference_sample: int = 0,
    sample_format: str = "pytree_npz",
    name: str | None = None,
) -> PosteriorBenchmarkCase:
    """Load a real SMILE/MILE experiment directory as a posterior case.

    All selected samples are materialized in memory, so restrict the selection
    (``samples_per_chain``, ``max_total``) for large posteriors. Function-space
    metrics are unavailable because real experiments carry no ``apply_fn``.
    """

    from align.runtime.loaders import SampleLoader
    from align.sample_manifest import SampleManifest

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
        reference_chain=reference_chain,
        reference_sample=reference_sample,
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
    reference_chain_index = chain_ids.index(manifest.reference_record.chain_id)
    reference_sample_ids = [
        sid for sid, _ in sorted(by_chain[chain_ids[reference_chain_index]])
    ]
    reference_sample_index = reference_sample_ids.index(
        manifest.reference_record.sample_id
    )

    reference = chains[reference_chain_index][reference_sample_index]
    recipe = get_recipe(architecture, **dict(recipe_kwargs or {}))
    graph = recipe.build_graph(reference)

    return PosteriorBenchmarkCase(
        name=name or f"experiment_{root.name}",
        graph=graph,
        chains=chains,
        reference_chain=reference_chain_index,
        reference_sample=reference_sample_index,
        metadata={
            "experiment_root": str(root),
            "architecture": architecture,
            "chain_ids": chain_ids,
            "total_samples": manifest.total,
        },
    )


def _canonicalize_tree(graph: SymmetryGraph, params: ParamTree) -> ParamTree:
    canonical, _, _ = ScaleCanonicalizer().canonicalize(
        graph,
        params,
        include_bias_in_norm=True,
    )
    return canonical


def _tree_mean(trees: Sequence[ParamTree]) -> ParamTree:
    return jax.tree_util.tree_map(
        lambda *leaves: np.mean(
            np.stack([np.asarray(leaf) for leaf in leaves]), axis=0
        ),
        *trees,
    )


def run_posterior_benchmark(
    case: PosteriorBenchmarkCase,
    *,
    objective: str = "euclidean",
    objective_kwargs: Mapping[str, Any] | None = None,
    schedule: Sequence[Mapping[str, Any]] | None = None,
    canonicalize: bool = False,
    rng_seed: int = 0,
    barycenter_passes: int = 2,
) -> PosteriorBenchmarkResult:
    """Align every sample of a posterior case to its reference and score it.

    ``metrics_before`` is computed on the raw input chains, ``metrics_after``
    on the optionally canonicalized and matched chains, so the comparison
    reflects the complete alignment treatment.

    ``barycenter_passes > 1`` (the default is 2) enables iterative barycenter
    refinement: after each pass, the reference is replaced by the mean of the
    aligned samples and every canonicalized sample is re-aligned from scratch
    against it. The single reference *sample* carries its own posterior
    noise, which becomes the matching bottleneck at high noise — and makes
    single-pass canonicalization reference-dependent at moderate noise; the
    aligned mean estimates the basin barycenter with noise reduced by
    ``1/sqrt(n_samples)``. Objectives with ``calibration`` kwargs re-resolve
    at each pass's reference (the metric lives at the matching base point).
    Past the matching breakdown point refinement can hurt (see
    ``docs/theory.md``); pass ``barycenter_passes=1`` there.
    """

    if barycenter_passes < 1:
        raise ValueError("barycenter_passes must be at least 1.")

    metrics_before = evaluate_posterior_metrics(case, case.chains)

    reference = case.reference
    if canonicalize:
        reference = _canonicalize_tree(case.graph, reference)

    flat_samples = case.iter_samples()
    targets = [
        _canonicalize_tree(case.graph, params) if canonicalize else params
        for _, _, params in flat_samples
    ]

    aligned_flat: list[ParamTree] = []
    objective_values: list[float] = []
    start = time.perf_counter()
    for pass_idx in range(barycenter_passes):
        solver_sequence = build_solver_sequence(
            objective=objective,
            objective_kwargs=resolve_calibration_kwargs(
                objective_kwargs,
                graph=case.graph,
                params=reference,
                apply_fn=case.apply_fn,
                inputs=case.inputs,
            ),
            schedule=schedule,
        )
        reference_data = case.graph.materialize(
            reference, backend=solver_sequence.backend, cache=True
        )
        aligned_flat = []
        objective_values = []
        for index, target in enumerate(targets):
            aligned, _, aux = match_sample(
                case.graph,
                reference,
                target,
                solver_sequence=solver_sequence,
                reference_data=reference_data,
                rng_key=jax.random.fold_in(
                    jax.random.PRNGKey(rng_seed), pass_idx * len(targets) + index
                ),
            )
            aligned_flat.append(aligned)
            if aux is not None and "objective_final" in aux:
                objective_values.append(float(aux["objective_final"]))
        if pass_idx < barycenter_passes - 1:
            reference = _tree_mean(aligned_flat)
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
    "make_synthetic_layernorm_mha_transformer_posterior_case",
    "run_posterior_benchmark",
    "split_rhat",
]
