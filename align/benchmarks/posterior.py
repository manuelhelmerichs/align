"""Posterior-level benchmarks: multi-chain quality metrics and real samples.

While :mod:`benchmarks.synthetic` scores single reference/target pairs on
exact orbits, this module benchmarks alignment across a whole posterior: many
samples organized into chains. It provides

- a synthetic multi-chain posterior with a known ground truth (one mode whose
  chains live in different symmetry copies, plus small within-chain noise),
- a loader that turns a real SMILE/MILE experiment directory into the same
  case representation, and
- chain-aware quality metrics (split R-hat, pairwise distances, the
  weight-averaging gap, linear-interpolation barriers) evaluated before and
  after alignment.
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
from scipy.special import ndtri
from scipy.stats import rankdata

from align.architectures import get_recipe
from align.canonicalization import ScaleCanonicalizer, ScaleState
from align.matching import (
    TransformState,
    build_solver_sequence,
    estimate_diag_fisher_tree,
    match_batch,
    match_sample,
    resolve_calibration_kwargs,
)
from align.symmetry import GQARoPECircuitConstraint, MHACircuitConstraint, SymmetryGraph

from .synthetic import (
    ParamTree,
    layernorm_mha_transformer_apply,
    make_dense_params,
    make_layernorm_mha_transformer_params,
    make_rmsnorm_gqa_rope_transformer_params,
    mlp_activation_samples,
    mlp_apply,
    permutation_matrix,
    rmsnorm_gqa_rope_transformer_apply,
)

ApplyFn = Callable[[ParamTree, jax.Array], jax.Array]
ActivationFn = Callable[[ParamTree, jax.Array], Mapping[str, Mapping[str, Any]]]
PredictionTransform = Callable[[jax.Array], jax.Array]

ChainList = tuple[tuple[ParamTree, ...], ...]

IndexPair = tuple[tuple[int, int], tuple[int, int]]

BARRIER_TS: tuple[float, ...] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
BARRIER_MAX_PAIRS: int = 16

_BARRIER_METRIC_KEYS: tuple[str, ...] = (
    "function_cross_barrier_mean",
    "function_cross_barrier_max",
    "function_within_barrier_mean",
    "function_cross_within_barrier_ratio",
)


@dataclass
class PosteriorBenchmarkCase:
    """A multi-chain posterior of parameter trees sharing one symmetry graph."""

    name: str
    graph: SymmetryGraph
    chains: ChainList
    reference_chain: int = 0
    reference_sample: int = 0
    apply_fn: ApplyFn | None = None
    activation_fn: ActivationFn | None = None
    inputs: jax.Array | None = None
    prediction_transform: PredictionTransform | None = None
    functional_batch_size: int | None = None
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
    function_variance_total: float | None
    function_within_chain_variance: float | None
    function_between_chain_mean_variance: float | None
    function_between_variance_fraction: float | None
    function_within_rms_distance: float | None
    function_cross_rms_distance: float | None
    function_cross_within_distance_ratio: float | None
    chain_mean_prediction_rmse_mean: float | None
    chain_mean_prediction_rmse_max: float | None
    function_cross_barrier_mean: float | None
    function_cross_barrier_max: float | None
    function_within_barrier_mean: float | None
    function_cross_within_barrier_ratio: float | None


@dataclass(frozen=True)
class PosteriorBenchmarkResult:
    """Before/after metrics and runtime statistics for one posterior run."""

    metrics_before: PosteriorMetrics
    metrics_after: PosteriorMetrics
    function_drift_max: float | None
    function_drift_rmse: float | None
    function_drift_relative_rmse: float | None
    objective_final_mean: float | None
    wall_time_s: float
    samples_per_s: float
    aligned_chains: ChainList
    head_assignment_accuracy: float | None = None
    head_assignment_exact_rate: float | None = None


def _flatten_sample(params: ParamTree) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(params)
    return np.concatenate(
        [np.ravel(np.asarray(leaf, dtype=np.float64)) for leaf in leaves]
    )


def stack_chains(chains: ChainList) -> np.ndarray:
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
    _, n, _ = split.shape

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


def rank_normalize(draws: np.ndarray) -> np.ndarray:
    """Rank-normalize each parameter across chains and draws.

    The Blom offset follows the rank-normalized split-R-hat construction of
    Vehtari et al. (2021). Ties receive their average rank.
    """

    if draws.ndim != 3:
        raise ValueError(f"Expected (chains, samples, params), got {draws.shape}.")
    n_chains, n_samples, n_parameters = draws.shape
    flat = draws.reshape(n_chains * n_samples, n_parameters)
    ranks = rankdata(flat, axis=0, method="average")
    normalized = ndtri((ranks - 0.375) / (flat.shape[0] + 0.25))
    return normalized.reshape(draws.shape)


def rank_normalized_split_rhat(draws: np.ndarray) -> dict[str, np.ndarray]:
    """Return bulk, folded, and combined rank-normalized split-R-hat.

    ``bulk`` applies split-R-hat after rank normalization. ``folded`` first
    folds each parameter around its pooled median, then rank-normalizes; this
    detects scale differences that bulk R-hat can miss. ``max`` is the
    coordinatewise maximum recommended as the final rank-R-hat diagnostic.
    """

    bulk = split_rhat(rank_normalize(draws))
    median = np.median(draws, axis=(0, 1), keepdims=True)
    folded_draws = np.abs(draws - median)
    folded = split_rhat(rank_normalize(folded_draws))
    return {"bulk": bulk, "folded": folded, "max": np.maximum(bulk, folded)}


def _apply_in_batches(case: PosteriorBenchmarkCase, params: ParamTree) -> np.ndarray:
    """Evaluate one parameter tree on the case inputs without changing order."""

    if case.apply_fn is None or case.inputs is None:
        raise ValueError("Functional evaluation requires both apply_fn and inputs.")
    batch_size = case.functional_batch_size
    if batch_size is None:
        return np.asarray(case.apply_fn(params, case.inputs))
    if batch_size < 1:
        raise ValueError("functional_batch_size must be positive when provided.")
    outputs = [
        np.asarray(case.apply_fn(params, case.inputs[start : start + batch_size]))
        for start in range(0, int(case.inputs.shape[0]), batch_size)
    ]
    return np.concatenate(outputs, axis=0)


def evaluate_function_outputs(
    case: PosteriorBenchmarkCase, chains: ChainList
) -> np.ndarray | None:
    """Return raw model outputs with shape ``(chains, draws, ...)``.

    The raw outputs are also the exactness certificate for a modeled symmetry:
    unlike probabilities, logits retain every functionally meaningful degree of
    freedom represented by the architecture recipe.
    """

    if case.apply_fn is None or case.inputs is None:
        return None
    shortest = min(len(chain) for chain in chains)
    return np.stack(
        [
            np.stack([_apply_in_batches(case, sample) for sample in chain[:shortest]])
            for chain in chains
        ]
    )


def _prediction_outputs(
    case: PosteriorBenchmarkCase, outputs: np.ndarray
) -> np.ndarray:
    if case.prediction_transform is None:
        return outputs
    return np.asarray(case.prediction_transform(jnp.asarray(outputs)))


def _interpolated_tree(a: ParamTree, b: ParamTree, t: float) -> ParamTree:
    return jax.tree_util.tree_map(
        lambda leaf_a, leaf_b: (1.0 - t) * np.asarray(leaf_a) + t * np.asarray(leaf_b),
        a,
        b,
    )


def _cross_chain_pairs(
    rng: np.random.Generator, n_chains: int, n_samples: int, max_pairs: int
) -> list[IndexPair]:
    """Unordered cross-chain draw pairs: all of them, or ``max_pairs`` sampled."""

    total = n_chains * (n_chains - 1) // 2 * n_samples * n_samples
    if total <= max_pairs:
        return [
            ((left, i), (right, j))
            for left in range(n_chains)
            for right in range(left + 1, n_chains)
            for i in range(n_samples)
            for j in range(n_samples)
        ]
    pairs: set[IndexPair] = set()
    while len(pairs) < max_pairs:
        left, right = (int(c) for c in rng.choice(n_chains, size=2, replace=False))
        i = int(rng.integers(n_samples))
        j = int(rng.integers(n_samples))
        if left > right:
            left, right, i, j = right, left, j, i
        pairs.add(((left, i), (right, j)))
    return sorted(pairs)


def _within_chain_pairs(
    rng: np.random.Generator, n_chains: int, n_samples: int, max_pairs: int
) -> list[IndexPair]:
    """Unordered within-chain draw pairs: all of them, or ``max_pairs`` sampled."""

    total = n_chains * n_samples * (n_samples - 1) // 2
    if total <= max_pairs:
        return [
            ((chain, i), (chain, j))
            for chain in range(n_chains)
            for i in range(n_samples)
            for j in range(i + 1, n_samples)
        ]
    pairs: set[IndexPair] = set()
    while len(pairs) < max_pairs:
        chain = int(rng.integers(n_chains))
        i, j = sorted(int(s) for s in rng.choice(n_samples, size=2, replace=False))
        pairs.add(((chain, i), (chain, j)))
    return sorted(pairs)


def evaluate_interpolation_barriers(
    case: PosteriorBenchmarkCase,
    chains: ChainList,
    *,
    function_outputs: np.ndarray | None = None,
    ts: Sequence[float] = BARRIER_TS,
    max_pairs: int = BARRIER_MAX_PAIRS,
    seed: int = 0,
) -> dict[str, float | None] | None:
    """Linear-interpolation barrier metrics over sampled draw pairs.

    For a draw pair ``(a, b)`` and predictions ``p = prediction_transform o
    apply_fn``, the function-space barrier is the largest chord deviation
    along the weight-space segment ``theta_t = (1 - t) a + t b``:

        barrier(a, b) = max_t RMSE( p(theta_t), (1 - t) p(a) + t p(b) )

    A model affine in its parameters satisfies ``p(theta_t) = (1 - t) p(a) +
    t p(b)`` identically, so the barrier measures exactly the nonlinear
    deviation of the path functions from the endpoint chord -- the
    community-standard evidence for two draws sharing one linearly connected
    basin (Garipov et al. 2018; Frankle et al. 2020; Entezari et al. 2022 use
    loss barriers). The function-space RMSE form needs no data targets, is
    symmetric in ``(a, b)`` for a t-grid symmetric about one half, and lives
    in the same prediction space as the other functional diagnostics. It is
    deliberately *not* invariant under alignment: alignment moves draws
    within their orbits, and the before/after barrier difference quantifies
    how much of the linear tunnel between chains is opened by removing
    modeled symmetries.

    Barriers are evaluated on up to ``max_pairs`` seeded cross-chain pairs --
    the "one basin?" question -- and up to ``max_pairs`` within-chain pairs,
    whose mean is the within-basin noise floor calibrating what "small"
    means; the ratio reported is the cross mean over the within mean. The
    endpoints ``t = 0, 1`` are exactly zero by construction, so the t-grid
    must lie strictly inside ``(0, 1)``.

    Requires ``apply_fn`` and ``inputs`` on the case; returns ``None``
    otherwise. Passing ``function_outputs`` (from
    :func:`evaluate_function_outputs`) reuses the endpoint predictions, so
    only interior grid points cost additional forward passes.
    """

    if case.apply_fn is None or case.inputs is None:
        return None
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive.")
    grid = tuple(float(t) for t in ts)
    if not grid or not all(0.0 < t < 1.0 for t in grid):
        raise ValueError(
            "Barrier t-grid values must lie strictly inside (0, 1); the "
            "endpoints are exactly zero by construction."
        )

    n_chains = len(chains)
    shortest = min(len(chain) for chain in chains)
    endpoint_predictions = (
        None
        if function_outputs is None
        else _prediction_outputs(case, function_outputs).astype(np.float64)
    )
    endpoint_cache: dict[tuple[int, int], np.ndarray] = {}

    def endpoint(chain: int, sample: int) -> np.ndarray:
        if endpoint_predictions is not None:
            return endpoint_predictions[chain, sample]
        key = (chain, sample)
        if key not in endpoint_cache:
            outputs = _apply_in_batches(case, chains[chain][sample])
            endpoint_cache[key] = _prediction_outputs(case, outputs[None, None, ...])[
                0, 0
            ].astype(np.float64)
        return endpoint_cache[key]

    def barrier(pair: IndexPair) -> float:
        (left, i), (right, j) = pair
        a, b = chains[left][i], chains[right][j]
        pred_a, pred_b = endpoint(left, i), endpoint(right, j)
        worst = 0.0
        for t in grid:
            outputs = _apply_in_batches(case, _interpolated_tree(a, b, t))
            prediction = _prediction_outputs(case, outputs[None, None, ...])[
                0, 0
            ].astype(np.float64)
            chord = (1.0 - t) * pred_a + t * pred_b
            deviation = float(np.sqrt(np.mean(np.square(prediction - chord))))
            worst = max(worst, deviation)
        return worst

    rng = np.random.default_rng(seed)
    cross = [
        barrier(pair) for pair in _cross_chain_pairs(rng, n_chains, shortest, max_pairs)
    ]
    within = [
        barrier(pair)
        for pair in _within_chain_pairs(rng, n_chains, shortest, max_pairs)
    ]

    cross_mean = float(np.mean(cross)) if cross else None
    within_mean = float(np.mean(within)) if within else None
    return {
        "function_cross_barrier_mean": cross_mean,
        "function_cross_barrier_max": float(np.max(cross)) if cross else None,
        "function_within_barrier_mean": within_mean,
        "function_cross_within_barrier_ratio": (
            cross_mean / max(within_mean, 1e-30)
            if cross_mean is not None and within_mean is not None
            else None
        ),
    }


def _functional_metrics(
    case: PosteriorBenchmarkCase,
    chains: ChainList,
    outputs: np.ndarray | None,
) -> dict[str, float | None]:
    empty = {
        "weight_averaging_gap": None,
        "function_variance_total": None,
        "function_within_chain_variance": None,
        "function_between_chain_mean_variance": None,
        "function_between_variance_fraction": None,
        "function_within_rms_distance": None,
        "function_cross_rms_distance": None,
        "function_cross_within_distance_ratio": None,
        "chain_mean_prediction_rmse_mean": None,
        "chain_mean_prediction_rmse_max": None,
        **{key: None for key in _BARRIER_METRIC_KEYS},
    }
    if outputs is None:
        return empty

    predictions = _prediction_outputs(case, outputs).astype(np.float64)
    flat = predictions.reshape(predictions.shape[0], predictions.shape[1], -1)
    n_chains, n_samples, _ = flat.shape
    chain_means = flat.mean(axis=1)
    within_variances = flat.var(axis=1, ddof=0).mean(axis=1)
    within_variance = float(np.mean(within_variances))
    total_variance = float(flat.reshape(-1, flat.shape[-1]).var(axis=0).mean())
    between_variance = float(chain_means.var(axis=0, ddof=0).mean())

    if n_samples > 1:
        within_squared = 2.0 * n_samples / (n_samples - 1) * within_variances
        within_rms = float(np.sqrt(max(float(np.mean(within_squared)), 0.0)))
    else:
        within_rms = 0.0

    second_moments = np.mean(np.square(flat), axis=(1, 2))
    cross_squared: list[float] = []
    chain_mean_rmse: list[float] = []
    for left in range(n_chains):
        for right in range(left + 1, n_chains):
            cross_squared.append(
                float(
                    second_moments[left]
                    + second_moments[right]
                    - 2.0 * np.mean(chain_means[left] * chain_means[right])
                )
            )
            chain_mean_rmse.append(
                float(
                    np.sqrt(np.mean(np.square(chain_means[left] - chain_means[right])))
                )
            )
    cross_rms = (
        float(np.sqrt(max(float(np.mean(cross_squared)), 0.0)))
        if cross_squared
        else 0.0
    )

    sample_list = [sample for chain in chains for sample in chain[:n_samples]]
    mean_params = tree_mean(sample_list)
    mean_prediction = _prediction_outputs(
        case, _apply_in_batches(case, mean_params)[None, None, ...]
    )[0, 0]
    bma_prediction = predictions.mean(axis=(0, 1))
    averaging_gap = float(np.sqrt(np.mean(np.square(mean_prediction - bma_prediction))))

    barriers = evaluate_interpolation_barriers(case, chains, function_outputs=outputs)
    if barriers is None:
        barriers = {key: None for key in _BARRIER_METRIC_KEYS}

    return {
        **barriers,
        "weight_averaging_gap": averaging_gap,
        "function_variance_total": total_variance,
        "function_within_chain_variance": within_variance,
        "function_between_chain_mean_variance": between_variance,
        "function_between_variance_fraction": between_variance
        / max(total_variance, 1e-30),
        "function_within_rms_distance": within_rms,
        "function_cross_rms_distance": cross_rms,
        "function_cross_within_distance_ratio": cross_rms / max(within_rms, 1e-30),
        "chain_mean_prediction_rmse_mean": float(np.mean(chain_mean_rmse))
        if chain_mean_rmse
        else 0.0,
        "chain_mean_prediction_rmse_max": float(np.max(chain_mean_rmse))
        if chain_mean_rmse
        else 0.0,
    }


def evaluate_function_drift(
    case: PosteriorBenchmarkCase,
    original: ChainList,
    transformed: ChainList,
    *,
    original_outputs: np.ndarray | None = None,
    transformed_outputs: np.ndarray | None = None,
) -> dict[str, float] | None:
    """Certify exact functional preservation between corresponding chains."""

    if original_outputs is None:
        original_outputs = evaluate_function_outputs(case, original)
    if transformed_outputs is None:
        transformed_outputs = evaluate_function_outputs(case, transformed)
    if original_outputs is None or transformed_outputs is None:
        return None
    if original_outputs.shape != transformed_outputs.shape:
        raise ValueError(
            "Function-drift comparison requires corresponding chain outputs; "
            f"got {original_outputs.shape} and {transformed_outputs.shape}."
        )
    delta = transformed_outputs.astype(np.float64) - original_outputs.astype(np.float64)
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    reference_rms = float(
        np.sqrt(np.mean(np.square(original_outputs.astype(np.float64))))
    )
    return {
        "max": float(np.max(np.abs(delta))),
        "rmse": rmse,
        "relative_rmse": rmse / max(reference_rms, 1e-30),
    }


def evaluate_posterior_metrics(
    case: PosteriorBenchmarkCase,
    chains: ChainList,
    *,
    function_outputs: np.ndarray | None = None,
) -> PosteriorMetrics:
    """Compute weight- and invariant function-space chain metrics."""

    draws = stack_chains(chains)
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

    if function_outputs is None:
        function_outputs = evaluate_function_outputs(case, chains)
    function_metrics = _functional_metrics(case, chains, function_outputs)

    return PosteriorMetrics(
        split_rhat_max=float(np.max(rhat)),
        split_rhat_mean=float(np.mean(rhat)),
        pairwise_distance_mean=float(np.mean(distances)),
        pairwise_distance_max=float(np.max(distances)),
        cross_chain_distance_mean=float(np.mean(cross)) if cross else 0.0,
        weight_variance_total=float(np.sum(flat.var(axis=0))),
        **function_metrics,
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
    base = make_dense_params(key=param_key, sizes=tuple(sizes))
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
        activation_fn=mlp_activation_samples,
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
    base mode -- the family's full implemented group: signed permutations on
    the stream and vo groups, plain permutations on kv/query-head/FFN groups,
    per-pair qk rotations, composed with the exact diagonal scales (RMSNorm
    gamma, pair-tied qk, vo) -- then Gaussian weight noise per sample. Matching
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
    name: str | None = None,
    apply_fn: ApplyFn | None = None,
    activation_fn: ActivationFn | None = None,
    inputs: jax.Array | None = None,
    prediction_transform: PredictionTransform | None = None,
    functional_batch_size: int | None = None,
) -> PosteriorBenchmarkCase:
    """Load a real SMILE/MILE experiment directory as a posterior case.

    All selected samples are materialized in memory, so restrict the selection
    (``samples_per_chain``, ``max_total``) for large posteriors. Pass the exact
    model ``apply_fn`` and evaluation ``inputs`` to enable invariant functional
    diagnostics. ``prediction_transform`` maps raw model outputs to the
    representation in which disagreement is measured (for example softmax
    probabilities); raw outputs remain the symmetry-drift certificate.
    """

    if (apply_fn is None) != (inputs is None):
        raise ValueError("apply_fn and inputs must be provided together.")
    if prediction_transform is not None and apply_fn is None:
        raise ValueError("prediction_transform requires apply_fn and inputs.")
    if functional_batch_size is not None and functional_batch_size < 1:
        raise ValueError("functional_batch_size must be positive when provided.")

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
        apply_fn=apply_fn,
        activation_fn=activation_fn,
        inputs=inputs,
        prediction_transform=prediction_transform,
        functional_batch_size=functional_batch_size,
        metadata={
            "experiment_root": str(root),
            "architecture": architecture,
            "chain_ids": chain_ids,
            "total_samples": manifest.total,
        },
    )


def _canonicalize_tree(graph: SymmetryGraph, params: ParamTree) -> ParamTree:
    architecture = graph.metadata.get("architecture")
    kwargs = {"include_bias_in_norm": True}
    if architecture in {"mlp", "convnet"}:
        kwargs["activation"] = "relu"
    elif architecture == "residual_convnet":
        kwargs["activation"] = "tlu"
    canonical, _, _ = ScaleCanonicalizer().canonicalize(
        graph,
        params,
        **kwargs,
    )
    return canonical


def tree_mean(trees: Sequence[ParamTree]) -> ParamTree:
    return jax.tree_util.tree_map(
        lambda *leaves: np.mean(
            np.stack([np.asarray(leaf) for leaf in leaves]), axis=0
        ),
        *trees,
    )


def _head_assignment_recovery(case, flat_samples, transforms_flat):
    """Direct oracle recovery for synthetic MHA head / GQA kv assignments."""

    chain_transforms = case.metadata.get("chain_transforms")
    if chain_transforms is None:
        return None, None
    head_groups = [
        constraint.head_group
        for constraint in case.graph.constraints
        if isinstance(constraint, MHACircuitConstraint)
    ] + [
        constraint.kv_group
        for constraint in case.graph.constraints
        if isinstance(constraint, GQARoPECircuitConstraint)
    ]
    if not head_groups:
        return None, None
    has_transformed_chain = any(
        bool(entry.get("transforms")) for entry in chain_transforms
    )

    correct = 0
    total = 0
    exact = 0
    comparisons = 0
    for (chain_index, _, _), transforms in zip(
        flat_samples, transforms_flat, strict=True
    ):
        forward = chain_transforms[chain_index]["transforms"]
        if has_transformed_chain and not forward:
            continue
        for group_id in head_groups:
            size = case.graph.groups[group_id].size
            expected = (
                np.asarray(forward[group_id]).T if group_id in forward else np.eye(size)
            )
            actual = np.asarray(transforms[group_id])
            expected_indices = np.argmax(expected, axis=1)
            actual_indices = (
                actual.astype(np.int64)
                if actual.ndim == 1
                else np.argmax(actual, axis=1)
            )
            matches = expected_indices == actual_indices
            correct += int(np.sum(matches))
            total += size
            exact += int(np.all(matches))
            comparisons += 1
    return correct / total, exact / comparisons


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
    noise, which becomes the matching bottleneck at high noise -- and makes
    single-pass canonicalization reference-dependent at moderate noise; the
    aligned mean estimates the basin barycenter with noise reduced by
    ``1/sqrt(n_samples)``. Objectives with ``calibration`` kwargs re-resolve
    at each pass's reference (the metric lives at the matching base point).
    Past the matching breakdown point refinement can hurt (see
    ``wiki/Matching-objectives-and-solvers.md``); pass ``barycenter_passes=1`` there.
    """

    if barycenter_passes < 1:
        raise ValueError("barycenter_passes must be at least 1.")

    outputs_before = evaluate_function_outputs(case, case.chains)
    metrics_before = evaluate_posterior_metrics(
        case, case.chains, function_outputs=outputs_before
    )

    reference = case.reference
    if canonicalize:
        reference = _canonicalize_tree(case.graph, reference)

    flat_samples = case.iter_samples()
    targets = [
        _canonicalize_tree(case.graph, params) if canonicalize else params
        for _, _, params in flat_samples
    ]

    aligned_flat: list[ParamTree] = []
    transforms_flat: list[Mapping[str, Any]] = []
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
                activation_fn=case.activation_fn,
            ),
            schedule=schedule,
        )
        reference_data = case.graph.materialize(
            reference, backend=solver_sequence.backend, cache=True
        )
        aligned_flat = []
        transforms_flat = []
        objective_values = []
        if solver_sequence.supports_batching:
            batch_results = match_batch(
                case.graph,
                reference,
                targets,
                solver_sequence=solver_sequence,
                reference_data=reference_data,
                rng_key=jax.random.fold_in(jax.random.PRNGKey(rng_seed), pass_idx),
            )
            for aligned, transforms, aux in batch_results:
                aligned_flat.append(aligned)
                transforms_flat.append(transforms)
                if aux is not None and "objective_final" in aux:
                    objective_values.append(float(aux["objective_final"]))
        else:
            for index, target in enumerate(targets):
                aligned, transforms, aux = match_sample(
                    case.graph,
                    reference,
                    target,
                    solver_sequence=solver_sequence,
                    reference_data=reference_data,
                    rng_key=jax.random.fold_in(
                        jax.random.PRNGKey(rng_seed),
                        pass_idx * len(targets) + index,
                    ),
                )
                aligned_flat.append(aligned)
                transforms_flat.append(transforms)
                if aux is not None and "objective_final" in aux:
                    objective_values.append(float(aux["objective_final"]))
        if pass_idx < barycenter_passes - 1:
            reference = tree_mean(aligned_flat)
    wall_time = time.perf_counter() - start

    aligned_chains: list[list[ParamTree]] = [[] for _ in case.chains]
    for (chain_idx, _, _), aligned in zip(flat_samples, aligned_flat, strict=True):
        aligned_chains[chain_idx].append(aligned)
    aligned_tuple: ChainList = tuple(tuple(chain) for chain in aligned_chains)

    outputs_after = evaluate_function_outputs(case, aligned_tuple)
    metrics_after = evaluate_posterior_metrics(
        case, aligned_tuple, function_outputs=outputs_after
    )
    function_drift = evaluate_function_drift(
        case,
        case.chains,
        aligned_tuple,
        original_outputs=outputs_before,
        transformed_outputs=outputs_after,
    )

    n_samples = len(flat_samples)
    head_accuracy, head_exact_rate = _head_assignment_recovery(
        case, flat_samples, transforms_flat
    )
    return PosteriorBenchmarkResult(
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        function_drift_max=None if function_drift is None else function_drift["max"],
        function_drift_rmse=None if function_drift is None else function_drift["rmse"],
        function_drift_relative_rmse=None
        if function_drift is None
        else function_drift["relative_rmse"],
        objective_final_mean=(
            float(np.mean(objective_values)) if objective_values else None
        ),
        wall_time_s=float(wall_time),
        samples_per_s=float(n_samples / wall_time) if wall_time > 0 else float("inf"),
        aligned_chains=aligned_tuple,
        head_assignment_accuracy=head_accuracy,
        head_assignment_exact_rate=head_exact_rate,
    )


__all__ = [
    "PosteriorBenchmarkCase",
    "PosteriorBenchmarkResult",
    "PosteriorMetrics",
    "evaluate_function_drift",
    "evaluate_function_outputs",
    "evaluate_interpolation_barriers",
    "evaluate_posterior_metrics",
    "load_experiment_posterior_case",
    "make_synthetic_mlp_posterior_case",
    "make_synthetic_layernorm_mha_transformer_posterior_case",
    "run_posterior_benchmark",
    "split_rhat",
    "stack_chains",
]
