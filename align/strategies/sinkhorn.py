"""Sinkhorn re-basin strategy (Guerrero Peña et al.).

This module implements the relaxed weight matching algorithm as a strategy class.
Instead of discrete permutations, it optimizes doubly stochastic transport plans
P_l in the polytope Pi = {P >= 0 | P 1 = 1, P^T 1 = 1} such that the
re-parameterized layer sigma(P_l W_l P_{l-1}^T z + P_l b_l) aligns target weights
to the reference. The Sinkhorn operator S_tau(X) = argmax_{P in Pi} <P, X>_F + tau h(P)
projects unconstrained logits onto the doubly stochastic set via unrolled
row/column renormalization.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..options import SinkhornOptions
from . import register_strategy
from .base import RebasinStrategy
from .costs import get_cost_function
from .weight_matching import solve_lap_maximize

if TYPE_CHECKING:
    from ..architecture import AlignmentSpec


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def sinkhorn_operator(
    logits: jnp.ndarray, tau: float = 0.1, n_iters: int = 50, eps: float = 1e-9
) -> jnp.ndarray:
    """Project ``logits`` (batched or single) onto the doubly stochastic set."""

    scaled = logits / tau
    shifted = scaled - jnp.max(scaled, axis=(-2, -1), keepdims=True)
    matrix = jnp.exp(shifted)

    def body(_, mat):
        mat = mat / (jnp.sum(mat, axis=-2, keepdims=True) + eps)
        mat = mat / (jnp.sum(mat, axis=-1, keepdims=True) + eps)
        return mat

    matrix = jax.lax.fori_loop(0, n_iters, body, matrix)
    return matrix


def _init_group_logits(
    spec: AlignmentSpec,
    *,
    batch_size: int,
    init_scale: float,
    rng_key: jax.Array,
    dtype,
) -> tuple[jnp.ndarray, ...]:
    """Initialize logits for each permutation group."""
    keys = jax.random.split(rng_key, len(spec.groups))
    logits = []
    for key, group in zip(keys, spec.groups.values(), strict=True):
        logits.append(
            init_scale
            * jax.random.normal(key, (batch_size, group.size, group.size), dtype=dtype)
        )
    return tuple(logits)


_HUNGARIAN_EXECUTOR: ThreadPoolExecutor | None = None
_HUNGARIAN_MAX_WORKERS = 16


def _get_hungarian_executor(total_jobs: int) -> ThreadPoolExecutor | None:
    """Get or create a thread pool for parallel Hungarian solving.

    Args:
        total_jobs: Total number of LAP problems to solve (batch * layers).
    """
    global _HUNGARIAN_EXECUTOR
    if total_jobs <= 1:
        return None
    if _HUNGARIAN_EXECUTOR is None:
        _HUNGARIAN_EXECUTOR = ThreadPoolExecutor(max_workers=_HUNGARIAN_MAX_WORKERS)
    return _HUNGARIAN_EXECUTOR


def _split_batched_permutations(
    perms: Sequence[jnp.ndarray],
    parallel: bool = True,
) -> list[list[jnp.ndarray]]:
    """Split batched soft permutations into per-sample hard permutations.

    Converts soft (doubly stochastic) permutation matrices to hard permutations
    via the Hungarian algorithm.

    Args:
        perms: Sequence of batched soft permutation matrices, each shape (batch, dim, dim)
        parallel: Whether to use parallel execution for Hungarian algorithm
    """
    if not perms:
        return []
    batch = perms[0].shape[0]
    num_layers = len(perms)
    total_jobs = batch * num_layers

    perms_np = [np.asarray(perm) for perm in perms]

    if parallel and total_jobs > 1:
        executor = _get_hungarian_executor(total_jobs)
        if executor is not None:
            all_futures = []
            job_indices = []

            for layer_idx, perm_np in enumerate(perms_np):
                for sample_idx in range(batch):
                    future = executor.submit(solve_lap_maximize, perm_np[sample_idx])
                    all_futures.append(future)
                    job_indices.append((sample_idx, layer_idx))

            outputs: list[list[jnp.ndarray | None]] = [
                [None] * num_layers for _ in range(batch)
            ]

            for future, (sample_idx, layer_idx) in zip(
                all_futures, job_indices, strict=True
            ):
                solved = future.result()
                outputs[sample_idx][layer_idx] = jnp.asarray(solved)

            return [[perm for perm in sample_perms] for sample_perms in outputs]  # type: ignore

    outputs = [[] for _ in range(batch)]
    for perm_np in perms_np:
        for idx in range(batch):
            solved = solve_lap_maximize(perm_np[idx])
            outputs[idx].append(jnp.asarray(solved))
    return outputs


_JIT_LOSS_FN_CACHE: dict[str, Any] = {}


def _make_loss_fn(
    cost_fn, tau: float, n_sinkhorn_iters: int, group_order: Sequence[str]
):
    """Create a loss function for grouped alignment data."""

    def loss_fn(logits, ref_data, tgt_data):
        soft_perms = {
            gid: sinkhorn_operator(logits[idx], tau=tau, n_iters=n_sinkhorn_iters)
            for idx, gid in enumerate(group_order)
        }
        losses = cost_fn.compute_batched_cost(None, ref_data, tgt_data, soft_perms)
        return jnp.sum(losses), losses

    return loss_fn


def _get_cached_loss_and_grad(cost_fn, tau: float, n_sinkhorn_iters: int, group_order):
    """Get or create a JIT-compiled loss-and-grad function."""
    cost_name = getattr(cost_fn, "name", type(cost_fn).__name__)
    cache_key = (cost_name, tau, n_sinkhorn_iters, tuple(group_order))

    if cache_key not in _JIT_LOSS_FN_CACHE:
        loss_fn = _make_loss_fn(cost_fn, tau, n_sinkhorn_iters, tuple(group_order))
        jit_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=True))
        _JIT_LOSS_FN_CACHE[cache_key] = (jit_fn, cost_fn)

    return _JIT_LOSS_FN_CACHE[cache_key][0]


@register_strategy("sinkhorn")
class SinkhornStrategy(RebasinStrategy):
    """Sinkhorn re-basin strategy using relaxed transport optimization.

    This strategy implements the relaxed weight matching algorithm from
    Guerrero Peña et al. using Sinkhorn projections onto the doubly stochastic
    polytope, followed by Hungarian discretization for hard permutations.
    """

    name = "sinkhorn"

    def __init__(
        self,
        tau: float = 0.1,
        n_sinkhorn_iters: int = 50,
        lr: float = 1e-2,
        max_steps: int = 200,
        tol: float = 1e-5,
        init_scale: float = 1e-2,
        record_loss_history: bool = False,
        cost_function: str = "l2_weight",
        cost_function_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Sinkhorn strategy.

        Args:
            tau: Entropic regularization temperature for the Sinkhorn operator.
            n_sinkhorn_iters: Number of row/column normalization iterations.
            lr: Optimizer learning rate for updating Sinkhorn logits.
            max_steps: Gradient steps for solving the relaxed transport problem.
            tol: Early-stop tolerance on consecutive Sinkhorn losses.
            init_scale: Standard deviation of the initial logits.
            record_loss_history: Whether to persist full per-sample loss curves.
            cost_function: Name of the cost function used to score soft permutations.
            cost_function_kwargs: Extra kwargs forwarded to the cost constructor.
            **kwargs: Additional keyword arguments ignored by this strategy.
        """
        self.tau = tau
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.lr = lr
        self.max_steps = max_steps
        self.tol = tol
        self.init_scale = init_scale
        self.record_loss_history = record_loss_history
        self.cost_fn = get_cost_function(cost_function, **(cost_function_kwargs or {}))

    @property
    def permutation_dtype(self) -> np.dtype:
        """Sinkhorn produces soft transport plans stored as float32."""
        return np.dtype(np.float32)

    @property
    def requires_numpy_views(self) -> bool:
        """Sinkhorn uses JAX for GPU-accelerated optimization."""
        return False

    def supports_batching(self) -> bool:
        """Sinkhorn supports efficient batched processing."""
        return True

    def identity_permutations(
        self, spec, ref_views: Mapping[str, Sequence[Any]] | None = None
    ) -> dict[str, jnp.ndarray]:
        return {
            gid: jnp.eye(group.size, dtype=jnp.float32)
            for gid, group in spec.groups.items()
        }

    def match(
        self,
        spec,
        ref_views: Mapping[str, Sequence[Any]],
        target_views: Mapping[str, Sequence[Any]],
        *,
        rng_key: jax.Array | None = None,
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any] | None]:
        soft_perms, aux = self._match_core(
            spec=spec,
            ref_data={
                gid: [jnp.asarray(v) for v in views] for gid, views in ref_views.items()
            },
            target_data={
                gid: [jnp.asarray(v) for v in views]
                for gid, views in target_views.items()
            },
            rng_key=rng_key,
        )
        hard_group_list = _split_batched_permutations(list(soft_perms.values()))
        group_order = list(soft_perms.keys())
        hard = {gid: hard_group_list[0][idx] for idx, gid in enumerate(group_order)}
        return hard, aux

    def batch_match(
        self,
        spec,
        ref_views: Mapping[str, Sequence[Any]],
        target_batches,
        *,
        rng_keys: Sequence[jax.Array | None] | None = None,
    ) -> tuple[list[dict[str, jnp.ndarray]], list[dict[str, Any] | None]]:
        if not target_batches:
            return [], []
        if rng_keys is not None and len(rng_keys) != len(target_batches):
            raise ValueError("rng_keys length must match target_batches length.")

        stacked_targets = self._stack_target_views(spec, target_batches)
        perms, aux = self._match_core(
            spec=spec,
            ref_data={
                gid: [jnp.asarray(v) for v in views] for gid, views in ref_views.items()
            },
            target_data=stacked_targets,
            rng_key=rng_keys[0] if rng_keys else None,
        )
        soft_perms = list(perms.values())
        hard_group_list = _split_batched_permutations(soft_perms)
        group_order = list(perms.keys())

        hard_perms: list[dict[str, jnp.ndarray]] = []
        for sample_perms in hard_group_list:
            hard_perms.append(
                {gid: perm for gid, perm in zip(group_order, sample_perms, strict=True)}
            )

        aux_payload = [aux for _ in hard_perms]
        return hard_perms, aux_payload

    def _stack_target_views(self, spec, target_batches):
        group_order = list(spec.groups.keys())
        stacked_targets: dict[str, list[jnp.ndarray]] = {}
        for gid in group_order:
            per_sample = [sample[gid] for sample in target_batches]
            view_count = len(per_sample[0])
            stacked_list: list[jnp.ndarray] = []
            for view_idx in range(view_count):
                stacked_list.append(
                    jnp.stack(
                        [jnp.asarray(sample[view_idx]) for sample in per_sample], axis=0
                    )
                )
            stacked_targets[gid] = stacked_list
        return stacked_targets

    def _match_core(
        self,
        *,
        spec,
        ref_data,
        target_data,
        rng_key: jax.Array | None = None,
    ):
        group_order = list(spec.groups.keys())
        batch_size = target_data[group_order[0]][0].shape[0]
        dtype = target_data[group_order[0]][0].dtype

        logits = _init_group_logits(
            spec,
            batch_size=batch_size,
            init_scale=self.init_scale,
            rng_key=rng_key if rng_key is not None else jax.random.PRNGKey(0),
            dtype=dtype,
        )
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(logits)

        loss_and_grad = _get_cached_loss_and_grad(
            self.cost_fn, self.tau, self.n_sinkhorn_iters, group_order
        )

        history = [[] for _ in range(batch_size)] if self.record_loss_history else None
        initial_losses: list[float | None] = [None for _ in range(batch_size)]
        final_losses: list[float | None] = [None for _ in range(batch_size)]
        steps_taken = 0
        prev_losses: jnp.ndarray | None = None

        for _ in range(self.max_steps):
            (loss_sum, losses), grads = loss_and_grad(logits, ref_data, target_data)
            del loss_sum
            loss_values = jnp.asarray(losses)
            for idx in range(loss_values.shape[0]):
                loss_value = float(loss_values[idx])
                if history is not None:
                    history[idx].append(loss_value)
                if initial_losses[idx] is None:
                    initial_losses[idx] = loss_value
                final_losses[idx] = loss_value
            updates, opt_state = optimizer.update(grads, opt_state, logits)
            logits = optax.apply_updates(logits, updates)
            steps_taken += 1
            if prev_losses is not None:
                delta = float(jnp.max(jnp.abs(loss_values - prev_losses)))
                if delta < self.tol:
                    break
            prev_losses = loss_values

        soft_perms_list = [
            sinkhorn_operator(logit, tau=self.tau, n_iters=self.n_sinkhorn_iters)
            for logit in logits
        ]
        soft_perms = {gid: soft_perms_list[idx] for idx, gid in enumerate(group_order)}
        aux: dict[str, Any] = {
            "loss_initial": [
                float(val) if val is not None else 0.0 for val in initial_losses
            ],
            "loss_final": [
                float(val) if val is not None else 0.0 for val in final_losses
            ],
            "steps": int(steps_taken),
        }
        if history is not None:
            aux["loss_history"] = history
        return soft_perms, aux


__all__ = [
    "SinkhornStrategy",
    "SinkhornOptions",
    "sinkhorn_operator",
]
