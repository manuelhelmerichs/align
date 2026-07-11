"""Diagonal Fisher estimation for geometry-aware matching objectives.

The Fisher information metric is the canonical local inner product on weight
space (Liang et al., 2019; Sun & Nielsen, 2017): directions that change the
realized function strongly are long, function-irrelevant (flat) directions are
short. ``estimate_diag_fisher_weights`` computes the diagonal of the
output-space (Gauss-Newton) Fisher

    F_i = sum_{n,k} (d f_k(x_n) / d theta_i)^2,

the pullback of function-space L2 into weight space at ``params`` --
equivalently the Fisher information for a Gaussian observation model
``y ~ N(f_theta(x), I)``. Evaluating it at the *reference* sample gives one
fixed metric for matching every target against that reference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..symmetry.tensor_ops import _descend


def estimate_diag_fisher_tree(
    params: Mapping[str, Any],
    apply_fn: Callable[[Mapping[str, Any], jax.Array], jax.Array],
    inputs: jax.Array,
    *,
    memory_limit_bytes: int = 2 * 1024**3,
):
    """Return the raw diagonal Gauss-Newton Fisher as a params-shaped tree.

    Computed with an exact dense Jacobian over all outputs, which is intended
    for the small research-scale networks this project targets.
    """

    output_shape = jax.eval_shape(lambda tree: apply_fn(tree, inputs), params)
    output_size = int(np.prod(output_shape.shape, dtype=np.int64))
    leaves = jax.tree_util.tree_leaves(params)
    parameter_size = sum(int(np.prod(leaf.shape, dtype=np.int64)) for leaf in leaves)
    itemsize = max(np.dtype(leaf.dtype).itemsize for leaf in leaves)
    estimated_bytes = output_size * parameter_size * itemsize
    if estimated_bytes > int(memory_limit_bytes):
        raise MemoryError(
            "Exact diagonal-Fisher Jacobian would require approximately "
            f"{estimated_bytes / 1024**3:.2f} GiB, exceeding the configured "
            f"limit of {memory_limit_bytes / 1024**3:.2f} GiB. Use a smaller "
            "calibration batch/network or a chunked estimator."
        )

    def _flat_outputs(tree):
        return jnp.ravel(apply_fn(tree, inputs))

    jacobian = jax.jacrev(_flat_outputs)(params)
    return jax.tree_util.tree_map(
        lambda leaf: jnp.sum(jnp.square(leaf), axis=0), jacobian
    )


def estimate_diag_fisher_weights(
    graph,
    params: Mapping[str, Any],
    apply_fn: Callable[[Mapping[str, Any], jax.Array], jax.Array],
    inputs: jax.Array,
    *,
    damping: float = 1.0,
    normalize: bool = True,
    memory_limit_bytes: int = 2 * 1024**3,
) -> dict[str, np.ndarray]:
    """Per-tensor diagonal Fisher weights for ``graph`` tensors.

    ``damping`` adds ``damping * mean(F)`` to every coordinate, interpolating
    between pure Fisher geometry (``damping -> 0``) and plain L2
    (``damping -> inf``). The floor is load-bearing, not just numerical:
    coordinates the calibration data does not excite carry ~zero Fisher
    weight, so with a small floor their placement is metric-free and both LAP
    coordinate descent and Sinkhorn can settle on wrong permutations that the
    metric considers equivalent (function-preserving but bad weight-space
    canonicalization). The default of 1.0 keeps the L2 tie-breaking signal at
    the same scale as the Fisher signal and was uniformly at least as good in
    benchmarks. With ``normalize`` the weights are rescaled to mean 1, which
    keeps ``diagonal_fisher`` objective values on the same scale as ``euclidean``.
    """

    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}.")
    fisher_tree = estimate_diag_fisher_tree(
        params, apply_fn, inputs, memory_limit_bytes=memory_limit_bytes
    )
    weights = {
        tensor_id: np.asarray(_descend(fisher_tree, spec.path), dtype=np.float64)
        for tensor_id, spec in graph.tensors.items()
    }
    total = sum(float(np.sum(value)) for value in weights.values())
    count = sum(int(value.size) for value in weights.values())
    mean_fisher = total / count
    if mean_fisher <= 0.0:
        raise ValueError(
            "Diagonal Fisher is identically zero; the calibration inputs do "
            "not excite the network."
        )
    floor = damping * mean_fisher
    weights = {tensor_id: value + floor for tensor_id, value in weights.items()}
    if normalize:
        scale = mean_fisher + floor
        weights = {tensor_id: value / scale for tensor_id, value in weights.items()}
    return weights


def save_tensor_weights_npz(
    weights: Mapping[str, np.ndarray], path: str | Path
) -> None:
    """Serialize per-tensor weights keyed by tensor id."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **{key: np.asarray(value) for key, value in weights.items()})


def load_tensor_weights_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load per-tensor weights keyed by tensor id."""

    with np.load(Path(path)) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def resolve_calibration_kwargs(
    objective_kwargs: Mapping[str, Any] | None,
    *,
    graph,
    params: Mapping[str, Any],
    apply_fn: Callable[[Mapping[str, Any], jax.Array], jax.Array] | None,
    inputs: jax.Array | None,
    activation_fn: Callable[[Mapping[str, Any], Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Materialize a data-dependent ``calibration`` objective specification.

    ``{"calibration": "fisher"}`` is replaced by diagonal ``tensor_weights``;
    ``{"calibration": "relative_fisher"}`` is replaced by activation-Gram
    ``tensor_metrics``.  Both are computed at ``params``, which must be the
    tree the objective will actually match against (after canonicalization).
    Kwargs without a ``calibration`` key pass through.
    """

    kwargs = dict(objective_kwargs or {})
    spec = kwargs.get("calibration")
    if spec is None:
        return kwargs
    if spec == "relative_fisher":
        from .relative_fisher import resolve_relative_fisher_calibration_kwargs

        return resolve_relative_fisher_calibration_kwargs(
            kwargs,
            graph=graph,
            params=params,
            activation_fn=activation_fn,
            inputs=inputs,
        )
    kwargs.pop("calibration")
    damping = float(kwargs.pop("calibration_damping", 1.0))
    if spec != "fisher":
        raise ValueError(
            f"Unknown calibration spec {spec!r}; expected 'fisher' or "
            "'relative_fisher'."
        )
    if apply_fn is None or inputs is None:
        raise ValueError(
            "calibration='fisher' requires an apply_fn and calibration inputs; "
            "for cases without them, precompute weights and pass "
            "'weights_path' instead."
        )
    kwargs["tensor_weights"] = estimate_diag_fisher_weights(
        graph, params, apply_fn, inputs, damping=damping
    )
    return kwargs


__all__ = [
    "estimate_diag_fisher_tree",
    "estimate_diag_fisher_weights",
    "load_tensor_weights_npz",
    "resolve_calibration_kwargs",
    "save_tensor_weights_npz",
]
