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
):
    """Return the raw diagonal Gauss-Newton Fisher as a params-shaped tree.

    Computed with an exact dense Jacobian over all outputs, which is intended
    for the small research-scale networks this project targets.
    """

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
    fisher_tree = estimate_diag_fisher_tree(params, apply_fn, inputs)
    weights = {
        tensor_id: np.asarray(_descend(fisher_tree, spec.path), dtype=np.float64)
        for tensor_id, spec in graph.tensors.items()
    }
    flat = np.concatenate([np.ravel(value) for value in weights.values()])
    mean_fisher = float(np.mean(flat))
    if mean_fisher <= 0.0:
        raise ValueError(
            "Diagonal Fisher is identically zero; the calibration inputs do "
            "not excite the network."
        )
    floor = damping * mean_fisher
    weights = {tensor_id: value + floor for tensor_id, value in weights.items()}
    if normalize:
        scale = float(np.mean(flat)) + floor
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
) -> dict[str, Any]:
    """Materialize a ``calibration`` spec in objective kwargs into weights.

    ``{"calibration": "fisher"}`` (optionally with ``calibration_damping``)
    is replaced by ``tensor_weights`` computed at ``params`` -- which must be
    the tree the objective will actually match against (i.e. after any
    normalization). Kwargs without a ``calibration`` key pass through.
    """

    kwargs = dict(objective_kwargs or {})
    spec = kwargs.pop("calibration", None)
    if spec is None:
        return kwargs
    damping = float(kwargs.pop("calibration_damping", 1.0))
    if spec != "fisher":
        raise ValueError(f"Unknown calibration spec {spec!r}; expected 'fisher'.")
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
