"""Posterior-cloud diagnostics used by refinement orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import jax
import numpy as np

from ..matching import TransformState, match_sample
from ..samples import ParamTree, WeightSample


def tree_mean(samples: Iterable[WeightSample]) -> WeightSample:
    """Compute a streaming, dtype-preserving mean of weight samples."""

    totals = None
    dtypes = None
    count = 0
    for sample in samples:
        if totals is None:
            totals = jax.tree_util.tree_map(
                lambda leaf: np.asarray(leaf, dtype=np.float64), sample.params
            )
            dtypes = jax.tree_util.tree_map(
                lambda leaf: np.asarray(leaf).dtype, sample.params
            )
        else:
            totals = jax.tree_util.tree_map(
                lambda total, leaf: total + np.asarray(leaf, dtype=np.float64),
                totals,
                sample.params,
            )
        count += 1

    if totals is None or dtypes is None or count == 0:
        raise ValueError("Cannot compute the mean of an empty sample collection.")
    params = jax.tree_util.tree_map(
        lambda total, dtype: np.asarray(total / count, dtype=dtype), totals, dtypes
    )
    return WeightSample(params=params)


def _tree_distance(left: ParamTree, right: ParamTree) -> float:
    squared = 0.0
    left_leaves, left_def = jax.tree_util.tree_flatten(left)
    right_leaves, right_def = jax.tree_util.tree_flatten(right)
    if left_def != right_def:
        raise ValueError("Cannot compare samples with different PyTree structures.")
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        diff = np.asarray(left_leaf, dtype=np.float64) - np.asarray(
            right_leaf, dtype=np.float64
        )
        squared += float(np.vdot(diff, diff).real)
    return float(np.sqrt(squared))


def reference_stability_diagnostic(
    *,
    previous_samples: Callable[[], Iterable[WeightSample]],
    current_samples: Callable[[], Iterable[WeightSample]],
    graph,
    solvers: Sequence[Mapping[str, Any]] | Sequence[Any],
    previous_pass: int,
    current_pass: int,
) -> dict[str, Any]:
    """Convergence ratio measuring reference stability across a barycenter update.

    Compares two consecutive canonicalized clouds modulo one global frame: the
    current barycenter is first matched to the previous barycenter, that single
    frame transform is applied to every current sample, and the mean paired
    shift is divided by the previous cloud's mean radius. The result is a
    convergence ratio -- near zero means canonical positions stopped moving when
    the reference was re-estimated (reference stability reached); large means the
    barycenter iteration has not settled. It is a stopping/self-consistency
    signal, not a certificate that the matching found the correct basin.
    """

    previous_mean = tree_mean(previous_samples())
    current_mean = tree_mean(current_samples())
    # Match the two barycenters with a plain L2 frame even under a data-dependent
    # objective (e.g. diagonal_fisher): here we want the geometric movement of canonical
    # positions between passes, not an objective-weighted distance.
    _, frame, _ = match_sample(
        graph,
        previous_mean.params,
        current_mean.params,
        objective="euclidean",
        schedule=solvers,
        rng_key=jax.random.PRNGKey(0),
    )
    frame_state = TransformState.from_transforms(graph, frame)

    shift_total = 0.0
    spread_total = 0.0
    count = 0
    previous_iter = iter(previous_samples())
    current_iter = iter(current_samples())
    while True:
        try:
            previous = next(previous_iter)
        except StopIteration:
            try:
                next(current_iter)
            except StopIteration:
                break
            raise ValueError(
                "Canonicalized clouds have different sample counts."
            ) from None
        try:
            current = next(current_iter)
        except StopIteration as exc:
            raise ValueError(
                "Canonicalized clouds have different sample counts."
            ) from exc

        moved = graph.apply_transforms(current.params, frame_state)
        shift_total += _tree_distance(previous.params, moved)
        spread_total += _tree_distance(previous.params, previous_mean.params)
        count += 1

    if count == 0:
        raise ValueError("Cannot diagnose empty canonicalized clouds.")
    mean_position_shift = shift_total / count
    cloud_spread = spread_total / count
    if cloud_spread > 0.0:
        convergence_ratio: float | None = mean_position_shift / cloud_spread
    elif mean_position_shift == 0.0:
        convergence_ratio = 0.0
    else:
        convergence_ratio = None

    return {
        "previous_pass": int(previous_pass),
        "current_pass": int(current_pass),
        "mean_position_shift": float(mean_position_shift),
        "cloud_spread": float(cloud_spread),
        "convergence_ratio": (
            float(convergence_ratio) if convergence_ratio is not None else None
        ),
    }


__all__ = ["reference_stability_diagnostic", "tree_mean"]
