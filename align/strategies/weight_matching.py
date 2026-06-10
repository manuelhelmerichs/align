"""Weight matching re-basin strategy (Ainsworth et al. 2023).

This module implements the Git Re-Basin weight matching algorithm as a strategy
class. The objective is to maximize <vec(Theta_A), vec(pi(Theta_B))> over
hidden-layer permutations pi = {P_1,...,P_{L-1}} while fixing P_0 = P_L = I.
Coordinate descent over layers turns this into repeated Linear Assignment
Problems (LAPs) where each P_l is updated by SOLVELAP(C_l). The discrete
alignment acts on kernels as P_l W_l P_{l-1}^T and on biases as P_l b_l, so
applying permuted layers keeps the network function invariant.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..options import WeightMatchingOptions
from . import register_strategy
from .base import RebasinStrategy

if TYPE_CHECKING:
    import jax


def solve_lap_maximize(cost: np.ndarray) -> np.ndarray:
    """Solve the LAP maximizing the Frobenius inner-product with ``cost``."""
    mat = np.ascontiguousarray(cost, dtype=np.float64)
    row_ind, col_ind = linear_sum_assignment(-mat)

    n = mat.shape[0]
    perm = np.zeros((n, n), dtype=np.float64)
    perm[row_ind, col_ind] = 1.0

    if np.issubdtype(cost.dtype, np.floating) and cost.dtype != np.float64:
        return perm.astype(cost.dtype, copy=False)
    return perm


def _collapse_views(
    views: Sequence[Any], roles: Sequence[str | None] | None, group_id: str
) -> tuple[np.ndarray, np.ndarray] | None:
    if roles is None or len(roles) == 0:
        if len(views) != 2:
            return None
        return np.asarray(views[0]), np.asarray(views[1])
    if len(roles) != len(views):
        raise ValueError(
            f"View role metadata for group '{group_id}' has length {len(roles)}, "
            f"but {len(views)} views were materialized."
        )

    incoming: list[np.ndarray] = []
    outgoing: list[np.ndarray] = []
    undecided: list[tuple[int, np.ndarray]] = []
    for idx, (view, role) in enumerate(zip(views, roles, strict=True)):
        arr = np.asarray(view)
        if role == "incoming":
            incoming.append(arr)
        elif role == "outgoing":
            outgoing.append(arr)
        elif role is None:
            # Preserve deterministic tie-breaking by position for undecided views.
            undecided.append((idx, arr))
        else:
            raise ValueError(f"Unknown view role '{role}' for group '{group_id}'.")

    if undecided:
        undecided_sorted = [
            arr for _, arr in sorted(undecided, key=lambda pair: pair[0])
        ]
        if not incoming and not outgoing:
            incoming = undecided_sorted[::2]
            outgoing = undecided_sorted[1::2]
        else:
            for arr in undecided_sorted:
                target = incoming if len(incoming) <= len(outgoing) else outgoing
                target.append(arr)

    if not incoming or not outgoing:
        return None

    def _concat(mats: Sequence[np.ndarray]) -> np.ndarray:
        if len(mats) == 1:
            return mats[0]
        return np.concatenate(mats, axis=1)

    return _concat(incoming), _concat(outgoing)


def _views_are_collapsible(
    views: Sequence[Any], roles: Sequence[str | None] | None, group_id: str
) -> bool:
    if roles is None or len(roles) == 0:
        return len(views) == 2
    if len(roles) != len(views):
        raise ValueError(
            f"View role metadata for group '{group_id}' has length {len(roles)}, "
            f"but {len(views)} views were materialized."
        )

    incoming = 0
    outgoing = 0
    undecided = 0
    for role in roles:
        if role == "incoming":
            incoming += 1
        elif role == "outgoing":
            outgoing += 1
        elif role is None:
            undecided += 1
        else:
            raise ValueError(f"Unknown view role '{role}' for group '{group_id}'.")

    if undecided:
        if incoming == 0 and outgoing == 0:
            incoming += (undecided + 1) // 2
            outgoing += undecided // 2
        else:
            for _ in range(undecided):
                if incoming <= outgoing:
                    incoming += 1
                else:
                    outgoing += 1

    return incoming > 0 and outgoing > 0


class _ChainViews:
    def __init__(
        self,
        *,
        spec,
        group_order: Sequence[str],
        roles_by_group: Mapping[str, Sequence[str | None]],
        ref_views: Mapping[str, Sequence[Any]],
        target_views: Mapping[str, Sequence[Any]],
    ) -> None:
        self.spec = spec
        self.group_order = tuple(group_order)
        self.roles_by_group = roles_by_group
        self.ref_views = ref_views
        self.target_views = target_views

    def __iter__(
        self,
    ) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        for gid in self.group_order:
            ref_list = self.ref_views.get(gid)
            tgt_list = self.target_views.get(gid)
            if ref_list is None or tgt_list is None:
                raise ValueError(f"Missing views for group '{gid}'.")

            ref_split = _collapse_views(ref_list, self.roles_by_group.get(gid), gid)
            if ref_split is None:
                raise ValueError(f"Could not split reference views for group '{gid}'.")
            incoming_ref, outgoing_ref = ref_split

            tgt_split = _collapse_views(tgt_list, self.roles_by_group.get(gid), gid)
            if tgt_split is None:
                raise ValueError(f"Could not split target views for group '{gid}'.")
            incoming_tgt, outgoing_tgt = tgt_split

            group = self.spec.groups[gid]
            if (
                incoming_ref.shape[0] != group.size
                or incoming_tgt.shape[0] != group.size
                or outgoing_ref.shape[0] != group.size
                or outgoing_tgt.shape[0] != group.size
            ):
                raise ValueError(
                    f"View shapes for group '{gid}' are incompatible with size {group.size}."
                )

            yield (
                gid,
                incoming_ref,
                outgoing_ref,
                incoming_tgt,
                outgoing_tgt,
            )


def _split_chain_views(
    spec,
    ref_views: Mapping[str, Sequence[Any]],
    target_views: Mapping[str, Sequence[Any]],
):
    """Return ordered incoming/outgoing views for coordinate descent.

    Returns None when the spec does not expose an ordered chain structure.
    """

    group_order = spec.metadata.get("group_order")
    if not group_order:
        return None
    if set(group_order) != set(ref_views):
        return None

    roles_by_group: dict[str, list[str | None]] = {gid: [] for gid in spec.groups}
    for tmpl in getattr(spec, "view_templates", []):
        roles_by_group.setdefault(tmpl.group, []).append(getattr(tmpl, "role", None))

    for gid in group_order:
        ref_list = ref_views.get(gid)
        tgt_list = target_views.get(gid)
        if ref_list is None or tgt_list is None:
            raise ValueError(f"Missing views for group '{gid}'.")
        if len(ref_list) != len(tgt_list):
            raise ValueError(f"Mismatched view counts for group '{gid}'.")
        if not _views_are_collapsible(ref_list, roles_by_group.get(gid), gid):
            return None
        if not _views_are_collapsible(tgt_list, roles_by_group.get(gid), gid):
            return None

    return _ChainViews(
        spec=spec,
        group_order=group_order,
        roles_by_group=roles_by_group,
        ref_views=ref_views,
        target_views=target_views,
    )


def _incoming_cost(
    incoming_ref: np.ndarray,
    incoming_tgt: np.ndarray,
    prev_perm: np.ndarray | None,
    prev_size: int | None,
) -> np.ndarray:
    """Compute the incoming-term cost for a group."""

    if prev_perm is None or prev_size is None:
        return incoming_ref @ incoming_tgt.T
    if prev_perm.shape != (prev_size, prev_size):
        raise ValueError(
            f"prev_perm has shape {prev_perm.shape}, expected {(prev_size, prev_size)}."
        )
    prev_perm = prev_perm.astype(incoming_ref.dtype, copy=False)

    group_size, total_features = incoming_ref.shape
    kernel_factor = total_features // prev_size
    weight_cols = kernel_factor * prev_size
    bias_cols = total_features - weight_cols

    if weight_cols == 0:
        raise ValueError(
            f"Incoming view has insufficient features to align with prev_size={prev_size}."
        )

    ref_weights = incoming_ref[:, :weight_cols].reshape(
        group_size, kernel_factor, prev_size
    )
    tgt_weights = incoming_tgt[:, :weight_cols].reshape(
        group_size, kernel_factor, prev_size
    )

    # Apply the previous permutation to incoming targets, then correlate rows.
    permuted = ref_weights @ prev_perm
    cost = np.einsum("gkp,hkp->gh", permuted, tgt_weights, optimize=True)

    if bias_cols:
        ref_bias = incoming_ref[:, weight_cols:]
        tgt_bias = incoming_tgt[:, weight_cols:]
        cost = cost + ref_bias @ tgt_bias.T

    return cost


def _outgoing_cost(
    outgoing_ref: np.ndarray,
    outgoing_tgt: np.ndarray,
    next_perm: np.ndarray | None,
    next_size: int | None,
) -> np.ndarray:
    """Compute the outgoing-term cost for a group."""

    if next_perm is None or next_size is None:
        return outgoing_ref @ outgoing_tgt.T
    if next_perm.shape != (next_size, next_size):
        raise ValueError(
            f"next_perm has shape {next_perm.shape}, expected {(next_size, next_size)}."
        )
    next_perm = next_perm.astype(outgoing_ref.dtype, copy=False)

    group_size, total_features = outgoing_ref.shape
    kernel_factor = total_features // next_size
    if kernel_factor * next_size != total_features:
        raise ValueError(
            f"Outgoing view features ({total_features}) not divisible by next_size={next_size}."
        )

    ref_weights = outgoing_ref.reshape(group_size, kernel_factor, next_size)
    tgt_weights = outgoing_tgt.reshape(group_size, kernel_factor, next_size)

    permuted = np.einsum("gkp,pq->gkq", ref_weights, next_perm, optimize=True)
    return np.einsum("gkq,hkq->gh", permuted, tgt_weights, optimize=True)


def _layer_cost(
    incoming_ref: np.ndarray,
    outgoing_ref: np.ndarray,
    incoming_tgt: np.ndarray,
    outgoing_tgt: np.ndarray,
    *,
    prev_perm: np.ndarray | None,
    next_perm: np.ndarray | None,
    prev_size: int | None,
    next_size: int | None,
) -> np.ndarray:
    """Compute the full coordinate-descent cost for one permutation group."""

    incoming = _incoming_cost(incoming_ref, incoming_tgt, prev_perm, prev_size)
    outgoing = _outgoing_cost(outgoing_ref, outgoing_tgt, next_perm, next_size)
    return incoming + outgoing


@register_strategy("weight_matching")
class WeightMatchingStrategy(RebasinStrategy):
    """Weight matching re-basin strategy using LAP solvers.

    This strategy implements the Git Re-Basin algorithm (Ainsworth et al. 2023)
    for aligning neural network weights through discrete permutations.
    """

    name = "weight_matching"

    def __init__(self, max_iters: int = 25, tol: float = 0.0, **kwargs: Any) -> None:
        """Initialize the weight matching strategy.

        Args:
            max_iters: Maximum number of LAP coordinate-descent sweeps.
            tol: Threshold for permutation updates; stop early when below tolerance.
            **kwargs: Additional keyword arguments ignored by this strategy.
        """
        self.max_iters = max_iters
        self.tol = tol

    @property
    def permutation_dtype(self) -> np.dtype:
        """Weight matching produces hard 0/1 permutations stored as uint8."""
        return np.dtype(np.uint8)

    @property
    def requires_numpy_views(self) -> bool:
        """Weight matching uses NumPy/SciPy LAP solvers."""
        return True

    def identity_permutations(
        self, spec, ref_views: Mapping[str, Sequence[Any]] | None = None
    ) -> dict[str, np.ndarray]:
        """Return numpy identity permutations for each permutation group."""
        return {
            gid: np.eye(group.size, dtype=np.float64)
            for gid, group in spec.groups.items()
        }

    def match(
        self,
        spec,
        ref_views: Mapping[str, Sequence[Any]],
        target_views: Mapping[str, Sequence[Any]],
        *,
        rng_key: jax.Array | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
        """Return permutation matrices aligning target_views to ref_views."""
        if set(ref_views) != set(target_views):
            raise ValueError("Reference and target views must share group keys.")

        chain = _split_chain_views(spec, ref_views, target_views)
        if chain is None:
            raise ValueError(
                "Weight matching requires an ordered chain AlignmentSpec, but the provided "
                "spec does not expose a valid chain structure. Expected: spec.metadata['group_order'] "
                "present; group_order keys match the view keys; and each group has exactly two views "
                "(incoming, outgoing) with shapes compatible with the group size."
            )

        group_order = spec.metadata["group_order"]
        group_sizes = {gid: spec.groups[gid].size for gid in group_order}
        perms: dict[str, np.ndarray] = {
            gid: np.eye(size, dtype=np.float64) for gid, size in group_sizes.items()
        }

        max_delta = 0.0
        sweeps = 0

        for _ in range(self.max_iters):
            sweeps += 1
            sweep_delta = 0.0
            for idx, (gid, inc_ref, out_ref, inc_tgt, out_tgt) in enumerate(chain):
                prev_perm = perms[group_order[idx - 1]] if idx > 0 else None
                next_perm = (
                    perms[group_order[idx + 1]] if idx + 1 < len(group_order) else None
                )
                prev_size = group_sizes[group_order[idx - 1]] if idx > 0 else None
                next_size = (
                    group_sizes[group_order[idx + 1]]
                    if idx + 1 < len(group_order)
                    else None
                )

                cost = _layer_cost(
                    inc_ref,
                    out_ref,
                    inc_tgt,
                    out_tgt,
                    prev_perm=prev_perm,
                    next_perm=next_perm,
                    prev_size=prev_size,
                    next_size=next_size,
                )
                updated = solve_lap_maximize(cost)
                delta = float(np.max(np.abs(updated - perms[gid])))
                sweep_delta = max(sweep_delta, delta)
                perms[gid] = updated

            max_delta = sweep_delta
            if sweep_delta <= self.tol:
                break

        aux = {"sweeps": sweeps, "max_delta": max_delta}
        return perms, aux

    def batch_match(
        self,
        spec,
        ref_views: Mapping[str, Sequence[Any]],
        target_batches,
        *,
        rng_keys: Sequence[jax.Array | None] | None = None,
    ) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any] | None]]:
        """Sequential batching path."""
        results = []
        aux_payload = []
        for target in target_batches:
            perms, aux = self.match(spec, ref_views, target)
            results.append(perms)
            aux_payload.append(aux)
        return results, aux_payload


__all__ = [
    "WeightMatchingStrategy",
    "WeightMatchingOptions",
    "solve_lap_maximize",
]
