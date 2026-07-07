"""Structured attention-module updates for LAP coordinate descent.

Attention symmetries form a wreath product: an inter-head permutation shared by
q/k/v/out plus per-slot intra-head permutations (qk and vo circuits). The head
group's generic LAP linearization is inexact once intra permutations are
non-identity, so head matching uses QK/OV circuit costs, which are invariant to
intra-head permutations (see the Transformer subsection in docs/theory.md).
Intra groups are then updated with the generic exact LAP linearization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..alignment import binding_axis_interval
from .objectives import _apply_other_groups_hard
from .permutation_state import PermutationState, as_permutation_matrix


@dataclass(frozen=True)
class AttentionModuleSpec:
    """One attention module's coupled groups and tensors."""

    head_group: str
    qk_groups: tuple[str, ...]
    vo_groups: tuple[str, ...]
    query: str
    key: str
    value: str
    out: str
    num_heads: int
    head_dim: int

    @classmethod
    def from_constraint(cls, problem, constraint) -> AttentionModuleSpec:
        metadata = constraint.metadata
        head_group = str(metadata["head_group"])
        qk_groups = tuple(str(gid) for gid in metadata["qk_groups"])
        vo_groups = tuple(str(gid) for gid in metadata["vo_groups"])
        tensors = dict(metadata["tensors"])
        return cls(
            head_group=head_group,
            qk_groups=qk_groups,
            vo_groups=vo_groups,
            query=str(tensors["query"]),
            key=str(tensors["key"]),
            value=str(tensors["value"]),
            out=str(tensors["out"]),
            num_heads=int(problem.groups[head_group].size),
            head_dim=int(problem.groups[qk_groups[0]].size),
        )

    @property
    def groups(self) -> tuple[str, ...]:
        return (self.head_group, *self.qk_groups, *self.vo_groups)


def attention_module_specs(problem) -> tuple[AttentionModuleSpec, ...]:
    """Return specs for every ``attention_block`` constraint of ``problem``."""

    return tuple(
        AttentionModuleSpec.from_constraint(problem, constraint)
        for constraint in problem.constraints
        if constraint.kind == "attention_block"
    )


def _head_matrices(problem, spec, tensor_id: str, tensor) -> np.ndarray:
    """Return per-head matrices ``(H, other, dk)`` for one q/k/v/out tensor.

    ``other`` collects the remaining axes (the stream axis for the bayesmates
    layout); ``dk`` is the intra-head axis bound by the module's qk/vo groups.
    """

    shape = problem.tensors[tensor_id].shape
    head_axis = None
    intra_axis = None
    intra_groups = {*spec.qk_groups, *spec.vo_groups}
    for binding in problem.bindings_for_tensor(tensor_id):
        axis, _, _ = binding_axis_interval(shape, binding)
        if binding.group == spec.head_group:
            head_axis = axis
        elif binding.group in intra_groups:
            intra_axis = axis
    if head_axis is None or intra_axis is None:
        raise ValueError(
            f"Tensor {tensor_id!r} lacks head/intra bindings for attention "
            f"module {spec.head_group!r}."
        )
    arr = np.asarray(tensor)
    moved = np.moveaxis(arr, (head_axis, intra_axis), (0, arr.ndim - 1))
    return moved.reshape(spec.num_heads, -1, spec.head_dim)


def _circuits(
    problem, spec, data, *, state=None, skip_groups: set[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return stacked QK and OV circuit matrices, one pair per head."""

    tensors = {}
    for tensor_id in (spec.query, spec.key, spec.value, spec.out):
        tensor = data[tensor_id]
        if state is not None:
            tensor = _apply_other_groups_hard(
                problem, tensor_id, tensor, state, skip_groups or set()
            )
        tensors[tensor_id] = _head_matrices(problem, spec, tensor_id, tensor)

    query, key = tensors[spec.query], tensors[spec.key]
    value, out = tensors[spec.value], tensors[spec.out]
    qk = np.einsum("hik,hjk->hij", query, key, optimize=True)
    ov = np.einsum("hik,hjk->hij", value, out, optimize=True)
    return qk, ov


def head_cost_matrix(
    problem, ref_data, target_data, state, spec: AttentionModuleSpec
) -> np.ndarray:
    """Circuit-distance cost of assigning target head ``j`` to ref slot ``i``.

    All groups outside the attention module (e.g. the residual stream) are
    applied to the target first; the module's own groups are excluded, which is
    exact because circuit costs are invariant to intra-head permutations.
    """

    ref_qk, ref_ov = _circuits(problem, spec, ref_data)
    target_qk, target_ov = _circuits(
        problem,
        spec,
        target_data,
        state=state,
        skip_groups=set(spec.groups),
    )
    qk_cost = (
        np.sum(ref_qk**2, axis=(1, 2))[:, None]
        + np.sum(target_qk**2, axis=(1, 2))[None, :]
        - 2.0 * np.einsum("hij,gij->hg", ref_qk, target_qk, optimize=True)
    )
    ov_cost = (
        np.sum(ref_ov**2, axis=(1, 2))[:, None]
        + np.sum(target_ov**2, axis=(1, 2))[None, :]
        - 2.0 * np.einsum("hij,gij->hg", ref_ov, target_ov, optimize=True)
    )
    return qk_cost + ov_cost


def update_attention_module(
    problem,
    objective,
    ref_data,
    target_data,
    state: PermutationState,
    spec: AttentionModuleSpec,
    *,
    scheduled_groups: set[str] | None = None,
) -> tuple[PermutationState, dict[str, Any]]:
    """One structured coordinate update for an attention module.

    Updates the head group via circuit-cost LAP, then every scheduled intra
    group via the generic exact class-dispatched update (which sees the fresh
    head permutation through the state; signed intra groups get the exact
    signed-LAP update).
    """

    from .solvers import update_group_transform

    cost = head_cost_matrix(problem, ref_data, target_data, state, spec)
    row_ind, col_ind = linear_sum_assignment(cost)
    head_matrix = np.zeros((spec.num_heads, spec.num_heads), dtype=np.float64)
    head_matrix[row_ind, col_ind] = 1.0

    previous = as_permutation_matrix(
        state.hard[spec.head_group], size=spec.num_heads, dtype=np.float64
    )
    delta = float(np.max(np.abs(head_matrix - previous)))
    state = state.with_hard({spec.head_group: head_matrix})

    intra_groups = [
        group_id
        for group_id in (*spec.qk_groups, *spec.vo_groups)
        if scheduled_groups is None or group_id in scheduled_groups
    ]
    for group_id in intra_groups:
        state, group_delta = update_group_transform(
            problem, objective, ref_data, target_data, state, group_id
        )
        delta = max(delta, group_delta)

    aux = {
        "solver": "attention",
        "head_group": spec.head_group,
        "delta": delta,
        "intra_groups": list(intra_groups),
    }
    return state, aux


@dataclass(frozen=True)
class GQAModuleSpec:
    """One grouped-query attention module's coupled groups and tensors.

    The head symmetry is the GQA quotient: kv-group permutations (size ``G``)
    times per-group query-head permutations (size ``H/G``), the wreath
    product. The per-slot qk groups are ``rotation_pairs`` groups — RoPE
    removes the qk permutation symmetry (see docs/theory.md) — and are
    updated by the generic rotation projection, not by this structured
    update.
    """

    kv_group: str
    query_head_groups: tuple[str, ...]
    qk_groups: tuple[str, ...]
    vo_groups: tuple[str, ...]
    query: str
    key: str
    value: str
    out: str
    num_kv_groups: int
    heads_per_group: int
    head_dim: int

    @classmethod
    def from_constraint(cls, problem, constraint) -> GQAModuleSpec:
        metadata = constraint.metadata
        tensors = dict(metadata["tensors"])
        return cls(
            kv_group=str(metadata["kv_group"]),
            query_head_groups=tuple(str(g) for g in metadata["query_head_groups"]),
            qk_groups=tuple(str(g) for g in metadata["qk_groups"]),
            vo_groups=tuple(str(g) for g in metadata["vo_groups"]),
            query=str(tensors["query"]),
            key=str(tensors["key"]),
            value=str(tensors["value"]),
            out=str(tensors["out"]),
            num_kv_groups=int(metadata["num_kv_groups"]),
            heads_per_group=int(metadata["heads_per_group"]),
            head_dim=int(metadata["head_dim"]),
        )

    @property
    def groups(self) -> tuple[str, ...]:
        return (
            self.kv_group,
            *self.query_head_groups,
            *self.qk_groups,
            *self.vo_groups,
        )


def gqa_module_specs(problem) -> tuple[GQAModuleSpec, ...]:
    """Return specs for every ``gqa_attention_block`` constraint of ``problem``."""

    return tuple(
        GQAModuleSpec.from_constraint(problem, constraint)
        for constraint in problem.constraints
        if constraint.kind == "gqa_attention_block"
    )


def _gqa_circuits(
    problem, spec: GQAModuleSpec, data, *, state=None, skip_groups=None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query-head QK and OV circuit matrices, shape ``(G, R, d, d)``.

    ``QK[g, r] = W^Q_{g,r} (W^K_g)^T`` is invariant to any invertible qk
    transform shared within the kv group (in particular the RoPE pair scaled
    rotations); ``OV[g, r] = W^V_g W^O_{g,r}`` is invariant to the vo
    transforms. Both are equivariant under query-head and kv-group
    permutations, which head matching resolves.
    """

    tensors = {}
    for tensor_id in (spec.query, spec.key, spec.value, spec.out):
        tensor = data[tensor_id]
        if state is not None:
            tensor = _apply_other_groups_hard(
                problem, tensor_id, tensor, state, skip_groups or set()
            )
        tensors[tensor_id] = np.asarray(tensor)

    query = tensors[spec.query]  # (d, G, R, dk)
    key = tensors[spec.key]  # (d, G, dk)
    value = tensors[spec.value]  # (d, G, dk)
    out = tensors[spec.out]  # (G, R, dk, d)
    qk = np.einsum("dgrk,egk->grde", query, key, optimize=True)
    ov = np.einsum("dgk,grke->grde", value, out, optimize=True)
    return qk, ov


def gqa_head_cost_matrix(
    problem, ref_data, target_data, state, spec: GQAModuleSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Two-level circuit cost of assigning target kv group ``j`` to ref slot ``i``.

    Returns ``(cost, inner)`` where ``cost[i, j]`` is the summed circuit
    distance under the best matching of the two groups' query heads (an inner
    R x R assignment, solved exactly per pair) and ``inner[i, j]`` is that
    matching as a permutation matrix. Groups outside the module are applied
    to the target first; the module's own groups are excluded, which is exact
    because circuit costs are invariant to the unresolved intra transforms.
    """

    ref_qk, ref_ov = _gqa_circuits(problem, spec, ref_data)
    target_qk, target_ov = _gqa_circuits(
        problem,
        spec,
        target_data,
        state=state,
        skip_groups=set(spec.groups),
    )

    def _pairwise(ref, target) -> np.ndarray:
        # (G, R, d, d) x (G, R, d, d) -> (G, R, G, R) squared distances.
        ref_sq = np.sum(ref**2, axis=(2, 3))
        target_sq = np.sum(target**2, axis=(2, 3))
        cross = np.einsum("irde,jsde->irjs", ref, target, optimize=True)
        return ref_sq[:, :, None, None] + target_sq[None, None] - 2.0 * cross

    head_cost = _pairwise(ref_qk, target_qk) + _pairwise(ref_ov, target_ov)
    num_groups, heads = spec.num_kv_groups, spec.heads_per_group
    cost = np.zeros((num_groups, num_groups), dtype=np.float64)
    inner = np.zeros((num_groups, num_groups, heads, heads), dtype=np.float64)
    for i in range(num_groups):
        for j in range(num_groups):
            row_ind, col_ind = linear_sum_assignment(head_cost[i, :, j, :])
            cost[i, j] = float(head_cost[i, :, j, :][row_ind, col_ind].sum())
            inner[i, j][row_ind, col_ind] = 1.0
    return cost, inner


def update_gqa_attention_module(
    problem,
    objective,
    ref_data,
    target_data,
    state: PermutationState,
    spec: GQAModuleSpec,
    *,
    scheduled_groups: set[str] | None = None,
) -> tuple[PermutationState, dict[str, Any]]:
    """One structured coordinate update for a grouped-query attention module.

    Updates the kv-group permutation via the two-level circuit-cost LAP (the
    inner query-head assignments enter only the cost), then updates every
    scheduled per-slot group (query-head and vo) via the generic exact
    class-dispatched update, which sees the fresh kv permutation through the
    state. The per-slot qk rotation groups are not part of this update; they
    are ordinary scheduled groups solved by the rotation projection.
    """

    from .solvers import update_group_transform

    cost, _ = gqa_head_cost_matrix(problem, ref_data, target_data, state, spec)
    row_ind, col_ind = linear_sum_assignment(cost)
    kv_matrix = np.zeros_like(cost)
    kv_matrix[row_ind, col_ind] = 1.0

    previous = as_permutation_matrix(
        state.hard[spec.kv_group], size=spec.num_kv_groups, dtype=np.float64
    )
    delta = float(np.max(np.abs(kv_matrix - previous)))
    state = state.with_hard({spec.kv_group: kv_matrix})

    intra_groups = [
        group_id
        for group_id in (*spec.query_head_groups, *spec.vo_groups)
        if scheduled_groups is None or group_id in scheduled_groups
    ]
    for group_id in intra_groups:
        state, group_delta = update_group_transform(
            problem, objective, ref_data, target_data, state, group_id
        )
        delta = max(delta, group_delta)

    aux = {
        "solver": "gqa_attention",
        "kv_group": spec.kv_group,
        "delta": delta,
        "intra_groups": list(intra_groups),
    }
    return state, aux


__all__ = [
    "AttentionModuleSpec",
    "GQAModuleSpec",
    "attention_module_specs",
    "gqa_head_cost_matrix",
    "gqa_module_specs",
    "head_cost_matrix",
    "update_attention_module",
    "update_gqa_attention_module",
]
