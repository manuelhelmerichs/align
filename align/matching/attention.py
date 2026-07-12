"""Structured attention-module updates for LAP coordinate descent.

Attention symmetries form a wreath product: an inter-head permutation shared by
q/k/v/out plus per-slot intra-head permutations (qk and vo circuits). The head
group's generic LAP linearization is inexact once intra permutations are
non-identity, so head matching uses QK/OV circuit costs, which are invariant to
intra-head permutations (see wiki/Architecture-LayerNorm-MHA-Transformer.md).
Intra groups are then updated with the generic exact LAP linearization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..symmetry import (
    GQARoPECircuitConstraint,
    MHACircuitConstraint,
    binding_axis_interval,
)
from .objectives import DiagonalFisherObjective
from .state import PermutationTransform, TransformState


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
    def from_constraint(
        cls, graph, constraint: MHACircuitConstraint
    ) -> AttentionModuleSpec:
        head_group = constraint.head_group
        qk_groups = constraint.qk_groups
        vo_groups = constraint.vo_groups
        return cls(
            head_group=head_group,
            qk_groups=qk_groups,
            vo_groups=vo_groups,
            query=constraint.query,
            key=constraint.key,
            value=constraint.value,
            out=constraint.out,
            num_heads=int(graph.groups[head_group].size),
            head_dim=int(graph.groups[qk_groups[0]].size),
        )

    @property
    def groups(self) -> tuple[str, ...]:
        return (self.head_group, *self.qk_groups, *self.vo_groups)


def attention_module_specs(graph) -> tuple[AttentionModuleSpec, ...]:
    """Return specs for every MHA circuit constraint in ``graph``."""

    return tuple(
        AttentionModuleSpec.from_constraint(graph, constraint)
        for constraint in graph.constraints
        if isinstance(constraint, MHACircuitConstraint)
    )


def _head_matrices(graph, spec, tensor_id: str, tensor) -> np.ndarray:
    """Return per-head matrices ``(H, other, dk)`` for one q/k/v/out tensor.

    ``other`` collects the remaining axes (the stream axis for the bundled
    layout); ``dk`` is the intra-head axis bound by the module's qk/vo groups.
    """

    shape = graph.tensors[tensor_id].shape
    head_axis = None
    intra_axis = None
    intra_groups = {*spec.qk_groups, *spec.vo_groups}
    for binding in graph.bindings_for_tensor(tensor_id):
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


def _projection_bias_id(graph, kernel_id: str) -> str | None:
    """Return a projection's sibling bias tensor id when the graph has one."""

    if not kernel_id.endswith("/kernel"):
        return None
    bias_id = kernel_id[: -len("kernel")] + "bias"
    return bias_id if bias_id in graph.tensors else None


def _head_factor(
    graph,
    objective,
    spec,
    data,
    tensor_id: str,
    *,
    include_bias: bool,
    state=None,
    skip_groups=None,
) -> np.ndarray:
    tensor = data[tensor_id]
    if state is not None:
        tensor = objective.apply_other_groups_hard(
            graph, tensor_id, tensor, state, skip_groups or set()
        )
    factor = _head_matrices(graph, spec, tensor_id, tensor)
    bias_id = _projection_bias_id(graph, tensor_id) if include_bias else None
    if bias_id is not None:
        bias = data[bias_id]
        if state is not None:
            bias = objective.apply_other_groups_hard(
                graph, bias_id, bias, state, skip_groups or set()
            )
        factor = np.concatenate(
            [factor, _head_matrices(graph, spec, bias_id, bias)], axis=1
        )
    return factor


def _circuits(
    graph,
    spec,
    data,
    *,
    objective=None,
    state=None,
    skip_groups: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stacked QK and OV circuit matrices, one pair per head."""

    tensors = {
        tensor_id: _head_factor(
            graph,
            objective,
            spec,
            data,
            tensor_id,
            # Query and value biases change their respective circuits.  Key
            # bias adds a key-position-constant score and cancels in softmax;
            # output bias is shared after the head sum.
            include_bias=tensor_id in (spec.query, spec.value),
            state=state,
            skip_groups=skip_groups,
        )
        for tensor_id in (spec.query, spec.key, spec.value, spec.out)
    }

    query, key = tensors[spec.query], tensors[spec.key]
    value, out = tensors[spec.value], tensors[spec.out]
    qk = np.einsum("hik,hjk->hij", query, key, optimize=True)
    ov = np.einsum("hik,hjk->hij", value, out, optimize=True)
    return qk, ov


def _quotient_circuit_precision(
    left: np.ndarray,
    right: np.ndarray,
    left_fisher: np.ndarray,
    right_fisher: np.ndarray,
) -> np.ndarray:
    """Diagonal quotient precision for the circuit ``left @ right.T``.

    Linearizing the circuit and minimizing parameter-space displacement under
    ``diag(left_fisher, right_fisher)`` induces circuit covariance
    ``J F^-1 J.T``.  Keeping its diagonal gives

    ``W[a,b]^-1 = sum_k right[b,k]^2/F_left[a,k]
                         + left[a,k]^2/F_right[b,k]``.

    The normal damped-Fisher path is strictly positive.  The explicit
    semidefinite handling below assigns zero circuit precision whenever a
    zero-Fisher parameter can change that circuit coordinate.
    """

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_fisher = np.asarray(left_fisher, dtype=np.float64)
    right_fisher = np.asarray(right_fisher, dtype=np.float64)
    if left.shape != left_fisher.shape or right.shape != right_fisher.shape:
        raise ValueError("Circuit factors and diagonal Fisher weights must match.")
    if np.any(left_fisher < 0.0) or np.any(right_fisher < 0.0):
        raise ValueError("Circuit Fisher weights must be non-negative.")

    def _variance_term(square, fisher):
        return np.where(
            fisher > 0.0,
            square / np.where(fisher > 0.0, fisher, 1.0),
            np.where(square > 0.0, np.inf, 0.0),
        )

    left_variance = _variance_term(right[None, :, :] ** 2, left_fisher[:, None, :])
    right_variance = _variance_term(left[:, None, :] ** 2, right_fisher[None, :, :])
    variance = np.sum(left_variance + right_variance, axis=-1)
    return np.where(np.isfinite(variance) & (variance > 0.0), 1.0 / variance, 0.0)


def _weighted_circuit_cost(
    reference: np.ndarray, target: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    """Directed pairwise circuit distances in a reference-slot metric."""

    ref = reference.reshape(reference.shape[0], -1)
    tgt = target.reshape(target.shape[0], -1)
    weight = precision.reshape(precision.shape[0], -1)
    ref_term = np.sum(weight * np.square(ref), axis=1)[:, None]
    target_term = weight @ np.square(tgt).T
    cross = (weight * ref) @ tgt.T
    distances = np.maximum(ref_term + target_term - 2.0 * cross, 0.0)
    scale = np.maximum(ref_term + target_term, 1.0)
    for ref_index, target_index in np.argwhere(distances <= 1e-6 * scale):
        residual = ref[ref_index] - tgt[target_index]
        distances[ref_index, target_index] = np.sum(
            weight[ref_index] * np.square(residual)
        )
    return distances


def _pairwise_squared(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pairwise squared distances without materializing broadcast residuals."""

    ref = np.asarray(reference).reshape(reference.shape[0], -1)
    tgt = np.asarray(target).reshape(target.shape[0], -1)
    distances = (
        np.sum(np.square(ref), axis=1)[:, None]
        + np.sum(np.square(tgt), axis=1)[None, :]
        - 2.0 * (ref @ tgt.T)
    )
    distances = np.maximum(distances, 0.0)
    scale = np.maximum(
        np.sum(np.square(ref), axis=1)[:, None]
        + np.sum(np.square(tgt), axis=1)[None, :],
        1.0,
    )
    for ref_index, target_index in np.argwhere(distances <= 1e-6 * scale):
        distances[ref_index, target_index] = np.sum(
            np.square(ref[ref_index] - tgt[target_index])
        )
    return distances


def _reference_attention_cache(objective) -> dict[Any, Any]:
    cache = getattr(objective, "_reference_attention_cache", None)
    if cache is None:
        cache = {}
        objective._reference_attention_cache = cache
    return cache


def _mha_circuit_precisions(graph, objective, reference_data, spec):
    factors = {
        tensor_id: _head_factor(
            graph,
            objective,
            spec,
            reference_data,
            tensor_id,
            include_bias=tensor_id in (spec.query, spec.value),
        )
        for tensor_id in (spec.query, spec.key, spec.value, spec.out)
    }
    metric_ids = {spec.query, spec.key, spec.value, spec.out}
    for tensor_id in (spec.query, spec.value):
        bias_id = _projection_bias_id(graph, tensor_id)
        if bias_id is not None:
            metric_ids.add(bias_id)
    fisher_data = {
        tensor_id: objective._weight(graph, tensor_id) for tensor_id in metric_ids
    }
    fisher = {
        tensor_id: _head_factor(
            graph,
            objective,
            spec,
            fisher_data,
            tensor_id,
            include_bias=tensor_id in (spec.query, spec.value),
        )
        for tensor_id in (spec.query, spec.key, spec.value, spec.out)
    }
    qk = np.stack(
        [
            _quotient_circuit_precision(
                factors[spec.query][head],
                factors[spec.key][head],
                fisher[spec.query][head],
                fisher[spec.key][head],
            )
            for head in range(spec.num_heads)
        ]
    )
    ov = np.stack(
        [
            _quotient_circuit_precision(
                factors[spec.value][head],
                factors[spec.out][head],
                fisher[spec.value][head],
                fisher[spec.out][head],
            )
            for head in range(spec.num_heads)
        ]
    )
    return qk, ov


def head_cost_matrix(
    graph, objective, reference_data, target_data, state, spec: AttentionModuleSpec
) -> np.ndarray:
    """Circuit-distance cost of assigning target head ``j`` to ref slot ``i``.

    All groups outside the attention module (e.g. the residual stream) are
    applied to the target first; the module's own groups are excluded, which is
    exact because circuit costs are invariant to intra-head permutations.
    """

    cache = _reference_attention_cache(objective)
    cache_key = ("mha", id(graph), objective.reference_token, spec)
    prepared = cache.get(cache_key)
    if prepared is None:
        ref_qk, ref_ov = _circuits(graph, spec, reference_data)
        precisions = (
            _mha_circuit_precisions(graph, objective, reference_data, spec)
            if isinstance(objective, DiagonalFisherObjective)
            else None
        )
        prepared = (ref_qk, ref_ov, precisions)
        cache[cache_key] = prepared
    ref_qk, ref_ov, precisions = prepared
    target_qk, target_ov = _circuits(
        graph,
        spec,
        target_data,
        objective=objective,
        state=state,
        skip_groups=set(spec.groups),
    )
    if isinstance(objective, DiagonalFisherObjective):
        qk_precision, ov_precision = precisions
        qk_cost = _weighted_circuit_cost(ref_qk, target_qk, qk_precision)
        ov_cost = _weighted_circuit_cost(ref_ov, target_ov, ov_precision)
    else:
        qk_cost = _pairwise_squared(ref_qk, target_qk)
        ov_cost = _pairwise_squared(ref_ov, target_ov)
    return qk_cost + ov_cost


def update_attention_module(
    graph,
    objective,
    reference_data,
    target_data,
    state: TransformState,
    spec: AttentionModuleSpec,
    *,
    scheduled_groups: set[str] | None = None,
    record_ambiguity: bool = False,
) -> tuple[TransformState, dict[str, Any]]:
    """One structured coordinate update for an attention module.

    Updates the head group via circuit-cost LAP, then every scheduled intra
    group via the generic exact transform-family update (which sees the fresh
    head permutation through the state; signed intra groups get the exact
    signed-LAP update).
    """

    from .solvers import (
        assignment_objective_margin,
        group_assignment_ambiguity,
        update_group_transform,
    )

    cost = head_cost_matrix(graph, objective, reference_data, target_data, state, spec)
    row_ind, col_ind = linear_sum_assignment(cost)
    head_indices = np.empty(spec.num_heads, dtype=np.int32)
    head_indices[row_ind] = col_ind
    previous = state.transforms[spec.head_group]
    delta = 0.0 if np.array_equal(previous.indices, head_indices) else 1.0
    state = state.with_transforms({spec.head_group: PermutationTransform(head_indices)})
    objective.notify_group_updated(graph, spec.head_group)
    ambiguity = {}
    if record_ambiguity:
        margin = assignment_objective_margin(-cost)
        if margin is not None:
            ambiguity[spec.head_group] = {"assignment_objective_margin": margin}

    intra_groups = [
        group_id
        for group_id in (*spec.qk_groups, *spec.vo_groups)
        if scheduled_groups is None or group_id in scheduled_groups
    ]
    for group_id in intra_groups:
        if record_ambiguity:
            group_value = group_assignment_ambiguity(
                graph, objective, reference_data, target_data, state, group_id
            )
            if group_value is not None:
                ambiguity[group_id] = group_value
        state, group_delta = update_group_transform(
            graph, objective, reference_data, target_data, state, group_id
        )
        delta = max(delta, group_delta)

    aux = {
        "solver": "attention",
        "head_group": spec.head_group,
        "delta": delta,
        "intra_groups": list(intra_groups),
    }
    if ambiguity:
        aux["ambiguity"] = ambiguity
    return state, aux


@dataclass(frozen=True)
class GQAModuleSpec:
    """One grouped-query attention module's coupled groups and tensors.

    The head symmetry is the GQA quotient: kv-group permutations (size ``G``)
    times per-group query-head permutations (size ``H/G``), the wreath
    product. The per-slot qk groups are ``rotation_pairs`` groups — RoPE
    removes the qk permutation symmetry (see the LayerNorm MHA wiki page) — and are
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
    def from_constraint(
        cls, graph, constraint: GQARoPECircuitConstraint
    ) -> GQAModuleSpec:
        return cls(
            kv_group=constraint.kv_group,
            query_head_groups=constraint.query_head_groups,
            qk_groups=constraint.qk_groups,
            vo_groups=constraint.vo_groups,
            query=constraint.query,
            key=constraint.key,
            value=constraint.value,
            out=constraint.out,
            num_kv_groups=constraint.num_kv_groups,
            heads_per_group=constraint.heads_per_group,
            head_dim=constraint.head_dim,
        )

    @property
    def groups(self) -> tuple[str, ...]:
        return (
            self.kv_group,
            *self.query_head_groups,
            *self.qk_groups,
            *self.vo_groups,
        )


def gqa_module_specs(graph) -> tuple[GQAModuleSpec, ...]:
    """Return specs for every GQA + RoPE circuit constraint in ``graph``."""

    return tuple(
        GQAModuleSpec.from_constraint(graph, constraint)
        for constraint in graph.constraints
        if isinstance(constraint, GQARoPECircuitConstraint)
    )


def _gqa_circuits(
    graph,
    spec: GQAModuleSpec,
    data,
    *,
    objective=None,
    state=None,
    skip_groups=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query-head QK and OV circuit matrices, shape ``(G, R, d, d)``.

    ``QK[g, r] = W^Q_{g,r} (W^K_g)^T`` is invariant to any invertible qk
    transform shared within the kv group (in particular the RoPE pair scaled
    rotations); ``OV[g, r] = W^V_g W^O_{g,r}`` is invariant to the vo
    transform_family. Both are equivariant under query-head and kv-group
    permutations, which head matching resolves.
    """

    tensors = {}
    for tensor_id in (spec.query, spec.key, spec.value, spec.out):
        tensor = data[tensor_id]
        if state is not None:
            tensor = objective.apply_other_groups_hard(
                graph, tensor_id, tensor, state, skip_groups or set()
            )
        tensors[tensor_id] = np.asarray(tensor)

    query = tensors[spec.query]  # (d, G, R, dk)
    key = tensors[spec.key]  # (d, G, dk)
    value = tensors[spec.value]  # (d, G, dk)
    out = tensors[spec.out]  # (G, R, dk, d)
    qk = np.einsum("dgrk,egk->grde", query, key, optimize=True)
    ov = np.einsum("dgk,grke->grde", value, out, optimize=True)
    return qk, ov


def _gqa_circuit_precisions(graph, objective, reference_data, spec):
    query = np.asarray(reference_data[spec.query])
    key = np.asarray(reference_data[spec.key])
    value = np.asarray(reference_data[spec.value])
    out = np.asarray(reference_data[spec.out])
    query_fisher = objective._weight(graph, spec.query)
    key_fisher = objective._weight(graph, spec.key)
    value_fisher = objective._weight(graph, spec.value)
    out_fisher = objective._weight(graph, spec.out)

    qk = np.empty(
        (
            spec.num_kv_groups,
            spec.heads_per_group,
            query.shape[0],
            key.shape[0],
        ),
        dtype=np.float64,
    )
    ov = np.empty_like(qk)
    # K and V are shared by every query head in a kv group.  Splitting their
    # precision evenly makes the sum of the per-query-head diagonal circuit
    # approximations use the original shared-factor metric.  Cross-head
    # covariance from that sharing is deliberately omitted by the nested LAP.
    shared_scale = float(spec.heads_per_group)
    for group in range(spec.num_kv_groups):
        for head in range(spec.heads_per_group):
            qk[group, head] = _quotient_circuit_precision(
                query[:, group, head, :],
                key[:, group, :],
                query_fisher[:, group, head, :],
                key_fisher[:, group, :] / shared_scale,
            )
            ov[group, head] = _quotient_circuit_precision(
                value[:, group, :],
                out[group, head, :, :].T,
                value_fisher[:, group, :] / shared_scale,
                out_fisher[group, head, :, :].T,
            )
    return qk, ov


def _weighted_gqa_pairwise(ref, target, precision) -> np.ndarray:
    """Directed ``(G,R,G,R)`` distances under ref-query-head precisions."""

    ref_shape = ref.shape[:2]
    target_shape = target.shape[:2]
    ref_flat = ref.reshape(-1, np.prod(ref.shape[-2:], dtype=int))
    target_flat = target.reshape(-1, np.prod(target.shape[-2:], dtype=int))
    precision_flat = precision.reshape(ref_flat.shape)
    ref_term = np.sum(precision_flat * np.square(ref_flat), axis=1)[:, None]
    target_term = precision_flat @ np.square(target_flat).T
    cross = (precision_flat * ref_flat) @ target_flat.T
    distances = np.maximum(ref_term + target_term - 2.0 * cross, 0.0)
    scale = np.maximum(ref_term + target_term, 1.0)
    for ref_index, target_index in np.argwhere(distances <= 1e-6 * scale):
        residual = ref_flat[ref_index] - target_flat[target_index]
        distances[ref_index, target_index] = np.sum(
            precision_flat[ref_index] * np.square(residual)
        )
    return distances.reshape(*ref_shape, *target_shape)


def gqa_head_cost_matrix(
    graph, objective, reference_data, target_data, state, spec: GQAModuleSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Two-level circuit cost of assigning target kv group ``j`` to ref slot ``i``.

    Returns ``(cost, inner)`` where ``cost[i, j]`` is the summed circuit
    distance under the best matching of the two groups' query heads (an inner
    R x R assignment, solved exactly per pair) and ``inner[i, j]`` is that
    matching as a permutation matrix. Groups outside the module are applied
    to the target first; the module's own groups are excluded, which is exact
    because circuit costs are invariant to the unresolved intra transform_family.
    """

    cache = _reference_attention_cache(objective)
    cache_key = ("gqa", id(graph), objective.reference_token, spec)
    prepared = cache.get(cache_key)
    if prepared is None:
        ref_qk, ref_ov = _gqa_circuits(graph, spec, reference_data)
        precisions = (
            _gqa_circuit_precisions(graph, objective, reference_data, spec)
            if isinstance(objective, DiagonalFisherObjective)
            else None
        )
        prepared = (ref_qk, ref_ov, precisions)
        cache[cache_key] = prepared
    ref_qk, ref_ov, precisions = prepared
    target_qk, target_ov = _gqa_circuits(
        graph,
        spec,
        target_data,
        objective=objective,
        state=state,
        skip_groups=set(spec.groups),
    )

    def _pairwise(ref, target) -> np.ndarray:
        shape = ref.shape[:2]
        other_shape = target.shape[:2]
        distances = _pairwise_squared(
            ref.reshape(np.prod(shape), *ref.shape[-2:]),
            target.reshape(np.prod(other_shape), *target.shape[-2:]),
        )
        return distances.reshape(*shape, *other_shape)

    if isinstance(objective, DiagonalFisherObjective):
        qk_precision, ov_precision = precisions
        head_cost = _weighted_gqa_pairwise(
            ref_qk, target_qk, qk_precision
        ) + _weighted_gqa_pairwise(ref_ov, target_ov, ov_precision)
    else:
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
    graph,
    objective,
    reference_data,
    target_data,
    state: TransformState,
    spec: GQAModuleSpec,
    *,
    scheduled_groups: set[str] | None = None,
    record_ambiguity: bool = False,
) -> tuple[TransformState, dict[str, Any]]:
    """One structured coordinate update for a grouped-query attention module.

    Updates the kv-group permutation via the two-level circuit-cost LAP (the
    inner query-head assignments enter only the cost), then updates every
    scheduled per-slot group (query-head and vo) via the generic exact
    transform-family update, which sees the fresh kv permutation through the
    state. The per-slot qk rotation groups are not part of this update; they
    are ordinary scheduled groups solved by the rotation projection.
    """

    from .solvers import (
        assignment_objective_margin,
        group_assignment_ambiguity,
        update_group_transform,
    )

    cost, _ = gqa_head_cost_matrix(
        graph, objective, reference_data, target_data, state, spec
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    kv_indices = np.empty(spec.num_kv_groups, dtype=np.int32)
    kv_indices[row_ind] = col_ind
    previous = state.transforms[spec.kv_group]
    delta = 0.0 if np.array_equal(previous.indices, kv_indices) else 1.0
    state = state.with_transforms({spec.kv_group: PermutationTransform(kv_indices)})
    objective.notify_group_updated(graph, spec.kv_group)
    ambiguity = {}
    if record_ambiguity:
        margin = assignment_objective_margin(-cost)
        if margin is not None:
            ambiguity[spec.kv_group] = {"assignment_objective_margin": margin}

    intra_groups = [
        group_id
        for group_id in (*spec.query_head_groups, *spec.vo_groups)
        if scheduled_groups is None or group_id in scheduled_groups
    ]
    for group_id in intra_groups:
        if record_ambiguity:
            group_value = group_assignment_ambiguity(
                graph, objective, reference_data, target_data, state, group_id
            )
            if group_value is not None:
                ambiguity[group_id] = group_value
        state, group_delta = update_group_transform(
            graph, objective, reference_data, target_data, state, group_id
        )
        delta = max(delta, group_delta)

    aux = {
        "solver": "gqa_rope",
        "kv_group": spec.kv_group,
        "delta": delta,
        "intra_groups": list(intra_groups),
    }
    if ambiguity:
        aux["ambiguity"] = ambiguity
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
