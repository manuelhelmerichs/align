"""Attention circuit scale canonicalization (qk / vo balancing).

Attention scores and outputs depend on the projection kernels only through the
per-head circuits ``QK_i = W_i^Q (W_i^K)^T`` and ``OV_i = W_i^V W_i^O``. For any
positive diagonal ``D`` (one factor per intra-head dimension),

    (W^Q, b^Q) -> (W^Q, b^Q) D^{-1},   (W^K, b^K) -> (W^K, b^K) D,
    (W^V, b^V) -> (W^V, b^V) D^{-1},   W^O -> D W^O

is an exact function-preserving symmetry, independent of the surrounding
activations. The canonical representative chosen here *balances* each circuit:
per intra-head dimension, the query and key (resp. value and out) energies are
equalized, which is the minimum-norm point of the diagonal-scale orbit.

The balancing factors are per-group :class:`ScaleState` vectors for the
``qk``/``vo`` intra-head groups emitted by
:class:`~align.architectures.blocks.AttentionBlockAdapter`, whose binding roles
encode the divide/multiply pairing (query/value ``out``, key/out-kernel
``in``). Application therefore goes through the ordinary
:meth:`AlignmentProblem.apply_scales` graph action.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from ..alignment import AlignmentProblem, GraphConstraint
from ..alignment.tensor_ops import _descend


def attention_constraints(problem: AlignmentProblem) -> tuple[GraphConstraint, ...]:
    """Return the problem's ``attention_block`` constraints."""

    return tuple(
        constraint
        for constraint in problem.constraints
        if constraint.kind == "attention_block"
    )


def _bias_energy(
    problem: AlignmentProblem,
    params: Mapping[str, Any],
    kernel_tensor_id: str,
) -> jnp.ndarray | None:
    """Squared per-``(head, dim)`` bias energy for a q/k/v kernel, if present."""

    kernel_path = problem.tensors[kernel_tensor_id].path
    bias_path = (*kernel_path[:-1], "bias")
    bias_id = "/".join(bias_path)
    if bias_id not in problem.tensors:
        return None
    return jnp.square(jnp.asarray(_descend(params, bias_path)))


def _projection_energy(
    problem: AlignmentProblem,
    params: Mapping[str, Any],
    tensor_id: str,
    *,
    normalize_biases: bool,
) -> jnp.ndarray:
    """Squared per-``(head, dim)`` energy of one q/k/v kernel (+ bias)."""

    kernel = jnp.asarray(_descend(params, problem.tensors[tensor_id].path))
    energy = jnp.sum(jnp.square(kernel), axis=0)
    if normalize_biases:
        bias_energy = _bias_energy(problem, params, tensor_id)
        if bias_energy is not None:
            energy = energy + bias_energy
    return energy


def attention_balancing_scales(
    problem: AlignmentProblem,
    params: Mapping[str, Any],
    *,
    epsilon: float = 1e-8,
    normalize_biases: bool = True,
) -> dict[str, jnp.ndarray]:
    """Compute per-group balancing scales for every attention block.

    For each head slot ``h`` and intra-head dimension ``i`` the qk factor is
    ``s = ((E_q + eps) / (E_k + eps))^{1/4}`` with squared energies ``E``;
    dividing the query side by ``s`` and multiplying the key side equalizes
    the two energies. Same for vo with the out kernel's row energies. The
    factors are equivariant under the diagonal symmetry (up to ``epsilon``),
    so scale-equivalent samples collapse to one representative.
    """

    scales: dict[str, jnp.ndarray] = {}
    for constraint in attention_constraints(problem):
        metadata = constraint.metadata
        tensor_ids = dict(metadata["tensors"])
        query_energy = _projection_energy(
            problem, params, tensor_ids["query"], normalize_biases=normalize_biases
        )
        key_energy = _projection_energy(
            problem, params, tensor_ids["key"], normalize_biases=normalize_biases
        )
        value_energy = _projection_energy(
            problem, params, tensor_ids["value"], normalize_biases=normalize_biases
        )
        out_kernel = jnp.asarray(
            _descend(params, problem.tensors[tensor_ids["out"]].path)
        )
        out_energy = jnp.sum(jnp.square(out_kernel), axis=-1)

        qk_factors = ((query_energy + epsilon) / (key_energy + epsilon)) ** 0.25
        vo_factors = ((value_energy + epsilon) / (out_energy + epsilon)) ** 0.25
        for slot, group_id in enumerate(metadata["qk_groups"]):
            scales[group_id] = qk_factors[slot]
        for slot, group_id in enumerate(metadata["vo_groups"]):
            scales[group_id] = vo_factors[slot]
    return scales


__all__ = ["attention_balancing_scales", "attention_constraints"]
