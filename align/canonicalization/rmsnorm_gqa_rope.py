"""RMSNorm/RoPE/GQA scale canonicalization: gamma, qk pairs, and vo circuits.

Three exact diagonal symmetries coexist in the RMSNorm + RoPE + GQA family
(derivations in wiki/Architecture-RMSNorm-GQA-RoPE-Transformer.md):

- **RMSNorm post-norm scale.** ``RMSNorm(x) = gamma * x / rms(x)`` depends on
  ``gamma`` only linearly, so per channel ``gamma_c -> gamma_c / s_c`` with
  every consumer kernel row multiplied by ``s_c`` preserves the function
  exactly for any nonzero ``s_c`` (including negative) and any ``eps`` — the
  transformer analogue of the FRN/BN post-norm affine symmetry, plus a sign
  freedom the affine norms lack. Canonicalized one-shot by *signed* producer
  energy, ``s_c = sign(gamma_c) * sqrt(gamma_c^2 + eps)``, folding gamma to
  ``+1``: this removes both the scale and the sign freedom (consumers absorb
  the signs) and is the precondition for orthogonal stream alignment. The
  residual stream itself carries no per-channel scale symmetry (rescaling a
  stream channel changes ``rms``), so these scales live on typed RMSNorm-scale
  constraints, not on the stream group; they are applied by a dedicated
  helper because the touched axes are also stream-bound.

- **RoPE qk pair scale.** Rotary embeddings restrict the qk circuit symmetry
  to per-frequency-pair scaled rotations ``alpha_p R(phi_p)``; the removable
  scale is pair-tied over the half-split pairs ``(p, p + dk/2)`` and shared
  across a kv group's query heads. Canonicalized by balancing the pair
  energies, ``s = ((E_q + eps) / (E_k + eps))^{1/4}`` (query divided, key
  multiplied), applied through the qk ``rotation_pairs`` groups' role
  bindings via :meth:`SymmetryGraph.apply_scales` with pair-tiled
  vectors. The rotation *angle* is matched by the matching rotation
  projection, not canonicalized here.

- **vo circuit scale.** As in flat attention: ``(W^V) D^{-1}`` with
  ``D W^O`` per intra-head dimension, per kv group (shared across the group's
  query heads). Canonicalized by balancing value/out energies through the vo
  groups' role bindings.

The gamma helper is also used to construct synthetic scale orbits, so
canonicalization and orbit construction cannot drift apart; qk/vo orbits use
the ordinary ``ScaleState`` graph action.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from ..symmetry import (
    GQARoPECircuitConstraint,
    RMSNormScaleConstraint,
    SymmetryGraph,
)
from ..symmetry.tensor_ops import _descend
from ._tree import replace_paths


def rmsnorm_scale_constraints(
    graph: SymmetryGraph,
) -> tuple[RMSNormScaleConstraint, ...]:
    """Return the graph's RMSNorm scale constraints."""

    return tuple(
        constraint
        for constraint in graph.constraints
        if isinstance(constraint, RMSNormScaleConstraint)
    )


def gqa_rope_circuit_constraints(
    graph: SymmetryGraph,
) -> tuple[GQARoPECircuitConstraint, ...]:
    """Return the graph's GQA + RoPE circuit constraints."""

    return tuple(
        constraint
        for constraint in graph.constraints
        if isinstance(constraint, GQARoPECircuitConstraint)
    )


def has_rmsnorm_gqa_rope_transformer_plan(graph: SymmetryGraph) -> bool:
    """True if the graph carries RMSNorm or GQA attention constraints."""

    return bool(rmsnorm_scale_constraints(graph)) or bool(
        gqa_rope_circuit_constraints(graph)
    )


def _multiply_axis(tensor: Any, factors: Any, *, axis: int) -> Any:
    factors = jnp.asarray(factors)
    broadcast = [1] * jnp.ndim(tensor)
    broadcast[axis] = int(factors.shape[0])
    return jnp.asarray(tensor) * factors.reshape(broadcast)


def rms_gamma_scales(
    graph: SymmetryGraph,
    params: Mapping[str, Any],
    *,
    epsilon: float = 1e-8,
) -> dict[str, jnp.ndarray]:
    """Canonical per-channel scales keyed by RMSNorm scale tensor id.

    One-shot *signed* producer energy,
    ``s_c = sign(gamma_c) * sqrt(gamma_c^2 + eps)``: dividing gamma maps it
    to ``+1`` (up to ``eps``), removing the scale and the sign freedom (the
    signs move into the consumers). Degenerate channels (energy at or below
    ``eps``) keep scale 1, as in the conv plan.
    """

    scales: dict[str, jnp.ndarray] = {}
    for constraint in rmsnorm_scale_constraints(graph):
        scale_id = constraint.scale
        gamma = jnp.asarray(_descend(params, graph.tensors[scale_id].path))
        energy = jnp.square(gamma)
        signs = jnp.where(gamma < 0.0, -1.0, 1.0)
        scales[scale_id] = jnp.where(
            energy <= epsilon, 1.0, signs * jnp.sqrt(energy + epsilon)
        )
    return scales


def apply_rms_gamma_scales(
    graph: SymmetryGraph,
    params: Mapping[str, Any],
    scales: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Apply the exact RMSNorm post-norm scale symmetry.

    For each RMSNorm scale constraint whose tensor id appears in
    ``scales``: divide gamma by ``s`` and multiply every recorded consumer
    kernel along its stream input axis by ``s``.
    """

    replacements: list[tuple[tuple[str, ...], Any]] = []
    for constraint in rmsnorm_scale_constraints(graph):
        scale_id = constraint.scale
        if scale_id not in scales:
            continue
        factors = jnp.asarray(scales[scale_id])
        gamma_path = graph.tensors[scale_id].path
        gamma = jnp.asarray(_descend(params, gamma_path))
        replacements.append((gamma_path, gamma / factors))
        for tensor_id, axis in constraint.consumers:
            path = graph.tensors[str(tensor_id)].path
            tensor = jnp.asarray(_descend(params, path))
            replacements.append((path, _multiply_axis(tensor, factors, axis=int(axis))))
    return replace_paths(params, replacements) if replacements else params


def _pair_tiled(factors: jnp.ndarray) -> jnp.ndarray:
    """Tile per-pair factors ``(..., dk/2)`` to the half-split axis ``(..., dk)``."""

    return jnp.concatenate([factors, factors], axis=-1)


def qk_pair_scales(
    graph: SymmetryGraph,
    params: Mapping[str, Any],
    *,
    epsilon: float = 1e-8,
) -> dict[str, jnp.ndarray]:
    """Balancing pair-tiled scales ``(head_dim,)`` keyed by qk group id.

    Per pair, ``s = ((E_q + eps) / (E_k + eps))^{1/4}`` with the query energy
    summed over stream and query heads and the key energy over the stream;
    the qk bindings' roles (query ``out`` divided, key ``in`` multiplied)
    equalize the two pair energies through
    :meth:`SymmetryGraph.apply_scales`. Pair-constant factors commute with
    the rotary rotations, so the action is exact under RoPE.
    """

    scales: dict[str, jnp.ndarray] = {}
    for constraint in gqa_rope_circuit_constraints(graph):
        tensors = {
            role: getattr(constraint, role) for role in ("query", "key", "value", "out")
        }
        half = constraint.head_dim // 2
        query = jnp.asarray(_descend(params, graph.tensors[tensors["query"]].path))
        key = jnp.asarray(_descend(params, graph.tensors[tensors["key"]].path))
        query_sq = jnp.square(query)
        key_sq = jnp.square(key)
        # (d, G, R, dk) -> (G, dk/2); (d, G, dk) -> (G, dk/2)
        query_energy = jnp.sum(query_sq[..., :half] + query_sq[..., half:], axis=(0, 2))
        key_energy = jnp.sum(key_sq[..., :half] + key_sq[..., half:], axis=0)
        factors = ((query_energy + epsilon) / (key_energy + epsilon)) ** 0.25
        for slot, group_id in enumerate(constraint.qk_groups):
            scales[group_id] = _pair_tiled(factors[slot])
    return scales


def gqa_vo_balancing_scales(
    graph: SymmetryGraph,
    params: Mapping[str, Any],
    *,
    epsilon: float = 1e-8,
) -> dict[str, jnp.ndarray]:
    """Per-vo-group balancing scales, as in flat attention circuit balancing.

    For kv group ``g`` and intra-head dimension ``i`` the factor is
    ``s = ((E_v + eps) / (E_o + eps))^{1/4}`` with the value energy summed
    over the stream axis and the out energy over query heads and the stream
    axis; dividing the value side and multiplying the out side equalizes the
    two. Applied through ``SymmetryGraph.apply_scales`` (value ``out``
    role, out-kernel ``in`` role).
    """

    scales: dict[str, jnp.ndarray] = {}
    for constraint in gqa_rope_circuit_constraints(graph):
        tensors = {
            role: getattr(constraint, role) for role in ("query", "key", "value", "out")
        }
        value = jnp.asarray(_descend(params, graph.tensors[tensors["value"]].path))
        out = jnp.asarray(_descend(params, graph.tensors[tensors["out"]].path))
        value_energy = jnp.sum(jnp.square(value), axis=0)  # (G, dk)
        out_energy = jnp.sum(jnp.square(out), axis=(1, 3))  # (G, dk)
        factors = ((value_energy + epsilon) / (out_energy + epsilon)) ** 0.25
        for slot, group_id in enumerate(constraint.vo_groups):
            scales[group_id] = factors[slot]
    return scales


__all__ = [
    "apply_rms_gamma_scales",
    "gqa_rope_circuit_constraints",
    "gqa_vo_balancing_scales",
    "has_rmsnorm_gqa_rope_transformer_plan",
    "qk_pair_scales",
    "rms_gamma_scales",
    "rmsnorm_scale_constraints",
]
