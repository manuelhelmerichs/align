"""Conv-stack scale canonicalization: the post-normalization affine symmetry.

For a conv channel followed by FRN/TLU (or BatchNorm/ReLU), dividing the
normalization layer's affine parameters — FRN ``(gamma, beta, tau)``, BN
``(scale, bias)`` — by a positive per-channel factor scales the channel output
by its inverse, and multiplying every consumer of the channel restores the
function *exactly*, for any ``eps`` and across residual adds. Channels of bare
convs (no norm layer) carry the ordinary ReLU incoming-scale symmetry on the
conv kernel and bias instead.

Which tensors carry the symmetry is encoded on the graph: ``out``-role
bindings with ``scale_power == 1`` are the value producers (norm affine
parameters, or bare conv kernel/bias), while pre-norm tensors (conv kernels
feeding a norm layer, FRN ``eps``, BN running stats) are bound with
``scale_power == 0`` and are exempt from the scale action.

The canonical per-channel scale is the joint energy of all producers,
``s_c = sqrt(sum_p ||p_c||^2 + eps)``. Norm-affine producers are 1-D and
unaffected by other groups' scales, so their energies are equivariant and the
normalization is one-shot for any residual topology. Bare-conv producers
consume upstream channels; their already-computed upstream scales are folded
into the kernel first (walking groups in graph order, as in the dense chain).
A bare-conv producer whose upstream group is not yet normalized (a residual
cycle of bare convs) has no one-shot canonical scale and is rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from ..symmetry import SymmetryGraph, binding_axis_interval
from ..symmetry.tensor_ops import _descend, binding_selector


def has_conv_stack_block(problem: SymmetryGraph) -> bool:
    """True if the problem contains a ``conv_stack`` component."""

    return any(
        component.kind == "conv_stack" for component in problem.components.values()
    )


def _multiply_axis_interval(
    tensor: jnp.ndarray,
    factors: jnp.ndarray,
    *,
    axis: int,
    start: int,
    stop: int,
) -> jnp.ndarray:
    broadcast = [1] * tensor.ndim
    broadcast[axis] = int(factors.shape[0])
    factors = factors.reshape(broadcast)
    if start == 0 and stop == int(tensor.shape[axis]):
        return tensor * factors
    indexer = tuple(
        slice(start, stop) if dim == axis else slice(None) for dim in range(tensor.ndim)
    )
    return tensor.at[indexer].multiply(factors)


def conv_stack_producer_scales(
    problem: SymmetryGraph,
    params: Mapping[str, Any],
    *,
    epsilon: float = 1e-8,
) -> dict[str, jnp.ndarray]:
    """Canonical per-group scales from joint producer energies.

    Walks groups in graph order, folding already-normalized upstream scales
    into bare-conv producers before measuring their energy (norm-affine
    producers are 1-D and need no folding). Degenerate channels (joint energy
    at or below ``epsilon``) keep scale 1.
    """

    scales: dict[str, jnp.ndarray] = {}
    for group_id in problem.group_order:
        group = problem.groups[group_id]
        producers = [
            binding
            for binding in problem.bindings_for_group(group_id)
            if binding.role == "out" and binding.scale_power != 0.0
        ]
        if not producers:
            raise ValueError(
                f"Conv-stack scale canonicalization found no scale-carrying "
                f"producers for group {group_id!r}; the graph marks every "
                "producer as scale-exempt."
            )
        energy = jnp.zeros((int(group.size),), dtype=jnp.float32)
        for binding in producers:
            spec = problem.tensors[binding.tensor_id]
            if binding_selector(spec.shape, binding):
                raise ValueError(
                    "Conv-stack scale canonicalization does not support "
                    f"selector-restricted producer bindings ({binding.tensor_id})."
                )
            tensor = jnp.asarray(_descend(params, spec.path))
            for other in problem.bindings_for_tensor(binding.tensor_id):
                if other.role != "in" or other.scale_power == 0.0:
                    continue
                if other.group not in scales:
                    raise ValueError(
                        f"Group {group_id!r} has a bare producer "
                        f"{binding.tensor_id!r} consuming group {other.group!r}, "
                        "which is not normalized yet (a residual cycle of "
                        "bare convs). This has no one-shot canonical scale; "
                        "attach normalization layers or disable the normalize "
                        "stage."
                    )
                o_axis, o_start, o_stop = binding_axis_interval(spec.shape, other)
                tensor = _multiply_axis_interval(
                    tensor,
                    scales[other.group] ** other.scale_power,
                    axis=o_axis,
                    start=o_start,
                    stop=o_stop,
                )
            axis, start, stop = binding_axis_interval(spec.shape, binding)
            if not (start == 0 and stop == int(spec.shape[axis])):
                indexer = tuple(
                    slice(start, stop) if dim == axis else slice(None)
                    for dim in range(tensor.ndim)
                )
                tensor = tensor[indexer]
            moved = jnp.moveaxis(tensor, axis, -1)
            energy = energy + jnp.sum(
                jnp.square(moved.reshape(-1, moved.shape[-1])), axis=0
            )
        norms = jnp.sqrt(energy + epsilon)
        scales[group_id] = jnp.where(energy <= epsilon, 1.0, norms)
    return scales


__all__ = ["conv_stack_producer_scales", "has_conv_stack_block"]
