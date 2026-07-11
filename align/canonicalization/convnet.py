"""Conv-stack scale canonicalization: post-norm affine and bare-conv symmetries.

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

Two canonical forms are implemented:

- **Balanced** (:func:`convnet_balanced_scales`, the default): the
  minimum-total-squared-norm representative of the positive scale orbit. Over
  log-scales ``x`` the orbit norm is a sum of exponentials of affine
  functions, hence convex, and its stationarity condition is per-channel
  energy balance: dividing-side energy (scale-carrying producers) equals
  multiplying-side energy (power-1 consumer bindings). The per-channel
  coordinate minimizer is ``s = ((E_div + eps) / (E_mult + eps))^(1/4)``.
  For fully norm-attached graphs both energies depend only on the group's own
  scale (consumer kernels' output axes are scale-exempt), so the objective
  separates per group and one Gauss-Seidel sweep is exact for any residual
  topology. Bare convs couple neighboring groups and the fixed point iterates;
  bare-conv residual cycles converge too (unlike the one-shot producer fold).
  A bare conv both consuming and producing the same residually tied group (a
  single-conv residual branch) couples channels *within* the group with
  matrix-balancing structure; parallel per-channel updates can oscillate
  there, so self-coupled groups take half log-steps plus an exact scalar
  line-minimization along the group's common-shift mode (under a uniform
  group shift the self-loop tensor is invariant, so the shift balances the
  remaining decoupled energies in closed form).

- **Producer energy** (:func:`convnet_producer_scales`,
  ``strategy: unit_norm``): unit joint producer energy per channel,
  ``s_c = sqrt(sum_p ||p_c||^2 + eps)``. Norm-affine producers are 1-D and
  unaffected by other groups' scales, so the normalization is one-shot for
  any residual topology. Bare-conv producers consume upstream channels; their
  already-computed upstream scales are folded into the kernel first (walking
  groups in graph order, as in the dense chain). A bare-conv producer whose
  upstream group is not yet canonicalized (a residual cycle of bare convs)
  has no one-shot canonical scale and is rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..symmetry import SymmetryGraph, binding_axis_intervals
from ..symmetry.tensor_ops import _descend, binding_selector


def has_convnet_component(graph: SymmetryGraph) -> bool:
    """True if the graph contains a ``convnet`` component."""

    return any(component.kind == "convnet" for component in graph.components.values())


def _multiply_axis_interval(
    tensor: np.ndarray,
    factors: np.ndarray,
    *,
    axis: int,
    start: int,
    stop: int,
) -> np.ndarray:
    broadcast = [1] * tensor.ndim
    broadcast[axis] = int(factors.shape[0])
    factors = factors.reshape(broadcast)
    if start == 0 and stop == int(tensor.shape[axis]):
        return tensor * factors
    indexer = tuple(
        slice(start, stop) if dim == axis else slice(None) for dim in range(tensor.ndim)
    )
    result = np.array(tensor, copy=True)
    result[indexer] *= factors
    return result


def convnet_producer_scales(
    graph: SymmetryGraph,
    params: Mapping[str, Any],
    *,
    epsilon: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Canonical per-group scales from joint producer energies.

    Walks groups in graph order, folding already-canonicalized upstream scales
    into bare-conv producers before measuring their energy (norm-affine
    producers are 1-D and need no folding). Degenerate channels (joint energy
    at or below ``epsilon``) keep scale 1.
    """

    scales: dict[str, np.ndarray] = {}
    for group_id in graph.group_order:
        group = graph.groups[group_id]
        producers = [
            binding
            for binding in graph.bindings_for_group(group_id)
            if binding.role == "out" and binding.scale_power != 0.0
        ]
        if not producers:
            raise ValueError(
                f"Conv-stack scale canonicalization found no scale-carrying "
                f"producers for group {group_id!r}; the graph marks every "
                "producer as scale-exempt."
            )
        energy = None
        for binding in producers:
            spec = graph.tensors[binding.tensor_id]
            if binding_selector(spec.shape, binding):
                raise ValueError(
                    "Conv-stack scale canonicalization does not support "
                    f"selector-restricted producer bindings ({binding.tensor_id})."
                )
            tensor = np.asarray(_descend(params, spec.path))
            if energy is None:
                energy = np.zeros((int(group.size),), dtype=tensor.dtype)
            for other in graph.bindings_for_tensor(binding.tensor_id):
                if other.role != "in" or other.scale_power == 0.0:
                    continue
                if other.group not in scales:
                    raise ValueError(
                        f"Group {group_id!r} has a bare producer "
                        f"{binding.tensor_id!r} consuming group {other.group!r}, "
                        "which is not canonicalized yet (a residual cycle of "
                        "bare convs). This has no one-shot canonical scale; "
                        "use strategy='balanced' or disable the canonicalize "
                        "stage."
                    )
                for o_axis, o_start, o_stop in binding_axis_intervals(
                    spec.shape, other
                ):
                    tensor = _multiply_axis_interval(
                        tensor,
                        scales[other.group] ** other.scale_power,
                        axis=o_axis,
                        start=o_start,
                        stop=o_stop,
                    )
            for axis, start, stop in binding_axis_intervals(spec.shape, binding):
                segment = tensor
                if not (start == 0 and stop == int(spec.shape[axis])):
                    indexer = tuple(
                        slice(start, stop) if dim == axis else slice(None)
                        for dim in range(tensor.ndim)
                    )
                    segment = tensor[indexer]
                moved = np.moveaxis(segment, axis, -1)
                energy = energy + np.sum(
                    np.square(moved.reshape(-1, moved.shape[-1])), axis=0
                )
        assert energy is not None
        norms = np.sqrt(energy + epsilon)
        scales[group_id] = np.where(energy <= epsilon, 1.0, norms).astype(
            energy.dtype, copy=False
        )
    return scales


@dataclass(frozen=True)
class _BalanceBinding:
    tensor_id: str
    axis: int
    start: int
    stop: int
    role: str


def _slice_energy(array: np.ndarray, binding: _BalanceBinding) -> np.ndarray:
    """Per-channel sum of squares over the binding's axis interval."""

    indexer = tuple(
        slice(binding.start, binding.stop) if dim == binding.axis else slice(None)
        for dim in range(array.ndim)
    )
    moved = np.moveaxis(array[indexer], binding.axis, -1)
    return np.sum(moved.reshape(-1, moved.shape[-1]) ** 2, axis=0)


def _scale_slice(
    array: np.ndarray, factors: np.ndarray, binding: _BalanceBinding
) -> None:
    indexer = tuple(
        slice(binding.start, binding.stop) if dim == binding.axis else slice(None)
        for dim in range(array.ndim)
    )
    shape = [1] * array.ndim
    shape[binding.axis] = factors.shape[0]
    factors = factors.reshape(shape)
    if binding.role == "out":
        array[indexer] = array[indexer] / factors
    else:
        array[indexer] = array[indexer] * factors


def _balance_plan(
    graph: SymmetryGraph, params: Mapping[str, Any]
) -> tuple[
    dict[str, np.ndarray], dict[str, list[_BalanceBinding]], dict[str, np.dtype]
]:
    """Materialize scale-carrying tensors and per-group power-1 bindings."""

    arrays: dict[str, np.ndarray] = {}
    group_bindings: dict[str, list[_BalanceBinding]] = {
        group_id: [] for group_id in graph.group_order
    }
    group_dtypes: dict[str, np.dtype] = {}
    for tensor_id, spec in graph.tensors.items():
        bindings = [
            binding
            for binding in graph.bindings_for_tensor(tensor_id)
            if binding.scale_power != 0.0
        ]
        if not bindings:
            continue
        arrays[tensor_id] = np.asarray(
            _descend(params, spec.path), dtype=np.float64
        ).copy()
        for binding in bindings:
            if binding.scale_power != 1.0:
                raise ValueError(
                    "Balanced conv-stack canonicalization requires scale powers "
                    f"0 or 1, but {tensor_id!r} binds group {binding.group!r} "
                    f"with power {binding.scale_power}."
                )
            if binding_selector(spec.shape, binding):
                raise ValueError(
                    "Balanced conv-stack canonicalization does not support "
                    f"selector-restricted bindings ({tensor_id})."
                )
            group_dtypes.setdefault(
                binding.group, np.asarray(_descend(params, spec.path)).dtype
            )
            for axis, start, stop in binding_axis_intervals(spec.shape, binding):
                group_bindings[binding.group].append(
                    _BalanceBinding(tensor_id, axis, start, stop, binding.role)
                )
    return arrays, group_bindings, group_dtypes


def convnet_balanced_scales(
    graph: SymmetryGraph,
    params: Mapping[str, Any],
    *,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Minimum-norm per-group scales: balance dividing and multiplying energies.

    Gauss-Seidel sweeps over groups in graph order; per channel the update is
    ``s = ((E_div + eps) / (E_mult + eps))^(1/4)`` with the dividing side the
    scale-carrying producers and the multiplying side the power-1 consumer
    bindings. Fully norm-attached graphs converge in one sweep (the energies
    decouple); bare-conv chains and residual cycles iterate to the unique
    minimum of the convex orbit-norm objective. Self-coupled groups (a tensor
    carrying both in- and out-role bindings of the group) take half log-steps
    plus an exact scalar common-shift minimization to avoid the parallel
    matrix-balancing oscillation. Degenerate channels (raw dividing-side
    energy at or below ``epsilon``) keep scale 1.
    """

    arrays, group_bindings, group_dtypes = _balance_plan(graph, params)

    self_loop_tensors: dict[str, set[str]] = {}
    masks: dict[str, np.ndarray] = {}
    for group_id in graph.group_order:
        bindings = group_bindings[group_id]
        roles: dict[str, set[str]] = {}
        for binding in bindings:
            roles.setdefault(binding.tensor_id, set()).add(binding.role)
        self_loop_tensors[group_id] = {
            tensor_id
            for tensor_id, tensor_roles in roles.items()
            if tensor_roles == {"in", "out"}
        }
        if not any(binding.role == "out" for binding in bindings):
            raise ValueError(
                f"Conv-stack scale canonicalization found no scale-carrying "
                f"producers for group {group_id!r}; the graph marks every "
                "producer as scale-exempt."
            )
        if not any(binding.role == "in" for binding in bindings):
            raise ValueError(
                f"Balanced conv-stack canonicalization requires group "
                f"{group_id!r} to have at least one scale-carrying consumer; "
                "with none, the minimum-norm representative does not exist "
                "(the orbit norm decreases without bound). Use "
                "strategy='unit_norm' instead."
            )
        raw_div = np.zeros(int(graph.groups[group_id].size))
        for binding in bindings:
            if binding.role == "out":
                raw_div += _slice_energy(arrays[binding.tensor_id], binding)
        masks[group_id] = raw_div <= epsilon

    totals = {
        group_id: np.ones(int(graph.groups[group_id].size))
        for group_id in graph.group_order
    }
    for _ in range(max_iter):
        delta = 0.0
        for group_id in graph.group_order:
            bindings = group_bindings[group_id]
            mask = masks[group_id]
            energy_div = np.zeros(mask.shape[0])
            energy_mult = np.zeros(mask.shape[0])
            for binding in bindings:
                energy = _slice_energy(arrays[binding.tensor_id], binding)
                if binding.role == "out":
                    energy_div += energy
                else:
                    energy_mult += energy
            self_coupled = bool(self_loop_tensors[group_id])
            exponent = 0.125 if self_coupled else 0.25
            factors = ((energy_div + epsilon) / (energy_mult + epsilon)) ** exponent
            factors = np.where(mask, 1.0, factors)
            for binding in bindings:
                _scale_slice(arrays[binding.tensor_id], factors, binding)
            totals[group_id] *= factors
            delta = max(delta, float(np.max(np.abs(np.log(factors)))))

            if self_coupled and not bool(np.any(mask)):
                # Exact line minimization along the group's common-shift
                # mode: the self-loop tensor is invariant under a uniform
                # shift, so the shift balances the remaining energies.
                shift_div = epsilon
                shift_mult = epsilon
                for binding in bindings:
                    if binding.tensor_id in self_loop_tensors[group_id]:
                        continue
                    energy = float(
                        np.sum(_slice_energy(arrays[binding.tensor_id], binding))
                    )
                    if binding.role == "out":
                        shift_div += energy
                    else:
                        shift_mult += energy
                shift = (shift_div / shift_mult) ** 0.25
                shift_factors = np.full(mask.shape[0], shift)
                for binding in bindings:
                    _scale_slice(arrays[binding.tensor_id], shift_factors, binding)
                totals[group_id] *= shift_factors
                delta = max(delta, abs(float(np.log(shift))))
        if delta < tol:
            break
    else:
        raise ValueError(
            f"Balanced conv-stack canonicalization did not converge within "
            f"{max_iter} sweeps (last max |log factor| {delta:.3e})."
        )

    return {
        group_id: np.asarray(total, dtype=group_dtypes[group_id])
        for group_id, total in totals.items()
    }


__all__ = [
    "convnet_balanced_scales",
    "convnet_producer_scales",
    "has_convnet_component",
]
