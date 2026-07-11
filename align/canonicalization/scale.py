"""Graph-driven scale canonicalization.

The :class:`ScaleCanonicalizer` is the deterministic canonicalization counterpart of
the matching solver stack: it reads producer/consumer axis roles off an
:class:`~align.symmetry.SymmetryGraph`, computes positive per-group scale
vectors, and applies the symmetry through
:meth:`align.symmetry.SymmetryGraph.apply_scales`. Degenerate-neuron cleanup
remains an explicit post-pass because it is not a positive scale symmetry.

Plans are dispatched on graph structure:

- **RMSNorm/RoPE/GQA canonicalization** for graphs with typed RMSNorm-scale or
  GQA + RoPE circuit constraints: RMSNorm
  post-norm gamma scales are folded to unit producer energy, RoPE qk pair
  scales and vo circuit scales are balanced (see
  ``align.canonicalization.rmsnorm_gqa_rope``).
- **Attention circuit balancing** for graphs with MHA circuit constraints
  (LayerNorm transformers): the exact diagonal qk/vo circuit
  symmetry is canonicalized by equalizing query/key and value/out energies
  per intra-head dimension. This symmetry is activation-independent; all
  other groups (stream, heads, GELU FFN hidden units) carry no per-channel
  scale symmetry and stay at identity.
- **Conv-stack canonicalization** for graphs with a ``convnet`` component
  (plain and residual conv nets). Two canonical forms are available via
  ``strategy``: ``balanced`` (default) picks the minimum-norm orbit
  representative by equalizing each channel's dividing-side (scale-carrying
  producers: norm affine parameters or bare conv kernel/bias) and
  multiplying-side (power-1 consumer bindings) energies — one-shot for
  norm-attached graphs, a convergent convex fixed point for bare-conv chains
  and residual cycles — while ``unit_norm`` folds the joint producer energy
  to 1 per channel (see ``align.canonicalization.convnet``).
- **Dense chain canonicalization** for linear ReLU-MLP graphs (each permutation
  group has a single ``out``-role producer kernel, kernels are 2-D, and the
  groups form a single chain feeding one output head). Residual ties and
  branches give a group multiple ``out``-role producers and are rejected;
  convolutional kernels are rejected as non-dense. Two canonical forms are
  available via ``strategy``: ``balanced`` (default) picks the minimum-norm orbit
  representative by equalizing each hidden unit's incoming and outgoing
  energies (a convex fixed point, the same philosophy as attention circuit
  balancing), while ``unit_norm`` normalizes incoming norms to 1
  (Erdogan-style). ``balanced`` keeps canonical weights near the data scale;
  ``unit_norm`` can inflate posteriors whose true unit norms are far from 1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..symmetry import AxisBinding, SymmetryGraph, binding_axis_interval
from ..symmetry.tensor_ops import _descend
from ._tree import replace_paths
from .activations import is_positive_homogeneous, validate_activation
from .convnet import (
    convnet_balanced_scales,
    convnet_producer_scales,
    has_convnet_component,
)
from .dense import (
    DenseLayer,
    _canonicalize_degenerate,
    _outgoing_scales,
    _require_bool,
    _validate_degenerate_channels,
    compute_degenerate_mask,
    compute_incoming_norms,
)
from .mha import attention_balancing_scales, mha_circuit_constraints
from .rmsnorm_gqa_rope import (
    apply_rms_gamma_scales,
    gqa_vo_balancing_scales,
    has_rmsnorm_gqa_rope_transformer_plan,
    qk_pair_scales,
    rms_gamma_scales,
)
from .state import ScaleState

ParamTree = Mapping[str, Any]


@dataclass(frozen=True)
class _DenseScaleSite:
    group_id: str
    kernel_path: tuple[str, ...]
    bias_path: tuple[str, ...]
    in_binding: AxisBinding | None


@dataclass(frozen=True)
class _DenseScalePlan:
    sites: tuple[_DenseScaleSite, ...]
    head_kernel_path: tuple[str, ...]


def _dense_scale_plan(graph: SymmetryGraph) -> _DenseScalePlan:
    """Validate and summarize the supported linear dense scale graph."""

    tensors = graph.tensors
    if not graph.group_order:
        raise ValueError(
            "Scale canonicalization requires at least one hidden permutation group."
        )
    sites: list[_DenseScaleSite] = []
    producer_kernel_ids: set[str] = set()
    previous_group: str | None = None

    for group_id in graph.group_order:
        out_bindings = [
            binding
            for binding in graph.bindings_for_group(group_id)
            if binding.role == "out"
        ]
        kernel_ids = list(
            dict.fromkeys(
                binding.tensor_id
                for binding in out_bindings
                if len(tensors[binding.tensor_id].shape) >= 2
            )
        )
        bias_ids = list(
            dict.fromkeys(
                binding.tensor_id
                for binding in out_bindings
                if len(tensors[binding.tensor_id].shape) == 1
            )
        )
        if not kernel_ids:
            raise ValueError(
                f"Scale canonicalization requires group {group_id!r} to have an "
                "out-role producer kernel, but none was found."
            )
        if len(kernel_ids) > 1:
            raise ValueError(
                f"Scale canonicalization is undefined for group {group_id!r}: it has "
                f"{len(kernel_ids)} out-role producers ({', '.join(kernel_ids)}). "
                "Residual ties and branches need explicit shared-scale semantics; "
                "drop the canonicalize stage for this architecture."
            )
        kernel_id = kernel_ids[0]
        kernel_shape = tensors[kernel_id].shape
        if len(kernel_shape) != 2:
            raise ValueError(
                f"Scale canonicalization only supports 2-D dense kernels, but the "
                f"producer for group {group_id!r} ({kernel_id}) has shape "
                f"{kernel_shape}. Convolutional / non-dense architectures are "
                "unsupported; drop the canonicalize stage."
            )
        kernel_out_bindings = [
            binding for binding in out_bindings if binding.tensor_id == kernel_id
        ]
        if len(kernel_out_bindings) != 1:
            raise ValueError(
                f"Scale canonicalization requires group {group_id!r} to bind exactly "
                f"one producer output axis on {kernel_id}, found "
                f"{len(kernel_out_bindings)}."
            )
        out_axis, _, _ = binding_axis_interval(kernel_shape, kernel_out_bindings[0])
        if out_axis != 1:
            raise ValueError(
                "Scale canonicalization expects dense producer kernels with the "
                f"group on axis 1, but {kernel_id} binds group {group_id!r} on "
                f"axis {out_axis}."
            )
        if len(bias_ids) != 1:
            raise ValueError(
                f"Scale canonicalization requires exactly one out-role bias for group "
                f"{group_id!r}, found {len(bias_ids)}."
            )
        bias_id = bias_ids[0]
        bias_bindings = [
            binding for binding in out_bindings if binding.tensor_id == bias_id
        ]
        if len(bias_bindings) != 1:
            raise ValueError(
                f"Scale canonicalization requires group {group_id!r} to bind exactly "
                f"one producer bias axis on {bias_id}, found {len(bias_bindings)}."
            )
        in_bindings = [
            binding
            for binding in graph.bindings_for_tensor(kernel_id)
            if binding.role == "in"
        ]
        in_binding: AxisBinding | None = None
        if previous_group is not None:
            if len(in_bindings) != 1 or in_bindings[0].group != previous_group:
                raise ValueError(
                    "Scale canonicalization requires a single linear chain, but the "
                    f"producer for group {group_id!r} does not consume the previous "
                    f"group {previous_group!r}."
                )
            in_axis, _, _ = binding_axis_interval(kernel_shape, in_bindings[0])
            if in_axis != 0:
                raise ValueError(
                    "Scale canonicalization expects dense producer kernels with the "
                    f"previous group on axis 0, but {kernel_id} binds group "
                    f"{previous_group!r} on axis {in_axis}."
                )
            in_binding = in_bindings[0]
        elif in_bindings:
            raise ValueError(
                "Scale canonicalization requires a single linear chain, but the first "
                f"producer for group {group_id!r} consumes another hidden group."
            )
        producer_kernel_ids.add(kernel_id)
        sites.append(
            _DenseScaleSite(
                group_id=group_id,
                kernel_path=tensors[kernel_id].path,
                bias_path=tensors[bias_id].path,
                in_binding=in_binding,
            )
        )
        previous_group = group_id

    last_group = graph.group_order[-1]
    head_ids = list(
        dict.fromkeys(
            binding.tensor_id
            for binding in graph.bindings_for_group(last_group)
            if binding.role == "in"
            and binding.tensor_id not in producer_kernel_ids
            and len(tensors[binding.tensor_id].shape) == 2
        )
    )
    if len(head_ids) != 1:
        raise ValueError(
            "Scale canonicalization requires exactly one output head consuming the "
            f"last group {last_group!r}, found {len(head_ids)}: {head_ids}."
        )
    head_path = tensors[head_ids[0]].path
    return _DenseScalePlan(
        sites=tuple(sites),
        head_kernel_path=head_path,
    )


def _multiply_axis(tensor: Any, factors: Any, *, axis: int) -> Any:
    xp = jnp if isinstance(tensor, jnp.ndarray) else np
    factor = xp.asarray(factors)
    broadcast_shape = [1] * xp.ndim(tensor)
    broadcast_shape[axis] = factor.shape[0]
    return tensor * factor.reshape(broadcast_shape)


def _has_degenerate(mask: Any) -> bool:
    return bool(np.any(np.asarray(mask)))


def _group_scales(
    params: ParamTree,
    plan: _DenseScalePlan,
    *,
    epsilon: float,
    degenerate_channels: str,
    include_bias_in_norm: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive per-group scales by folding each group's scale into the next kernel.

    The action scale for a group is the incoming norm of its producer *after* the
    previous group's outgoing scale has been folded into the kernel, so the chain
    must be walked in order. ``action_scales`` (degenerate neurons clamped to 1)
    drive the positive :meth:`apply_scales` symmetry; ``outgoing_scales`` (which
    may be 0 for degenerate neurons) are the consumer-side factors propagated
    forward and later realised by the degenerate post-passes.
    """

    action_scales: dict[str, Any] = {}
    outgoing_scales: dict[str, Any] = {}
    reported_scales: dict[str, Any] = {}
    degenerate_masks: dict[str, Any] = {}

    for site in plan.sites:
        kernel = jnp.asarray(_descend(params, site.kernel_path))
        bias = jnp.asarray(_descend(params, site.bias_path))
        if site.in_binding is not None:
            in_axis, _, _ = binding_axis_interval(kernel.shape, site.in_binding)
            kernel = _multiply_axis(
                kernel,
                outgoing_scales[site.in_binding.group],
                axis=in_axis,
            )

        layer = DenseLayer(kernel=kernel, bias=bias)
        norms = compute_incoming_norms(
            layer, epsilon=epsilon, include_bias_in_norm=include_bias_in_norm
        )
        mask = compute_degenerate_mask(
            layer, epsilon=epsilon, include_bias_in_norm=include_bias_in_norm
        )

        action_scales[site.group_id] = _outgoing_scales(
            norms, mask, degenerate_channels="preserve"
        )
        outgoing_scales[site.group_id] = _outgoing_scales(
            norms, mask, degenerate_channels=degenerate_channels
        )
        reported_scales[site.group_id] = norms
        degenerate_masks[site.group_id] = mask

    return action_scales, reported_scales, degenerate_masks


def _balanced_group_scales(
    params: ParamTree,
    plan: _DenseScalePlan,
    *,
    epsilon: float,
    include_bias_in_norm: bool,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive per-group scales that equalize incoming and outgoing energies.

    Minimizing the total squared weight norm over the positive scale orbit is
    convex in the log-scales; per-group coordinate descent (divide the
    producer column and bias by ``s``, multiply the consumer row by ``s``,
    with the balanced update ``s = ((E_out + eps) / (E_in + eps))^(1/4)``)
    converges to the unique minimum-norm representative. Degenerate units
    (mask from the raw incoming energy, as in ``unit_norm`` strategy) keep scale 1
    so the post-passes see the same contract.
    """

    mats = [
        np.asarray(_descend(params, site.kernel_path), dtype=np.float64).copy()
        for site in plan.sites
    ]
    biases = [
        np.asarray(_descend(params, site.bias_path), dtype=np.float64).copy()
        for site in plan.sites
    ]
    mats.append(
        np.asarray(_descend(params, plan.head_kernel_path), dtype=np.float64).copy()
    )

    degenerate_masks: dict[str, Any] = {}
    for site, kernel, bias in zip(plan.sites, mats[:-1], biases, strict=True):
        layer = DenseLayer(kernel=jnp.asarray(kernel), bias=jnp.asarray(bias))
        degenerate_masks[site.group_id] = compute_degenerate_mask(
            layer, epsilon=epsilon, include_bias_in_norm=include_bias_in_norm
        )

    totals = [np.ones(kernel.shape[1], dtype=np.float64) for kernel in mats[:-1]]
    masks = [np.asarray(degenerate_masks[site.group_id]) for site in plan.sites]
    for _ in range(max_iter):
        delta = 0.0
        for index in range(len(plan.sites)):
            energy_out = np.sum(mats[index] ** 2, axis=0)
            if include_bias_in_norm:
                energy_out = energy_out + biases[index] ** 2
            energy_in = np.sum(mats[index + 1] ** 2, axis=1)
            factors = ((energy_out + epsilon) / (energy_in + epsilon)) ** 0.25
            factors = np.where(masks[index], 1.0, factors)
            mats[index] /= factors[None, :]
            biases[index] /= factors
            mats[index + 1] *= factors[:, None]
            totals[index] *= factors
            delta = max(delta, float(np.max(np.abs(np.log(factors)))))
        if delta < tol:
            break

    action_scales = {
        site.group_id: jnp.asarray(total, dtype=jnp.float32)
        for site, total in zip(plan.sites, totals, strict=True)
    }
    reported_scales = {
        site.group_id: action_scales[site.group_id] for site in plan.sites
    }
    return action_scales, reported_scales, degenerate_masks


def _zero_degenerate_consumers(
    graph: SymmetryGraph,
    params: ParamTree,
    degenerate_masks: Mapping[str, Any],
) -> ParamTree:
    active_masks = {
        group_id: jnp.where(mask, 0.0, 1.0)
        for group_id, mask in degenerate_masks.items()
        if _has_degenerate(mask)
    }
    if not active_masks:
        return params

    def _mask(segment, axis, binding):
        # Zero scales are cleanup, not a valid positive ScaleState action.
        if binding.role != "in" or binding.group not in active_masks:
            return segment
        return _multiply_axis(segment, active_masks[binding.group], axis=axis)

    return graph._transform_bound_tensors(params, _mask)


def _canonicalize_degenerate_producers(
    params: ParamTree,
    plan: _DenseScalePlan,
    degenerate_masks: Mapping[str, Any],
) -> ParamTree:
    replacements: list[tuple[tuple[str, ...], Any]] = []
    for site in plan.sites:
        mask = degenerate_masks[site.group_id]
        if not _has_degenerate(mask):
            continue
        mask = jnp.asarray(mask, dtype=bool)
        kernel = jnp.asarray(_descend(params, site.kernel_path))
        bias = jnp.asarray(_descend(params, site.bias_path))
        kernel, bias = _canonicalize_degenerate(kernel, bias, mask)
        replacements.append((site.kernel_path, kernel))
        replacements.append((site.bias_path, bias))
    return replace_paths(params, replacements) if replacements else params


class ScaleCanonicalizer:
    """Deterministic graph-native scale canonicalization."""

    def canonicalize(
        self,
        graph: SymmetryGraph,
        params: ParamTree,
        **canonicalizer_kwargs: Any,
    ) -> tuple[ParamTree, dict[str, Any], dict[str, Any]]:
        """Return ``(canonical_params, scales, diagnostics)`` for ``params``."""

        canonicalizer_kwargs = dict(canonicalizer_kwargs)
        epsilon = float(canonicalizer_kwargs.pop("epsilon", 1e-8))
        degenerate_channels = _validate_degenerate_channels(
            canonicalizer_kwargs.pop("degenerate_channels", "preserve")
        )
        include_bias_in_norm = _require_bool(
            "include_bias_in_norm",
            canonicalizer_kwargs.pop("include_bias_in_norm", True),
        )
        activation = validate_activation(
            canonicalizer_kwargs.pop("activation", "relu"), None
        )
        strategy = canonicalizer_kwargs.pop("strategy", None)
        if strategy is not None and strategy not in ("balanced", "unit_norm"):
            raise ValueError(
                f"Unknown canonicalize strategy {strategy!r}. Available: "
                "balanced, unit_norm."
            )
        if canonicalizer_kwargs:
            unexpected = next(iter(canonicalizer_kwargs))
            raise TypeError(
                "ScaleCanonicalizer.canonicalize() got an unexpected keyword argument "
                f"{unexpected!r}"
            )

        if has_rmsnorm_gqa_rope_transformer_plan(graph):
            if strategy is not None:
                raise ValueError(
                    "The RMSNorm/RoPE/GQA plan has one canonical form (gamma "
                    "producer energy plus qk-pair/vo balancing); the 'strategy' "
                    "option is undefined for it. Omit the option."
                )
            return self._canonicalize_rmsnorm_gqa_rope_transformer(
                graph,
                params,
                epsilon=epsilon,
                degenerate_channels=degenerate_channels,
                activation=activation,
                include_bias_in_norm=include_bias_in_norm,
            )

        if mha_circuit_constraints(graph):
            if strategy == "unit_norm":
                raise ValueError(
                    "strategy='unit_norm' is undefined for attention circuit "
                    "canonicalization: circuits are canonicalized by balancing "
                    "(the minimum-norm representative). Use strategy='balanced' "
                    "or omit the option."
                )
            return self._canonicalize_attention_circuits(
                graph,
                params,
                epsilon=epsilon,
                include_bias_in_norm=include_bias_in_norm,
                degenerate_channels=degenerate_channels,
                activation=activation,
            )

        if has_convnet_component(graph):
            return self._canonicalize_convnet(
                graph,
                params,
                strategy=strategy or "balanced",
                epsilon=epsilon,
                include_bias_in_norm=include_bias_in_norm,
                degenerate_channels=degenerate_channels,
                activation=activation,
            )

        if not is_positive_homogeneous(activation):
            raise ValueError(
                f"Dense-chain canonicalization requires a positively homogeneous "
                f"activation, got {activation!r}. Non-homogeneous hidden units "
                "carry no scale symmetry; drop the canonicalize stage for this "
                "architecture."
            )
        dense_strategy = strategy or "balanced"
        plan = _dense_scale_plan(graph)
        if dense_strategy == "balanced":
            action_scales, scales, degenerate_masks = _balanced_group_scales(
                params,
                plan,
                epsilon=epsilon,
                include_bias_in_norm=include_bias_in_norm,
            )
        else:
            action_scales, scales, degenerate_masks = _group_scales(
                params,
                plan,
                epsilon=epsilon,
                degenerate_channels=degenerate_channels,
                include_bias_in_norm=include_bias_in_norm,
            )
        canonical_params = graph.apply_scales(
            params,
            ScaleState.from_scales(graph, action_scales, backend="jax"),
        )

        if degenerate_channels in ("zero_outgoing", "canonical_vector"):
            canonical_params = _zero_degenerate_consumers(
                graph, canonical_params, degenerate_masks
            )
        if degenerate_channels == "canonical_vector":
            canonical_params = _canonicalize_degenerate_producers(
                canonical_params, plan, degenerate_masks
            )

        aux: dict[str, Any] = {
            "plan": "dense_chain",
            "strategy": dense_strategy,
            "epsilon": epsilon,
            "degenerate_channels": degenerate_channels,
            "include_bias_in_norm": include_bias_in_norm,
            "activation": activation,
        }
        return canonical_params, scales, aux

    def _canonicalize_convnet(
        self,
        graph: SymmetryGraph,
        params: ParamTree,
        *,
        strategy: str,
        epsilon: float,
        include_bias_in_norm: bool,
        degenerate_channels: str,
        activation: str,
    ) -> tuple[ParamTree, dict[str, Any], dict[str, Any]]:
        """Canonicalize conv-stack scales (norm-affine or bare-conv producers).

        ``balanced`` (default) picks the minimum-norm orbit representative by
        equalizing each channel's dividing-side (producer) and
        multiplying-side (consumer) energies; ``unit_norm`` folds the joint
        producer energy to 1 per channel.
        """

        if not is_positive_homogeneous(activation):
            raise ValueError(
                f"Conv-stack canonicalization requires a positively homogeneous "
                f"activation (relu, leaky_relu, tlu), got {activation!r}."
            )
        if degenerate_channels != "preserve":
            raise ValueError(
                f"degenerate_channels={degenerate_channels!r} is not defined for "
                "conv-stack canonicalization; degenerate channels keep scale 1 "
                "('preserve')."
            )
        if not include_bias_in_norm:
            raise ValueError(
                "include_bias_in_norm=false is not defined for conv-stack "
                "canonicalization; the canonical scale measures the joint energy "
                "of all scale-carrying producers (norm affine parameters or conv "
                "kernel plus bias)."
            )

        if strategy == "balanced":
            scales = convnet_balanced_scales(graph, params, epsilon=epsilon)
            plan = "conv_balanced"
        else:
            scales = convnet_producer_scales(graph, params, epsilon=epsilon)
            plan = "conv_producer_energy"
        canonical_params = graph.apply_scales(
            params,
            ScaleState.from_scales(graph, scales, backend="jax"),
        )
        aux: dict[str, Any] = {
            "plan": plan,
            "strategy": strategy,
            "epsilon": epsilon,
            "degenerate_channels": degenerate_channels,
            "include_bias_in_norm": include_bias_in_norm,
            "activation": activation,
        }
        return canonical_params, scales, aux

    def _canonicalize_rmsnorm_gqa_rope_transformer(
        self,
        graph: SymmetryGraph,
        params: ParamTree,
        *,
        epsilon: float,
        degenerate_channels: str,
        activation: str,
        include_bias_in_norm: bool,
    ) -> tuple[ParamTree, dict[str, Any], dict[str, Any]]:
        """RMSNorm gamma folding, qk pair balancing, and vo circuit balancing.

        The three exact diagonal symmetries of the RMSNorm + RoPE + GQA
        family, applied in a fixed order (gamma, qk pairs, vo) so the
        composed canonical form is deterministic. GELU FFN hidden units and
        the residual stream carry no per-channel scale symmetry and stay at
        identity.
        """

        if degenerate_channels != "preserve":
            raise ValueError(
                f"degenerate_channels={degenerate_channels!r} has no meaning "
                "for RMSNorm/RoPE/GQA canonicalization (degenerate gamma "
                "channels keep scale 1); use 'preserve'."
            )

        gamma_scales = rms_gamma_scales(graph, params, epsilon=epsilon)
        canonical = apply_rms_gamma_scales(graph, params, gamma_scales)
        pair_scales = qk_pair_scales(graph, canonical, epsilon=epsilon)
        vo_scales = gqa_vo_balancing_scales(graph, canonical, epsilon=epsilon)
        circuit_scales = {**pair_scales, **vo_scales}
        if circuit_scales:
            canonical = graph.apply_scales(
                canonical,
                ScaleState.from_scales(graph, circuit_scales, backend="jax"),
            )

        scales = {
            **gamma_scales,
            **{
                group_id: circuit_scales[group_id]
                for group_id in graph.group_order
                if group_id in circuit_scales
            },
        }
        aux: dict[str, Any] = {
            "plan": "rmsnorm_gqa_rope",
            "strategy": "balanced",
            "epsilon": epsilon,
            "degenerate_channels": degenerate_channels,
            "include_bias_in_norm": include_bias_in_norm,
            "activation": activation,
            "rmsnorm_scales": sorted(gamma_scales),
            "balanced_qk_groups": [
                gid for gid in graph.group_order if gid in pair_scales
            ],
            "balanced_vo_groups": [
                gid for gid in graph.group_order if gid in vo_scales
            ],
        }
        return canonical, scales, aux

    def _canonicalize_attention_circuits(
        self,
        graph: SymmetryGraph,
        params: ParamTree,
        *,
        epsilon: float,
        include_bias_in_norm: bool,
        degenerate_channels: str,
        activation: str,
    ) -> tuple[ParamTree, dict[str, Any], dict[str, Any]]:
        """Balance qk/vo circuit scales; all other groups stay at identity."""

        if degenerate_channels != "preserve":
            raise ValueError(
                f"degenerate_channels={degenerate_channels!r} has no meaning for "
                "attention circuit balancing (the epsilon-stabilized balance is "
                "already defined for zero circuits); use 'preserve'."
            )

        scales = attention_balancing_scales(
            graph,
            params,
            epsilon=epsilon,
            include_bias_in_norm=include_bias_in_norm,
        )
        canonical_params = graph.apply_scales(
            params,
            ScaleState.from_scales(graph, scales, backend="jax"),
        )
        aux: dict[str, Any] = {
            "plan": "attention_balance",
            "strategy": "balanced",
            "epsilon": epsilon,
            "degenerate_channels": degenerate_channels,
            "include_bias_in_norm": include_bias_in_norm,
            "activation": activation,
            "balanced_groups": [
                group_id for group_id in graph.group_order if group_id in scales
            ],
        }
        ordered_scales = {
            group_id: scales[group_id]
            for group_id in graph.group_order
            if group_id in scales
        }
        return canonical_params, ordered_scales, aux


__all__ = ["ScaleCanonicalizer"]
