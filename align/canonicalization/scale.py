"""Graph-driven scale normalization.

The :class:`ScaleCanonicalizer` is the deterministic canonicalization counterpart of
the rebasin solver stack: it reads producer/consumer axis roles off an
:class:`~align.symmetry.SymmetryGraph`, computes positive per-group scale
vectors, and applies the symmetry through
:meth:`align.symmetry.SymmetryGraph.apply_scales`. Degenerate-neuron cleanup
and optional classification-head rescaling remain explicit post-passes because
they are not positive scale symmetries.

Plans are dispatched on problem structure:

- **Modern-transformer canonicalization** for problems with ``rms_norm`` or
  ``gqa_attention_block`` constraints (RMSNorm + RoPE + GQA stacks): RMSNorm
  post-norm gamma scales are folded to unit producer energy, RoPE qk pair
  scales and vo circuit scales are balanced (see
  ``align.canonicalization.rmsnorm_gqa_rope``).
- **Attention circuit balancing** for problems with ``attention_block``
  constraints (LayerNorm transformers): the exact diagonal qk/vo circuit
  symmetry is canonicalized by equalizing query/key and value/out energies
  per intra-head dimension. This symmetry is activation-independent; all
  other groups (stream, heads, GELU FFN hidden units) carry no per-channel
  scale symmetry and stay at identity.
- **Dense chain normalization** for linear ReLU-MLP graphs (each permutation
  group has a single ``out``-role producer kernel, kernels are 2-D, and the
  groups form a single chain feeding one output head). Residual ties and
  branches give a group multiple ``out``-role producers and are rejected;
  convolutional kernels are rejected as non-dense. Two canonical forms are
  available via ``mode``: ``balanced`` (default) picks the minimum-norm orbit
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
from .convnet import conv_stack_producer_scales, has_conv_stack_block
from .dense import DenseLayer
from .kernel import (
    _canonicalize_degenerate,
    _outgoing_scale_factors,
    _require_bool,
    _validate_degenerate_handling,
    compute_degenerate_mask,
    compute_incoming_norms,
    normalize_last_layer_classification,
)
from .mha import attention_balancing_scales, attention_constraints
from .rmsnorm_gqa_rope import (
    apply_rms_gamma_scales,
    gqa_vo_balancing_scales,
    has_modern_transformer_plan,
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
    head_bias_path: tuple[str, ...]


def _dense_scale_plan(problem: SymmetryGraph) -> _DenseScalePlan:
    """Validate and summarize the supported linear dense scale graph."""

    tensors = problem.tensors
    if not problem.group_order:
        raise ValueError(
            "Scale normalization requires at least one hidden permutation group."
        )
    sites: list[_DenseScaleSite] = []
    producer_kernel_ids: set[str] = set()
    previous_group: str | None = None

    for group_id in problem.group_order:
        out_bindings = [
            binding
            for binding in problem.bindings_for_group(group_id)
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
                f"Scale normalization requires group {group_id!r} to have an "
                "out-role producer kernel, but none was found."
            )
        if len(kernel_ids) > 1:
            raise ValueError(
                f"Scale normalization is undefined for group {group_id!r}: it has "
                f"{len(kernel_ids)} out-role producers ({', '.join(kernel_ids)}). "
                "Residual ties and branches need explicit shared-scale semantics; "
                "disable the normalize stage for this architecture."
            )
        kernel_id = kernel_ids[0]
        kernel_shape = tensors[kernel_id].shape
        if len(kernel_shape) != 2:
            raise ValueError(
                f"Scale normalization only supports 2-D dense kernels, but the "
                f"producer for group {group_id!r} ({kernel_id}) has shape "
                f"{kernel_shape}. Convolutional / non-dense architectures are "
                "unsupported; disable the normalize stage."
            )
        kernel_out_bindings = [
            binding for binding in out_bindings if binding.tensor_id == kernel_id
        ]
        if len(kernel_out_bindings) != 1:
            raise ValueError(
                f"Scale normalization requires group {group_id!r} to bind exactly "
                f"one producer output axis on {kernel_id}, found "
                f"{len(kernel_out_bindings)}."
            )
        out_axis, _, _ = binding_axis_interval(kernel_shape, kernel_out_bindings[0])
        if out_axis != 1:
            raise ValueError(
                "Scale normalization expects dense producer kernels with the "
                f"group on axis 1, but {kernel_id} binds group {group_id!r} on "
                f"axis {out_axis}."
            )
        if len(bias_ids) != 1:
            raise ValueError(
                f"Scale normalization requires exactly one out-role bias for group "
                f"{group_id!r}, found {len(bias_ids)}."
            )
        bias_id = bias_ids[0]
        bias_bindings = [
            binding for binding in out_bindings if binding.tensor_id == bias_id
        ]
        if len(bias_bindings) != 1:
            raise ValueError(
                f"Scale normalization requires group {group_id!r} to bind exactly "
                f"one producer bias axis on {bias_id}, found {len(bias_bindings)}."
            )
        in_bindings = [
            binding
            for binding in problem.bindings_for_tensor(kernel_id)
            if binding.role == "in"
        ]
        in_binding: AxisBinding | None = None
        if previous_group is not None:
            if len(in_bindings) != 1 or in_bindings[0].group != previous_group:
                raise ValueError(
                    "Scale normalization requires a single linear chain, but the "
                    f"producer for group {group_id!r} does not consume the previous "
                    f"group {previous_group!r}."
                )
            in_axis, _, _ = binding_axis_interval(kernel_shape, in_bindings[0])
            if in_axis != 0:
                raise ValueError(
                    "Scale normalization expects dense producer kernels with the "
                    f"previous group on axis 0, but {kernel_id} binds group "
                    f"{previous_group!r} on axis {in_axis}."
                )
            in_binding = in_bindings[0]
        elif in_bindings:
            raise ValueError(
                "Scale normalization requires a single linear chain, but the first "
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

    last_group = problem.group_order[-1]
    head_ids = list(
        dict.fromkeys(
            binding.tensor_id
            for binding in problem.bindings_for_group(last_group)
            if binding.role == "in"
            and binding.tensor_id not in producer_kernel_ids
            and len(tensors[binding.tensor_id].shape) == 2
        )
    )
    if len(head_ids) != 1:
        raise ValueError(
            "Scale normalization requires exactly one output head consuming the "
            f"last group {last_group!r}, found {len(head_ids)}: {head_ids}."
        )
    head_path = tensors[head_ids[0]].path
    return _DenseScalePlan(
        sites=tuple(sites),
        head_kernel_path=head_path,
        head_bias_path=(*head_path[:-1], "bias"),
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
    degenerate_handling: str,
    normalize_biases: bool,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
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
    reported_scales: list[Any] = []
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
            layer, epsilon=epsilon, normalize_biases=normalize_biases
        )
        mask = compute_degenerate_mask(
            layer, epsilon=epsilon, normalize_biases=normalize_biases
        )

        action_scales[site.group_id] = _outgoing_scale_factors(
            norms, mask, degenerate_handling="preserve"
        )
        outgoing_scales[site.group_id] = _outgoing_scale_factors(
            norms, mask, degenerate_handling=degenerate_handling
        )
        reported_scales.append(norms)
        degenerate_masks[site.group_id] = mask

    return action_scales, reported_scales, degenerate_masks


def _balanced_group_scales(
    params: ParamTree,
    plan: _DenseScalePlan,
    *,
    epsilon: float,
    normalize_biases: bool,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    """Derive per-group scales that equalize incoming and outgoing energies.

    Minimizing the total squared weight norm over the positive scale orbit is
    convex in the log-scales; per-group coordinate descent (divide the
    producer column and bias by ``s``, multiply the consumer row by ``s``,
    with the balanced update ``s = ((E_out + eps) / (E_in + eps))^(1/4)``)
    converges to the unique minimum-norm representative. Degenerate units
    (mask from the raw incoming energy, as in ``unit_norm`` mode) keep scale 1
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
            layer, epsilon=epsilon, normalize_biases=normalize_biases
        )

    totals = [np.ones(kernel.shape[1], dtype=np.float64) for kernel in mats[:-1]]
    masks = [np.asarray(degenerate_masks[site.group_id]) for site in plan.sites]
    for _ in range(max_iter):
        delta = 0.0
        for index in range(len(plan.sites)):
            energy_out = np.sum(mats[index] ** 2, axis=0)
            if normalize_biases:
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
    reported_scales = [action_scales[site.group_id] for site in plan.sites]
    return action_scales, reported_scales, degenerate_masks


def _zero_degenerate_consumers(
    problem: SymmetryGraph,
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

    return problem._transform_bound_tensors(params, _mask)


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


def _rescale_classification_head(
    params: ParamTree,
    plan: _DenseScalePlan,
    *,
    num_classes: int,
    epsilon: float,
) -> tuple[ParamTree, Any]:
    head = DenseLayer(
        kernel=jnp.asarray(_descend(params, plan.head_kernel_path)),
        bias=jnp.asarray(_descend(params, plan.head_bias_path)),
    )
    normalized_head, gamma = normalize_last_layer_classification(
        head,
        int(num_classes),
        epsilon=epsilon,
    )
    return (
        replace_paths(
            params,
            (
                (plan.head_kernel_path, normalized_head.kernel),
                (plan.head_bias_path, normalized_head.bias),
            ),
        ),
        gamma,
    )


class ScaleCanonicalizer:
    """Deterministic graph-native scale canonicalization."""

    def normalize(
        self,
        problem: SymmetryGraph,
        params: ParamTree,
        *,
        task_type: str = "regression",
        num_classes: int | None = None,
        **method_kwargs: Any,
    ) -> tuple[ParamTree, list[Any], dict[str, Any]]:
        """Return ``(normalized_params, scale_factors, aux)`` for ``params``."""

        method_kwargs = dict(method_kwargs)
        epsilon = float(method_kwargs.pop("epsilon", 1e-8))
        degenerate_handling = _validate_degenerate_handling(
            method_kwargs.pop("degenerate_handling", "preserve")
        )
        normalize_biases = _require_bool(
            "normalize_biases", method_kwargs.pop("normalize_biases", True)
        )
        activation = validate_activation(
            method_kwargs.pop("activation", "relu"),
            method_kwargs.pop("activation_kwargs", None),
        )
        classification_head_rescale = _require_bool(
            "classification_head_rescale",
            method_kwargs.pop("classification_head_rescale", False),
        )
        mode = method_kwargs.pop("mode", None)
        if mode is not None and mode not in ("balanced", "unit_norm"):
            raise ValueError(
                f"Unknown scale normalization mode {mode!r}. Available: "
                "balanced, unit_norm."
            )
        if method_kwargs:
            unexpected = next(iter(method_kwargs))
            raise TypeError(
                "ScaleCanonicalizer.normalize() got an unexpected keyword argument "
                f"{unexpected!r}"
            )

        task = (task_type or "regression").lower()
        if task not in ("regression", "classification"):
            raise ValueError("task_type must be 'regression' or 'classification'")
        if task == "classification" and classification_head_rescale:
            if num_classes is None:
                raise ValueError(
                    "num_classes required when classification_head_rescale is enabled"
                )

        if has_modern_transformer_plan(problem):
            if mode is not None:
                raise ValueError(
                    "The modern-transformer plan has one canonical form "
                    "(gamma producer energy plus qk-pair/vo balancing); the "
                    "'mode' option is undefined for it. Omit the option."
                )
            return self._normalize_modern_transformer(
                problem,
                params,
                epsilon=epsilon,
                degenerate_handling=degenerate_handling,
                classification_head_rescale=classification_head_rescale,
                task=task,
                activation=activation,
                normalize_biases=normalize_biases,
            )

        if attention_constraints(problem):
            if mode == "unit_norm":
                raise ValueError(
                    "mode='unit_norm' is undefined for attention circuit "
                    "normalization: circuits are canonicalized by balancing "
                    "(the minimum-norm representative). Use mode='balanced' or "
                    "omit the option."
                )
            return self._normalize_attention_circuits(
                problem,
                params,
                epsilon=epsilon,
                normalize_biases=normalize_biases,
                degenerate_handling=degenerate_handling,
                classification_head_rescale=classification_head_rescale,
                task=task,
                activation=activation,
            )

        if has_conv_stack_block(problem):
            if mode == "balanced":
                raise ValueError(
                    "mode='balanced' is not implemented for conv-stack "
                    "normalization; the conv plan canonicalizes by unit joint "
                    "producer energy. Use mode='unit_norm' or omit the option."
                )
            return self._normalize_conv_stack(
                problem,
                params,
                epsilon=epsilon,
                normalize_biases=normalize_biases,
                degenerate_handling=degenerate_handling,
                classification_head_rescale=classification_head_rescale,
                task=task,
                activation=activation,
            )

        if not is_positive_homogeneous(activation):
            raise ValueError(
                f"Dense-chain scale normalization requires a positively "
                f"homogeneous activation, got {activation!r}. Non-homogeneous "
                "hidden units carry no scale symmetry; disable the normalize "
                "stage for this architecture."
            )
        dense_mode = mode or "balanced"
        plan = _dense_scale_plan(problem)
        if dense_mode == "balanced":
            action_scales, scale_factors, degenerate_masks = _balanced_group_scales(
                params,
                plan,
                epsilon=epsilon,
                normalize_biases=normalize_biases,
            )
        else:
            action_scales, scale_factors, degenerate_masks = _group_scales(
                params,
                plan,
                epsilon=epsilon,
                degenerate_handling=degenerate_handling,
                normalize_biases=normalize_biases,
            )
        normalized_params = problem.apply_scales(
            params,
            ScaleState.from_scales(problem, action_scales, backend="jax"),
        )

        if degenerate_handling in ("zero_outgoing", "canonical_vector"):
            normalized_params = _zero_degenerate_consumers(
                problem, normalized_params, degenerate_masks
            )
        if degenerate_handling == "canonical_vector":
            normalized_params = _canonicalize_degenerate_producers(
                normalized_params, plan, degenerate_masks
            )

        aux: dict[str, Any] = {
            "plan": "dense_chain",
            "mode": dense_mode,
            "task_type": task,
            "epsilon": epsilon,
            "degenerate_handling": degenerate_handling,
            "normalize_biases": normalize_biases,
            "activation": activation,
            "classification_head_rescale": classification_head_rescale,
        }
        if task == "classification" and classification_head_rescale:
            normalized_params, gamma = _rescale_classification_head(
                normalized_params,
                plan,
                num_classes=int(num_classes),
                epsilon=epsilon,
            )
            aux["last_layer_gamma"] = gamma
            aux["num_classes"] = int(num_classes)

        return normalized_params, scale_factors, aux

    def _normalize_conv_stack(
        self,
        problem: SymmetryGraph,
        params: ParamTree,
        *,
        epsilon: float,
        normalize_biases: bool,
        degenerate_handling: str,
        classification_head_rescale: bool,
        task: str,
        activation: str,
    ) -> tuple[ParamTree, list[Any], dict[str, Any]]:
        """Normalize joint producer energies (norm-affine or bare-conv)."""

        if not is_positive_homogeneous(activation):
            raise ValueError(
                f"Conv-stack scale normalization requires a positively "
                f"homogeneous activation (relu, leaky_relu, tlu), got "
                f"{activation!r}."
            )
        if classification_head_rescale:
            raise ValueError(
                "classification_head_rescale is not supported for conv-stack "
                "normalization; it is a dense-chain post-pass."
            )
        if degenerate_handling != "preserve":
            raise ValueError(
                f"degenerate_handling={degenerate_handling!r} is not defined for "
                "conv-stack normalization; degenerate channels keep scale 1 "
                "('preserve')."
            )
        if not normalize_biases:
            raise ValueError(
                "normalize_biases=false is not defined for conv-stack "
                "normalization; the canonical scale is the joint energy of all "
                "scale-carrying producers (norm affine parameters or conv "
                "kernel plus bias)."
            )

        scales = conv_stack_producer_scales(problem, params, epsilon=epsilon)
        normalized_params = problem.apply_scales(
            params,
            ScaleState.from_scales(problem, scales, backend="jax"),
        )
        scale_factors = [scales[group_id] for group_id in problem.group_order]
        aux: dict[str, Any] = {
            "plan": "conv_producer_energy",
            "mode": "unit_norm",
            "task_type": task,
            "epsilon": epsilon,
            "degenerate_handling": degenerate_handling,
            "normalize_biases": normalize_biases,
            "activation": activation,
            "classification_head_rescale": classification_head_rescale,
        }
        return normalized_params, scale_factors, aux

    def _normalize_modern_transformer(
        self,
        problem: SymmetryGraph,
        params: ParamTree,
        *,
        epsilon: float,
        degenerate_handling: str,
        classification_head_rescale: bool,
        task: str,
        activation: str,
        normalize_biases: bool,
    ) -> tuple[ParamTree, list[Any], dict[str, Any]]:
        """RMSNorm gamma folding, qk pair balancing, and vo circuit balancing.

        The three exact diagonal symmetries of the RMSNorm + RoPE + GQA
        family, applied in a fixed order (gamma, qk pairs, vo) so the
        composed canonical form is deterministic. GELU FFN hidden units and
        the residual stream carry no per-channel scale symmetry and stay at
        identity.
        """

        if classification_head_rescale:
            raise ValueError(
                "classification_head_rescale is not supported for "
                "modern-transformer normalization; it is a dense-chain "
                "post-pass."
            )
        if degenerate_handling != "preserve":
            raise ValueError(
                f"degenerate_handling={degenerate_handling!r} has no meaning "
                "for modern-transformer normalization (degenerate gamma "
                "channels keep scale 1); use 'preserve'."
            )

        gamma_scales = rms_gamma_scales(problem, params, epsilon=epsilon)
        normalized = apply_rms_gamma_scales(problem, params, gamma_scales)
        pair_scales = qk_pair_scales(problem, normalized, epsilon=epsilon)
        vo_scales = gqa_vo_balancing_scales(problem, normalized, epsilon=epsilon)
        circuit_scales = {**pair_scales, **vo_scales}
        if circuit_scales:
            normalized = problem.apply_scales(
                normalized,
                ScaleState.from_scales(problem, circuit_scales, backend="jax"),
            )

        scale_factors = [
            *gamma_scales.values(),
            *(
                circuit_scales[gid]
                for gid in problem.group_order
                if gid in circuit_scales
            ),
        ]
        aux: dict[str, Any] = {
            "plan": "modern_transformer",
            "mode": "balanced",
            "task_type": task,
            "epsilon": epsilon,
            "degenerate_handling": degenerate_handling,
            "normalize_biases": normalize_biases,
            "activation": activation,
            "classification_head_rescale": classification_head_rescale,
            "rms_norms": sorted(gamma_scales),
            "balanced_qk_groups": [
                gid for gid in problem.group_order if gid in pair_scales
            ],
            "balanced_vo_groups": [
                gid for gid in problem.group_order if gid in vo_scales
            ],
        }
        return normalized, scale_factors, aux

    def _normalize_attention_circuits(
        self,
        problem: SymmetryGraph,
        params: ParamTree,
        *,
        epsilon: float,
        normalize_biases: bool,
        degenerate_handling: str,
        classification_head_rescale: bool,
        task: str,
        activation: str,
    ) -> tuple[ParamTree, list[Any], dict[str, Any]]:
        """Balance qk/vo circuit scales; all other groups stay at identity."""

        if classification_head_rescale:
            raise ValueError(
                "classification_head_rescale is not supported for attention "
                "circuit normalization; it is a dense-chain post-pass."
            )
        if degenerate_handling != "preserve":
            raise ValueError(
                f"degenerate_handling={degenerate_handling!r} has no meaning for "
                "attention circuit balancing (the epsilon-stabilized balance is "
                "already defined for zero circuits); use 'preserve'."
            )

        scales = attention_balancing_scales(
            problem,
            params,
            epsilon=epsilon,
            normalize_biases=normalize_biases,
        )
        normalized_params = problem.apply_scales(
            params,
            ScaleState.from_scales(problem, scales, backend="jax"),
        )
        scale_factors = [
            scales[group_id] for group_id in problem.group_order if group_id in scales
        ]
        aux: dict[str, Any] = {
            "plan": "attention_balance",
            "mode": "balanced",
            "task_type": task,
            "epsilon": epsilon,
            "degenerate_handling": degenerate_handling,
            "normalize_biases": normalize_biases,
            "activation": activation,
            "classification_head_rescale": classification_head_rescale,
            "balanced_groups": [
                group_id for group_id in problem.group_order if group_id in scales
            ],
        }
        return normalized_params, scale_factors, aux


__all__ = ["ScaleCanonicalizer"]
