"""Graph-driven scale normalization.

The :class:`ScaleNormalizer` is the deterministic canonicalization counterpart of
the rebasin solver stack: it reads producer/consumer axis roles off an
:class:`~align.alignment.AlignmentProblem`, computes positive per-group scale
vectors, and applies the symmetry through
:meth:`align.alignment.AlignmentProblem.apply_scales`. Degenerate-neuron cleanup
and optional classification-head rescaling remain explicit post-passes because
they are not positive scale symmetries.

Two plans are dispatched on problem structure:

- **Attention circuit balancing** for problems with ``attention_block``
  constraints (transformers): the exact diagonal qk/vo circuit symmetry is
  canonicalized by equalizing query/key and value/out energies per intra-head
  dimension. This symmetry is activation-independent; all other groups
  (stream, heads, GELU FFN hidden units) carry no per-channel scale symmetry
  and stay at identity.
- **Dense chain normalization** for linear ReLU-MLP graphs (each permutation
  group has a single ``out``-role producer kernel, kernels are 2-D, and the
  groups form a single chain feeding one output head). Residual ties and
  branches give a group multiple ``out``-role producers and are rejected;
  convolutional kernels are rejected as non-dense.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict

from ..alignment import AlignmentProblem, AxisBinding, binding_axis_interval
from ..alignment.tensor_ops import _descend, _set_path
from .activations import is_positive_homogeneous, validate_activation
from .attention import attention_balancing_scales, attention_constraints
from .conv import conv_stack_producer_scales, has_conv_stack_block
from .dense_layers import DenseLayer
from .kernel import (
    _canonicalize_degenerate,
    _outgoing_scale_factors,
    _require_bool,
    _validate_degenerate_handling,
    compute_degenerate_mask,
    compute_incoming_norms,
    normalize_last_layer_classification,
)
from .scale_state import ScaleState

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


def _dense_scale_plan(problem: AlignmentProblem) -> _DenseScalePlan:
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


def _replace_paths(
    params: ParamTree,
    replacements: Sequence[tuple[tuple[str, ...], Any]],
) -> ParamTree:
    is_frozen = isinstance(params, frozen_dict.FrozenDict)
    mutable = frozen_dict.unfreeze(params) if is_frozen else copy.deepcopy(params)
    for path, value in replacements:
        _set_path(mutable, path, value)
    return frozen_dict.freeze(mutable) if is_frozen else mutable


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


def _zero_degenerate_consumers(
    problem: AlignmentProblem,
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
    return _replace_paths(params, replacements) if replacements else params


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
        _replace_paths(
            params,
            (
                (plan.head_kernel_path, normalized_head.kernel),
                (plan.head_bias_path, normalized_head.bias),
            ),
        ),
        gamma,
    )


class ScaleNormalizer:
    """Deterministic graph-native scale canonicalization."""

    def normalize(
        self,
        problem: AlignmentProblem,
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
        if method_kwargs:
            unexpected = next(iter(method_kwargs))
            raise TypeError(
                "ScaleNormalizer.normalize() got an unexpected keyword argument "
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

        if attention_constraints(problem):
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
        plan = _dense_scale_plan(problem)
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
        problem: AlignmentProblem,
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
            "task_type": task,
            "epsilon": epsilon,
            "degenerate_handling": degenerate_handling,
            "normalize_biases": normalize_biases,
            "activation": activation,
            "classification_head_rescale": classification_head_rescale,
        }
        return normalized_params, scale_factors, aux

    def _normalize_attention_circuits(
        self,
        problem: AlignmentProblem,
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


__all__ = ["ScaleNormalizer"]
