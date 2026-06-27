"""Graph-driven scale normalization.

The :class:`ScaleNormalizer` is the deterministic canonicalization counterpart of
the rebasin solver stack: it reads producer/consumer axis roles off an
:class:`~align.alignment.AlignmentProblem` and reuses the
:func:`align.normalize.normalize_layers` math kernel so its output matches the
former adapter-specific normalization byte-for-byte.

It supports the linear ReLU-MLP class of graphs (each permutation group has a
single ``out``-role producer kernel, kernels are 2-D, and the groups form a
single chain feeding one output head). Residual ties and branches give a group
multiple ``out``-role producers and are rejected pending issue #4; convolutional
kernels are rejected as non-dense. This preserves the previous "ResNet normalize
unsupported" behaviour with a clearer message.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
from flax.core import frozen_dict

from .alignment import AlignmentProblem
from .architecture import _descend, _set_path
from .dense_layers import DenseLayer

ParamTree = Mapping[str, Any]
_LayerPaths = tuple[tuple[str, ...], tuple[str, ...]]


def _scale_chain(problem: AlignmentProblem) -> list[_LayerPaths]:
    """Return ordered ``(kernel_path, bias_path)`` pairs for the dense chain.

    The chain is ``[producer(g0), producer(g1), ..., producer(g_last), head]``
    where each ``producer`` is a group's single ``out``-role dense kernel (plus
    bias) and ``head`` is the lone ungrouped output kernel that consumes the last
    group. Raises with an actionable message when the graph is outside the
    supported linear ReLU-MLP class.
    """

    tensors = problem.tensors
    chain: list[_LayerPaths] = []
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
                "Residual ties and branches need shared-scale semantics that are "
                "deferred to issue #4; disable the normalize stage for this "
                "architecture."
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
        if len(bias_ids) != 1:
            raise ValueError(
                f"Scale normalization requires exactly one out-role bias for group "
                f"{group_id!r}, found {len(bias_ids)}."
            )
        if previous_group is not None:
            consumed_groups = {
                binding.group
                for binding in problem.bindings_for_tensor(kernel_id)
                if binding.role == "in"
            }
            if previous_group not in consumed_groups:
                raise ValueError(
                    "Scale normalization requires a single linear chain, but the "
                    f"producer for group {group_id!r} does not consume the previous "
                    f"group {previous_group!r}."
                )
        producer_kernel_ids.add(kernel_id)
        chain.append((tensors[kernel_id].path, tensors[bias_ids[0]].path))
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
    chain.append((head_path, (*head_path[:-1], "bias")))
    return chain


def _inject(
    params: ParamTree,
    chain: Sequence[_LayerPaths],
    layers: Sequence[DenseLayer],
) -> ParamTree:
    is_frozen = isinstance(params, frozen_dict.FrozenDict)
    mutable = frozen_dict.unfreeze(params) if is_frozen else copy.deepcopy(params)
    for (kernel_path, bias_path), layer in zip(chain, layers, strict=True):
        _set_path(mutable, kernel_path, layer.kernel)
        _set_path(mutable, bias_path, layer.bias)
    return frozen_dict.freeze(mutable) if is_frozen else mutable


class ScaleNormalizer:
    """Deterministic graph-native scale canonicalization for dense ReLU MLPs."""

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

        from .normalize import normalize_layers

        chain = _scale_chain(problem)
        layers = [
            DenseLayer(
                kernel=jnp.asarray(_descend(params, kernel_path)),
                bias=jnp.asarray(_descend(params, bias_path)),
            )
            for kernel_path, bias_path in chain
        ]
        normalized_layers, scale_factors, aux = normalize_layers(
            layers,
            task_type=task_type,
            num_classes=num_classes,
            **method_kwargs,
        )
        normalized_params = _inject(params, chain, normalized_layers)
        return normalized_params, scale_factors, aux


__all__ = ["ScaleNormalizer"]
