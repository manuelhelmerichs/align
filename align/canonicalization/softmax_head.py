"""Softmax-head translation centering: probability-preserving, not logit-preserving.

For a classification head producing logits ``z = h @ W + b``, adding the same
vector to every class column (``W <- W + v 1^T``) or the same constant to every
bias entry (``b <- b + c 1``) shifts all logits by a common per-input amount,
which the softmax removes: predictive *probabilities* are exactly invariant,
but the logits change. This translation is therefore not part of the
exact-logit symmetry group the ``canonicalize`` stage removes; it is exposed
as the opt-in ``center_softmax_head`` pipeline stage for classification runs.

The canonical representative has zero-mean class columns: subtracting the
row-wise column mean from ``W`` and the mean from ``b`` is the orthogonal
projection along the orbit directions, hence the unique minimum-norm orbit
point (the same philosophy as balanced scale canonicalization), exactly
idempotent, and exactly equivariant under further translations.

Head discovery is structural: a softmax head kernel is a bound 2-D tensor
whose input axis consumes a symmetry group (``in`` role) and whose output
(class) axis is unbound — nothing downstream constrains the logits. Trees
with several such heads are ambiguous and require the explicit ``head``
option (the module path containing ``kernel`` and optionally ``bias``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..symmetry import SymmetryGraph, binding_axis_interval
from ..symmetry.tensor_ops import _descend
from ..tree_utils import replace_paths

ParamTree = Mapping[str, Any]


def detect_softmax_head(graph: SymmetryGraph) -> tuple[str, ...]:
    """Return the module path of the unique structural softmax head.

    Candidates are 2-D bound tensors with at least one ``in``-role binding on
    axis 0, no ``out``-role bindings, and an unbound axis 1. Zero or several
    candidates raise with instructions to configure ``head`` explicitly.
    """

    candidates: list[tuple[str, ...]] = []
    for tensor_id, spec in graph.tensors.items():
        bindings = graph.bindings_for_tensor(tensor_id)
        if not bindings or len(spec.shape) != 2:
            continue
        if any(binding.role == "out" for binding in bindings):
            continue
        axes = {binding_axis_interval(spec.shape, binding)[0] for binding in bindings}
        if axes != {0}:
            continue
        candidates.append(spec.path)

    if len(candidates) != 1:
        described = ", ".join(".".join(path) for path in sorted(candidates))
        raise ValueError(
            f"Expected exactly one structural softmax head (a bound 2-D kernel "
            f"with an unbound class axis), found {len(candidates)}"
            + (f": {described}" if candidates else "")
            + ". Set center_softmax_head.head to the head module path."
        )
    kernel_path = candidates[0]
    if kernel_path[-1] != "kernel":
        raise ValueError(
            f"Detected softmax head tensor {'.'.join(kernel_path)} is not a "
            "module 'kernel' entry; set center_softmax_head.head explicitly."
        )
    return kernel_path[:-1]


def center_softmax_head(
    params: ParamTree, head_path: tuple[str, ...]
) -> tuple[ParamTree, dict[str, Any]]:
    """Return ``params`` with the head translated to zero-mean class columns.

    Subtracts the per-row mean over class columns from the head kernel and the
    mean from the head bias (if present). Probabilities are preserved exactly;
    logits change by a common per-input shift.
    """

    module = _descend(params, head_path)
    if not isinstance(module, Mapping) or "kernel" not in module:
        raise ValueError(
            f"center_softmax_head: no 'kernel' under module path {'.'.join(head_path)}."
        )
    kernel = np.asarray(module["kernel"])
    if kernel.ndim != 2:
        raise ValueError(
            f"center_softmax_head expects a 2-D head kernel, got shape "
            f"{tuple(kernel.shape)} at {'.'.join(head_path)}."
        )
    column_mean = np.mean(kernel, axis=1, keepdims=True)
    replacements: list[tuple[tuple[str, ...], Any]] = [
        ((*head_path, "kernel"), kernel - column_mean)
    ]
    diagnostics: dict[str, Any] = {
        "plan": "softmax_head_translation",
        "head": ".".join(head_path),
        "kernel_translation_norm": float(np.linalg.norm(np.asarray(column_mean))),
    }
    if "bias" in module:
        bias = np.asarray(module["bias"])
        bias_mean = np.mean(bias)
        replacements.append(((*head_path, "bias"), bias - bias_mean))
        diagnostics["bias_translation"] = float(bias_mean)
    return replace_paths(params, replacements), diagnostics


__all__ = ["center_softmax_head", "detect_softmax_head"]
