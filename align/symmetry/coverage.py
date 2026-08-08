"""Parameter-coordinate coverage induced by symmetry-graph bindings."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .graph import SymmetryGraph
from .tensor_ops import (
    binding_axis_intervals,
    binding_indexer,
    binding_selector,
)


def _parameter_leaves(
    params: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    for key, value in params.items():
        path = (*prefix, str(key))
        if isinstance(value, Mapping):
            yield from _parameter_leaves(value, path)
        else:
            yield path, value


def symmetry_parameter_coverage(
    graph: SymmetryGraph, params: Mapping[str, Any]
) -> dict[str, Any]:
    """Count the union of scalar coordinates selected by graph bindings.

    A coordinate is graph-bound when at least one :class:`AxisBinding` selects
    it. Multiple groups or axes touching the same coordinate count it once.
    Parameter leaves absent from ``graph.tensors`` and registered tensors with
    no bindings count as unbound.

    The returned mapping is JSON-friendly and includes an exact per-leaf audit
    trail. ``bound_fraction`` always uses the complete parameter tree as its
    denominator, not only tensors registered in the graph.
    """

    leaves = {path: value for path, value in _parameter_leaves(params)}
    if not leaves:
        raise ValueError("Cannot measure symmetry coverage of an empty parameter tree.")

    masks = {
        path: np.zeros(tuple(int(dim) for dim in np.shape(value)), dtype=bool)
        for path, value in leaves.items()
    }
    for binding in graph.axis_bindings:
        tensor = graph.tensors[binding.tensor_id]
        if tensor.path not in masks:
            raise ValueError(
                f"Graph tensor {binding.tensor_id!r} at {tensor.path} is not a "
                "parameter leaf."
            )
        mask = masks[tensor.path]
        if tuple(mask.shape) != tuple(tensor.shape):
            raise ValueError(
                f"Graph tensor {binding.tensor_id!r} at {tensor.path} has shape "
                f"{tensor.shape}, but the parameter leaf has shape {mask.shape}."
            )
        selector = binding_selector(tensor.shape, binding)
        for axis, start, stop in binding_axis_intervals(tensor.shape, binding):
            mask[binding_indexer(mask.ndim, axis, start, stop, selector)] = True

    leaf_rows: dict[str, dict[str, Any]] = {}
    total_parameters = 0
    bound_parameters = 0
    for path, value in leaves.items():
        parameters = int(np.size(value))
        bound = int(np.count_nonzero(masks[path]))
        total_parameters += parameters
        bound_parameters += bound
        leaf_rows["/".join(path)] = {
            "shape": [int(dim) for dim in np.shape(value)],
            "parameters": parameters,
            "bound_parameters": bound,
            "unbound_parameters": parameters - bound,
            "bound_fraction": bound / parameters if parameters else 0.0,
        }

    if total_parameters == 0:
        raise ValueError(
            "Cannot measure symmetry coverage of a parameter tree with no scalars."
        )
    return {
        "total_parameters": total_parameters,
        "bound_parameters": bound_parameters,
        "unbound_parameters": total_parameters - bound_parameters,
        "bound_fraction": bound_parameters / total_parameters,
        "leaves": leaf_rows,
    }


__all__ = ["symmetry_parameter_coverage"]
