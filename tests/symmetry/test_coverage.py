"""Tests for exact parameter-coordinate coverage of symmetry graphs."""

import numpy as np

from align.architectures import get_recipe
from align.symmetry import (
    AxisBinding,
    SymmetryGraph,
    SymmetryGroup,
    TensorSpec,
    symmetry_parameter_coverage,
)


def test_mlp_coverage_uses_the_complete_parameter_tree_denominator():
    params = {
        "params": {
            "fcn": {
                "dense0": {
                    "kernel": np.zeros((3, 5)),
                    "bias": np.zeros(5),
                },
                "dense1": {
                    "kernel": np.zeros((5, 4)),
                    "bias": np.zeros(4),
                },
                "dense2": {
                    "kernel": np.zeros((4, 2)),
                    "bias": np.zeros(2),
                },
            },
            "sigma": np.zeros(()),
        }
    }
    graph = get_recipe("mlp").build_graph(params)

    coverage = symmetry_parameter_coverage(graph, params)

    assert coverage["total_parameters"] == 55
    assert coverage["bound_parameters"] == 52
    assert coverage["unbound_parameters"] == 3
    assert coverage["bound_fraction"] == 52 / 55
    assert coverage["leaves"]["params/fcn/dense2/bias"]["bound_parameters"] == 0
    assert coverage["leaves"]["params/sigma"]["bound_parameters"] == 0


def test_coverage_counts_overlapping_bound_axes_only_once():
    params = {"tensor": np.zeros((2, 3)), "fixed": np.zeros(4)}
    graph = SymmetryGraph(
        groups={
            "rows": SymmetryGroup(id="rows", size=2),
            "columns": SymmetryGroup(id="columns", size=3),
        },
        tensors={
            "tensor": TensorSpec(id="tensor", path=("tensor",), shape=(2, 3)),
        },
        axis_bindings=(
            AxisBinding(tensor_id="tensor", axis=0, group="rows"),
            AxisBinding(tensor_id="tensor", axis=1, group="columns"),
        ),
    )
    graph.validate(params)

    coverage = symmetry_parameter_coverage(graph, params)

    assert coverage["total_parameters"] == 10
    assert coverage["bound_parameters"] == 6
    assert coverage["leaves"]["tensor"]["bound_parameters"] == 6
    assert coverage["leaves"]["fixed"]["bound_parameters"] == 0


def test_coverage_expands_selectors_and_repeated_intervals():
    params = {
        "selected": np.zeros((2, 2, 3)),
        "repeated": np.zeros((2, 8)),
    }
    graph = SymmetryGraph(
        groups={
            "features": SymmetryGroup(id="features", size=3),
            "pairs": SymmetryGroup(id="pairs", size=2),
        },
        tensors={
            "selected": TensorSpec(id="selected", path=("selected",), shape=(2, 2, 3)),
            "repeated": TensorSpec(id="repeated", path=("repeated",), shape=(2, 8)),
        },
        axis_bindings=(
            AxisBinding(
                tensor_id="selected",
                axis=2,
                group="features",
                selector=((1, 0),),
            ),
            AxisBinding(
                tensor_id="repeated",
                axis=1,
                group="pairs",
                start=0,
                stop=2,
                repeat=2,
                stride=4,
            ),
        ),
    )
    graph.validate(params)

    coverage = symmetry_parameter_coverage(graph, params)

    assert coverage["total_parameters"] == 28
    assert coverage["leaves"]["selected"]["bound_parameters"] == 6
    assert coverage["leaves"]["repeated"]["bound_parameters"] == 8
    assert coverage["bound_parameters"] == 14
