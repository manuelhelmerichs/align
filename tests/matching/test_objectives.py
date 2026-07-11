"""Tests for graph objectives."""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_util

from align.architectures import MLPRecipe
from align.matching import (
    EuclideanObjective,
    TransformState,
    solve_lap_maximize,
)
from tests.matching._helpers import permutation_matrix, two_layer_graph


def test_euclidean_objective_prefers_correct_permutation():
    recipe = MLPRecipe(parameter_root="fcn")
    params = {
        "fcn": {
            "Dense_0": {
                "kernel": jnp.array([[1.0, -0.5], [0.2, 1.2]]),
                "bias": jnp.array([0.3, -0.4]),
            },
            "Dense_1": {
                "kernel": jnp.array([[0.7, -1.1], [0.5, 0.2]]),
                "bias": jnp.array([0.1, -0.2]),
            },
        }
    }
    graph = recipe.build_graph(params)
    reference_data = graph.materialize(params, backend="jax")
    swap = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    target_params = graph.apply_transforms(
        params, TransformState.from_transforms(graph, {"mlp/h0": swap})
    )
    target_data = graph.materialize(target_params, backend="jax")

    objective = EuclideanObjective()
    identity_state = TransformState.identity(graph)
    swap_state = TransformState.from_transforms(graph, {"mlp/h0": swap})
    cost_identity = float(
        objective.value(graph, reference_data, target_data, identity_state)
    )
    cost_swap = float(objective.value(graph, reference_data, target_data, swap_state))

    assert cost_swap < cost_identity


def test_euclidean_objective_matches_explicit_applied_distance():
    recipe = MLPRecipe(parameter_root="fcn")
    ref = {
        "fcn": {
            "Dense_0": {
                "kernel": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
                "bias": jnp.array([0.5, -0.5], dtype=jnp.float32),
            },
            "Dense_1": {
                "kernel": jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32),
                "bias": jnp.array([0.1, -0.2], dtype=jnp.float32),
            },
        }
    }
    graph = recipe.build_graph(ref)
    target = {
        "fcn": {
            "Dense_0": {
                "kernel": jnp.array([[2.0, 1.0], [4.0, 3.0]], dtype=jnp.float32),
                "bias": jnp.array([-0.5, 0.5], dtype=jnp.float32),
            },
            "Dense_1": {
                "kernel": jnp.array([[8.0, 7.0], [6.0, 5.0]], dtype=jnp.float32),
                "bias": jnp.array([0.1, -0.2], dtype=jnp.float32),
            },
        }
    }
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    state = TransformState.from_transforms(graph, {"mlp/h0": swap})
    objective_value = float(
        EuclideanObjective().value(
            graph,
            graph.materialize(ref, backend="jax"),
            graph.materialize(target, backend="jax"),
            state,
        )
    )

    applied = graph.apply_transforms(target, state)
    ref_leaves, ref_def = tree_util.tree_flatten(ref)
    applied_leaves, applied_def = tree_util.tree_flatten(applied)
    assert ref_def == applied_def
    explicit = 0.0
    for lhs, rhs in zip(ref_leaves, applied_leaves, strict=True):
        diff = np.asarray(lhs) - np.asarray(rhs)
        explicit += float(np.sum(np.square(diff)))

    np.testing.assert_allclose(objective_value, explicit, atol=1e-6)


def test_linearize_group_is_exact_affine_reduction_of_objective():
    """The LAP cost is an exact affine reduction of the L2 objective.

    For a group binding each tensor on a single axis, applying a permutation
    leaves ``||ref||^2 + ||target||^2`` unchanged, so the only permutation-
    dependent term is ``-2 <ref, P target> = -2 S(P)`` where ``S(P)`` is the
    assignment score read off the linearized cost. Hence ``value(P) + 2 S(P)``
    must be constant across *all* permutations -- the defining property of an
    exact LAP linearization (as opposed to a heuristic surrogate).
    """

    hidden = 4
    graph, ref, target = two_layer_graph(seed=3, hidden=hidden)
    objective = EuclideanObjective()
    reference_data = graph.materialize(ref, backend="numpy")
    target_data = graph.materialize(target, backend="numpy")
    identity = TransformState.identity(graph, backend="numpy")
    cost = np.asarray(
        objective.linearize_group(
            graph, reference_data, target_data, identity, "mlp/h0"
        )
    )

    affine_constants = []
    for perm in itertools.permutations(range(hidden)):
        state = TransformState.from_transforms(
            graph, {"mlp/h0": permutation_matrix(perm)}
        )
        value = float(objective.value(graph, reference_data, target_data, state))
        score = float(sum(cost[i, perm[i]] for i in range(hidden)))
        affine_constants.append(value + 2.0 * score)

    spread = float(np.ptp(affine_constants))
    scale = abs(float(np.mean(affine_constants))) + 1.0
    assert spread <= 1e-4 * scale


def test_lap_linearization_selects_global_optimum_over_all_permutations():
    """Hungarian on the linearized cost finds the true objective minimizer."""

    hidden = 4
    graph, ref, target = two_layer_graph(seed=7, hidden=hidden)
    objective = EuclideanObjective()
    reference_data = graph.materialize(ref, backend="numpy")
    target_data = graph.materialize(target, backend="numpy")
    identity = TransformState.identity(graph, backend="numpy")
    cost = objective.linearize_group(
        graph, reference_data, target_data, identity, "mlp/h0"
    )

    values = {
        perm: float(
            objective.value(
                graph,
                reference_data,
                target_data,
                TransformState.from_transforms(
                    graph, {"mlp/h0": permutation_matrix(perm)}
                ),
            )
        )
        for perm in itertools.permutations(range(hidden))
    }
    brute_force_best = min(values, key=values.get)
    lap_solution = tuple(int(j) for j in solve_lap_maximize(cost))

    assert lap_solution == brute_force_best


def test_repeated_group_linearization_raises_quadratic_assignment_error():
    """A group binding one tensor on two axes is a QAP and must be rejected."""

    from align.matching import UnsupportedGroupLinearization
    from align.symmetry import (
        AxisBinding,
        SymmetryGraph,
        SymmetryGroup,
        TensorSpec,
    )

    graph = SymmetryGraph(
        groups={"g0": SymmetryGroup(id="g0", size=2)},
        tensors={"w": TensorSpec(id="w", path=("w",), shape=(2, 2))},
        axis_bindings=(
            AxisBinding(tensor_id="w", axis=0, group="g0", role="in"),
            AxisBinding(tensor_id="w", axis=1, group="g0", role="out"),
        ),
    )
    params = {"w": np.eye(2, dtype=np.float32)}
    data = graph.materialize(params, backend="numpy")
    state = TransformState.identity(graph, backend="numpy")

    with pytest.raises(UnsupportedGroupLinearization, match="quadratic assignment"):
        EuclideanObjective().linearize_group(graph, data, data, state, "g0")
