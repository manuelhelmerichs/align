"""Tests for alignment registries and graph solver behavior."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures import MLPRecipe
from align.matching import (
    EuclideanObjective,
    SolverSequence,
    SolverStep,
    TransformState,
    UnsupportedGroupLinearization,
    available_objectives,
    available_solvers,
    get_objective,
)
from align.symmetry import AxisBinding, SymmetryGraph, SymmetryGroup, TensorSpec


def _perm_matrix(indices) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    matrix = np.zeros((idx.size, idx.size), dtype=np.float32)
    matrix[np.arange(idx.size), idx] = 1.0
    return matrix


def _three_layer_params(seed: int, hidden: tuple[int, int]):
    rng = np.random.default_rng(seed)
    h0, h1 = hidden
    sizes = [(3, h0), (h0, h1), (h1, 2)]
    fcn = {}
    for idx, (din, dout) in enumerate(sizes):
        fcn[f"Dense_{idx}"] = {
            "kernel": jnp.asarray(
                0.7 * rng.standard_normal((din, dout)), dtype=jnp.float32
            ),
            "bias": jnp.asarray(0.3 * rng.standard_normal((dout,)), dtype=jnp.float32),
        }
    return {"fcn": fcn}


def _params():
    return {
        "params": {
            "fcn": {
                "l0": {"kernel": jnp.eye(2), "bias": jnp.zeros((2,))},
                "l1": {"kernel": jnp.eye(2), "bias": jnp.zeros((2,))},
            }
        }
    }


class TestAlignmentRegistries:
    def test_available_objectives_and_solvers(self):
        assert "euclidean" in available_objectives()
        assert get_objective("euclidean").name == "euclidean"
        assert available_solvers() == ["lap", "procrustes", "sinkhorn"]

    def test_unknown_objective_raises(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            get_objective("missing")


class TestLAPSchedule:
    def test_identity_state(self):
        recipe = MLPRecipe(parameter_root="params.fcn")
        graph = recipe.build_graph(_params())
        state = TransformState.identity(graph, backend="numpy")
        assert set(state.transforms) == {"mlp/h0"}
        np.testing.assert_allclose(state.matrix("mlp/h0"), np.eye(2))

    def test_lap_recovers_permutation(self):
        recipe = MLPRecipe(parameter_root="params.fcn")
        params = _params()
        graph = recipe.build_graph(params)
        swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        target = graph.apply_transforms(
            params, TransformState.from_transforms(graph, {"mlp/h0": swap})
        )

        solver_sequence = SolverSequence(
            EuclideanObjective(),
            [SolverStep(solver="lap", max_sweeps=50, tolerance=0.0)],
        )
        state, aux = solver_sequence.solve(
            graph,
            graph.materialize(params, backend="numpy"),
            graph.materialize(target, backend="numpy"),
        )
        assert aux["steps"][0]["sweeps"] >= 1
        np.testing.assert_allclose(state.matrix("mlp/h0"), swap.T, atol=1e-6)

    def test_repeated_group_lap_is_rejected(self):
        params = {
            "core": {
                "Conv_0": {
                    "kernel": np.zeros((1, 1, 1, 2), dtype=np.float32),
                    "bias": np.zeros((2,), dtype=np.float32),
                },
                "Conv_1": {
                    "kernel": np.zeros((1, 1, 2, 2), dtype=np.float32),
                    "bias": np.zeros((2,), dtype=np.float32),
                },
            }
        }
        from align.architectures import ResidualConvNetRecipe

        recipe = ResidualConvNetRecipe(
            parameter_root="core",
            residual_topology={
                "schema": "align.residual_module_graph",
                "version": 1,
                "nodes": [
                    {"id": "input", "kind": "input"},
                    {"id": "core/Conv_0", "kind": "conv"},
                    {"id": "core/Conv_1", "kind": "conv"},
                    {"id": "add", "kind": "add"},
                ],
                "edges": [
                    {"source": "input", "target": "core/Conv_0"},
                    {"source": "core/Conv_0", "target": "core/Conv_1"},
                    {"source": "core/Conv_0", "target": "add"},
                    {"source": "core/Conv_1", "target": "add"},
                ],
            },
        )
        graph = recipe.build_graph(params)
        solver_sequence = SolverSequence(
            EuclideanObjective(),
            [SolverStep(solver="lap", max_sweeps=1)],
        )
        with pytest.raises(UnsupportedGroupLinearization, match="quadratic assignment"):
            solver_sequence.solve(
                graph,
                graph.materialize(params, backend="numpy"),
                graph.materialize(params, backend="numpy"),
            )

    def test_lap_coordinate_descent_recovers_exact_two_group_orbit(self):
        """Alternating LAP recovers the exact joint minimizer of a 2-group orbit.

        Both hidden groups feed the shared middle kernel, so the per-group
        updates are coupled; on an exact orbit the global optimum (loss 0) is
        attainable and must be reached and verified against brute force.
        """

        recipe = MLPRecipe(parameter_root="fcn")
        params = _three_layer_params(seed=11, hidden=(3, 3))
        graph = recipe.build_graph(params)
        forward = {
            "mlp/h0": _perm_matrix([2, 0, 1]),
            "mlp/h1": _perm_matrix([1, 2, 0]),
        }
        target = graph.apply_transforms(
            params, TransformState.from_transforms(graph, forward)
        )

        reference_data = graph.materialize(params, backend="numpy")
        target_data = graph.materialize(target, backend="numpy")
        objective = EuclideanObjective()

        brute_force_best = min(
            float(
                objective.value(
                    graph,
                    reference_data,
                    target_data,
                    TransformState.from_transforms(
                        graph,
                        {
                            "mlp/h0": _perm_matrix(p0),
                            "mlp/h1": _perm_matrix(p1),
                        },
                    ),
                )
            )
            for p0 in itertools.permutations(range(3))
            for p1 in itertools.permutations(range(3))
        )

        solver_sequence = SolverSequence(
            objective, [SolverStep(solver="lap", max_sweeps=50, tolerance=0.0)]
        )
        state, _ = solver_sequence.solve(graph, reference_data, target_data)
        solved_value = float(objective.value(graph, reference_data, target_data, state))

        assert brute_force_best < 1e-6
        np.testing.assert_allclose(solved_value, brute_force_best, atol=1e-6)
        # The recovered permutations invert the forward action.
        np.testing.assert_allclose(
            np.asarray(state.matrix("mlp/h0")), np.asarray(forward["mlp/h0"]).T
        )
        np.testing.assert_allclose(
            np.asarray(state.matrix("mlp/h1")), np.asarray(forward["mlp/h1"]).T
        )

    def test_lap_schedule_converges_before_exhausting_sweep_budget(self):
        """Coordinate descent reaches a fixed point and stops early."""

        recipe = MLPRecipe(parameter_root="fcn")
        params = _three_layer_params(seed=2, hidden=(4, 3))
        graph = recipe.build_graph(params)
        forward = {
            "mlp/h0": _perm_matrix([3, 1, 0, 2]),
            "mlp/h1": _perm_matrix([2, 0, 1]),
        }
        target = graph.apply_transforms(
            params, TransformState.from_transforms(graph, forward)
        )

        solver_sequence = SolverSequence(
            EuclideanObjective(),
            [SolverStep(solver="lap", max_sweeps=25, tolerance=0.0)],
        )
        _, aux = solver_sequence.solve(
            graph,
            graph.materialize(params, backend="numpy"),
            graph.materialize(target, backend="numpy"),
        )
        step = aux["steps"][0]
        # A converged sweep makes no further changes (max_delta 0) and the
        # solver_sequence stops before burning the whole budget.
        assert step["max_delta"] == 0.0
        assert step["sweeps"] < 25


class TestSinkhornSchedule:
    def test_sinkhorn_returns_aux_info(self):
        recipe = MLPRecipe(parameter_root="params.fcn")
        params = _params()
        graph = recipe.build_graph(params)
        solver_sequence = SolverSequence(
            EuclideanObjective(),
            [
                SolverStep(
                    solver="sinkhorn",
                    max_steps=5,
                    tolerance=1e-4,
                    sinkhorn_iterations=20,
                )
            ],
        )
        state, aux = solver_sequence.solve(
            graph,
            graph.materialize(params, backend="jax"),
            graph.materialize(params, backend="jax"),
            rng_key=jax.random.PRNGKey(0),
        )

        assert set(state.transforms) == {"mlp/h0"}
        assert "loss_final" in aux["steps"][0]
        assert "loss_initial" in aux["steps"][0]
        assert aux["steps"][0]["steps"] >= 1

    def test_sinkhorn_rejects_signed_groups_before_execution(self):
        graph = SymmetryGraph(
            groups={
                "g": SymmetryGroup(
                    id="g", size=2, transform_family="signed_permutation"
                )
            },
            tensors={"w": TensorSpec(id="w", path=("w",), shape=(2,))},
            axis_bindings=(AxisBinding(tensor_id="w", axis=0, group="g"),),
        )
        sequence = SolverSequence(
            EuclideanObjective(), [SolverStep(solver="sinkhorn", max_steps=1)]
        )
        with pytest.raises(ValueError, match="represents only plain permutations"):
            sequence.validate_graph(graph)

    def test_sinkhorn_records_compact_ambiguity(self):
        graph = MLPRecipe(parameter_root="params.fcn").build_graph(_params())
        sequence = SolverSequence(
            EuclideanObjective(),
            [
                SolverStep(
                    solver="sinkhorn",
                    max_steps=2,
                    record_ambiguity=True,
                )
            ],
        )
        _, aux = sequence.solve(
            graph,
            graph.materialize(_params(), backend="jax"),
            graph.materialize(_params(), backend="jax"),
            rng_key=jax.random.PRNGKey(0),
        )
        ambiguity = aux["steps"][0]["ambiguity"]["mlp/h0"]
        assert set(ambiguity) == {
            "row_entropy_mean",
            "row_entropy_max",
            "assignment_margin_mean",
            "assignment_margin_min",
        }


def test_lap_records_assignment_margin():
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(_params())
    sequence = SolverSequence(
        EuclideanObjective(),
        [SolverStep(solver="lap", max_sweeps=1, record_ambiguity=True)],
    )
    _, aux = sequence.solve(
        graph,
        graph.materialize(_params(), backend="numpy"),
        graph.materialize(_params(), backend="numpy"),
    )
    margin = aux["steps"][0]["ambiguity"]["mlp/h0"]
    assert margin["assignment_objective_margin"] >= 0.0
