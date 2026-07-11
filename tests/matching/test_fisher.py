"""Tests for the diagonal Fisher-metric objective and Fisher estimation."""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from align.matching import (
    DiagonalFisherObjective,
    TransformState,
    estimate_diag_fisher_tree,
    estimate_diag_fisher_weights,
    load_tensor_weights_npz,
    resolve_calibration_kwargs,
    save_tensor_weights_npz,
    solve_lap_maximize,
)
from align.matching.objectives import EuclideanObjective
from align.symmetry import (
    AxisBinding,
    SymmetryGraph,
    SymmetryGroup,
    TensorSpec,
)
from tests.matching._helpers import permutation_matrix, two_layer_graph


def _mlp_apply(params, x):
    """ReLU MLP forward for the {"fcn": {...}} layout of _two_layer_problem."""

    layers = params["fcn"]
    h = jnp.maximum(0.0, x @ layers["Dense_0"]["kernel"] + layers["Dense_0"]["bias"])
    return h @ layers["Dense_1"]["kernel"] + layers["Dense_1"]["bias"]


def _random_weights(graph, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        tensor_id: rng.uniform(0.1, 3.0, size=spec.shape)
        for tensor_id, spec in graph.tensors.items()
    }


def test_diagonal_fisher_requires_exactly_one_weight_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        DiagonalFisherObjective()
    weights = {"w": np.ones((2, 2))}
    path = tmp_path / "weights.npz"
    save_tensor_weights_npz(weights, path)
    with pytest.raises(ValueError, match="exactly one"):
        DiagonalFisherObjective(tensor_weights=weights, weights_path=path)


def test_diagonal_fisher_value_matches_explicit_weighted_distance():
    graph, ref, target = two_layer_graph(seed=11, hidden=3)
    weights = _random_weights(graph, seed=5)
    objective = DiagonalFisherObjective(tensor_weights=weights)
    perm = permutation_matrix([2, 0, 1])
    state = TransformState.from_transforms(graph, {"mlp/h0": perm})

    reference_data = graph.materialize(ref, backend="jax")
    target_data = graph.materialize(target, backend="jax")
    value = float(objective.value(graph, reference_data, target_data, state))

    applied = graph.materialize(
        graph.apply_transforms(target, state), backend="numpy"
    ).materialize()
    explicit = sum(
        float(
            np.sum(
                weights[tid] * np.square(np.asarray(reference_data[tid]) - applied[tid])
            )
        )
        for tid in graph.tensors
    )
    np.testing.assert_allclose(value, explicit, rtol=1e-5)


def test_diagonal_fisher_value_reduces_to_l2_with_unit_weights():
    graph, ref, target = two_layer_graph(seed=2, hidden=4)
    unit = {tid: np.ones(spec.shape) for tid, spec in graph.tensors.items()}
    state = TransformState.from_transforms(
        graph, {"mlp/h0": permutation_matrix([1, 3, 0, 2])}
    )
    reference_data = graph.materialize(ref, backend="jax")
    target_data = graph.materialize(target, backend="jax")
    fisher_value = float(
        DiagonalFisherObjective(tensor_weights=unit).value(
            graph, reference_data, target_data, state
        )
    )
    l2_value = float(
        EuclideanObjective().value(graph, reference_data, target_data, state)
    )
    np.testing.assert_allclose(fisher_value, l2_value, rtol=1e-6)


def test_diagonal_fisher_linearization_is_exact_affine_reduction():
    """value(P) + 2*score(P) must be constant over all permutations.

    Unlike plain L2 the weighted target self-energy is assignment-dependent,
    so this only holds because the linearized cost carries the self term.
    """

    hidden = 4
    graph, ref, target = two_layer_graph(seed=3, hidden=hidden)
    objective = DiagonalFisherObjective(tensor_weights=_random_weights(graph, seed=9))
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


def test_diagonal_fisher_lap_selects_global_optimum():
    hidden = 4
    graph, ref, target = two_layer_graph(seed=7, hidden=hidden)
    objective = DiagonalFisherObjective(tensor_weights=_random_weights(graph, seed=1))
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


def test_fisher_weighting_flips_the_optimal_permutation():
    """The metric changes matching decisions, not just the objective scale.

    Coordinate 0 carries large values that agree under identity; coordinate 1
    is small but function-critical and agrees under the swap. Unweighted L2
    picks identity, the Fisher metric picks the swap.
    """

    graph = SymmetryGraph(
        groups={"g0": SymmetryGroup(id="g0", size=2)},
        tensors={"w": TensorSpec(id="w", path=("w",), shape=(2, 2))},
        axis_bindings=(AxisBinding(tensor_id="w", axis=0, group="g0", role="out"),),
    )
    ref = {"w": np.array([[5.0, 1.0], [0.0, 0.0]], dtype=np.float32)}
    target = {"w": np.array([[5.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
    weights = {"w": np.array([[0.001, 1.0], [0.001, 1.0]], dtype=np.float64)}

    reference_data = graph.materialize(ref, backend="numpy")
    target_data = graph.materialize(target, backend="numpy")
    identity = TransformState.identity(graph, backend="numpy")

    l2_cost = EuclideanObjective().linearize_group(
        graph, reference_data, target_data, identity, "g0"
    )
    fisher_cost = DiagonalFisherObjective(tensor_weights=weights).linearize_group(
        graph, reference_data, target_data, identity, "g0"
    )
    l2_perm = tuple(int(j) for j in solve_lap_maximize(l2_cost))
    fisher_perm = tuple(int(j) for j in solve_lap_maximize(fisher_cost))

    assert l2_perm == (0, 1)
    assert fisher_perm == (1, 0)


def test_diagonal_fisher_batched_value_matches_per_sample_values():
    graph, ref, target_a = two_layer_graph(seed=4, hidden=3)
    _, _, target_b = two_layer_graph(seed=5, hidden=3)
    weights = _random_weights(graph, seed=3)
    objective = DiagonalFisherObjective(tensor_weights=weights)
    perm = jnp.asarray(permutation_matrix([1, 2, 0]))
    state = TransformState.from_transforms(graph, {"mlp/h0": perm})

    reference_data = graph.materialize(ref, backend="jax")
    data_a = graph.materialize(target_a, backend="jax")
    data_b = graph.materialize(target_b, backend="jax")
    batched = {
        tid: jnp.stack([jnp.asarray(data_a[tid]), jnp.asarray(data_b[tid])])
        for tid in graph.tensors
    }

    batch_values = np.asarray(objective.value(graph, reference_data, batched, state))
    single_values = [
        float(objective.value(graph, reference_data, data, state))
        for data in (data_a, data_b)
    ]
    np.testing.assert_allclose(batch_values, single_values, rtol=1e-5)


def test_diagonal_fisher_validates_weight_coverage_and_shape():
    graph, ref, target = two_layer_graph(seed=1, hidden=3)
    reference_data = graph.materialize(ref, backend="jax")
    target_data = graph.materialize(target, backend="jax")
    state = TransformState.identity(graph, backend="numpy")

    missing = _random_weights(graph, seed=0)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="no weights for tensor"):
        DiagonalFisherObjective(tensor_weights=missing).value(
            graph, reference_data, target_data, state
        )

    bad_shape = _random_weights(graph, seed=0)
    first = next(iter(bad_shape))
    bad_shape[first] = np.ones((1,))
    with pytest.raises(ValueError, match="have shape"):
        DiagonalFisherObjective(tensor_weights=bad_shape).value(
            graph, reference_data, target_data, state
        )


def test_estimate_diag_fisher_matches_linear_model_analytics():
    """For a linear map f(x) = x @ W, F[i, j] = sum_n x[n, i]^2 for all j."""

    rng = np.random.default_rng(0)
    inputs = jnp.asarray(rng.standard_normal((7, 3)), dtype=jnp.float32)
    params = {"w": jnp.asarray(rng.standard_normal((3, 2)), dtype=jnp.float32)}

    fisher = estimate_diag_fisher_tree(params, lambda p, x: x @ p["w"], inputs)
    expected = np.repeat(
        np.sum(np.square(np.asarray(inputs)), axis=0)[:, None], 2, axis=1
    )
    np.testing.assert_allclose(np.asarray(fisher["w"]), expected, rtol=1e-5)


def test_estimate_diag_fisher_fails_before_oversized_dense_jacobian():
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    inputs = jnp.ones((4, 4), dtype=jnp.float32)
    with pytest.raises(MemoryError, match="Jacobian would require"):
        estimate_diag_fisher_tree(
            params,
            lambda p, x: x @ p["w"],
            inputs,
            memory_limit_bytes=1,
        )


def test_estimate_diag_fisher_weights_are_damped_and_normalized():
    graph, ref, _ = two_layer_graph(seed=6, hidden=3)
    rng = np.random.default_rng(1)
    inputs = jnp.asarray(rng.standard_normal((9, 3)), dtype=jnp.float32)
    weights = estimate_diag_fisher_weights(graph, ref, _mlp_apply, inputs, damping=1e-3)
    assert set(weights) == set(graph.tensors)
    flat = np.concatenate([np.ravel(w) for w in weights.values()])
    assert np.all(flat > 0.0)
    np.testing.assert_allclose(np.mean(flat), 1.0, rtol=1e-6)


def test_tensor_weights_npz_roundtrip_with_slash_keys(tmp_path):
    weights = {
        "fcn/Dense_0/kernel": np.arange(6.0).reshape(2, 3),
        "fcn/Dense_0/bias": np.array([1.0, 2.0]),
    }
    path = tmp_path / "weights.npz"
    save_tensor_weights_npz(weights, path)
    loaded = load_tensor_weights_npz(path)
    assert set(loaded) == set(weights)
    for key, value in weights.items():
        np.testing.assert_array_equal(loaded[key], value)


def test_resolve_calibration_kwargs():
    graph, ref, _ = two_layer_graph(seed=8, hidden=3)

    passthrough = resolve_calibration_kwargs(
        {"weights_path": "x.npz"},
        graph=graph,
        params=ref,
        apply_fn=None,
        inputs=None,
    )
    assert passthrough == {"weights_path": "x.npz"}

    with pytest.raises(ValueError, match="requires an apply_fn"):
        resolve_calibration_kwargs(
            {"calibration": "fisher"},
            graph=graph,
            params=ref,
            apply_fn=None,
            inputs=None,
        )

    rng = np.random.default_rng(2)
    inputs = jnp.asarray(rng.standard_normal((5, 3)), dtype=jnp.float32)
    resolved = resolve_calibration_kwargs(
        {"calibration": "fisher"},
        graph=graph,
        params=ref,
        apply_fn=_mlp_apply,
        inputs=inputs,
    )
    assert set(resolved) == {"tensor_weights"}
    assert set(resolved["tensor_weights"]) == set(graph.tensors)
