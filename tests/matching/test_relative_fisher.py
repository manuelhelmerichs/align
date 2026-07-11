"""Tests for the activation-Gram / relative-Fisher objective."""

import jax.numpy as jnp
import numpy as np
import pytest

from align.matching import (
    RelativeFisherObjective,
    TransformState,
    estimate_activation_gram_metrics,
    load_activation_gram_metrics_npz,
    match_sample,
    resolve_calibration_kwargs,
    save_activation_gram_metrics_npz,
)
from align.matching.objectives import EuclideanObjective
from benchmarks.synthetic import mlp_activation_samples
from tests.matching._helpers import permutation_matrix, two_layer_graph


def _metrics(graph, seed=0):
    rng = np.random.default_rng(seed)
    metrics = {}
    for tensor_id, spec in graph.tensors.items():
        if tensor_id.endswith("kernel"):
            size = spec.shape[0]
            factor = rng.standard_normal((size, size))
            gram = factor @ factor.T + 0.1 * np.eye(size)
            axes = (0,)
        else:
            gram = np.ones((1, 1))
            axes = ()
        metrics[tensor_id] = {"input_axes": axes, "gram": gram}
    return metrics


def test_relative_fisher_value_matches_explicit_mahalanobis_distance():
    graph, ref, target = two_layer_graph(seed=3, hidden=4)
    metrics = _metrics(graph)
    objective = RelativeFisherObjective(tensor_metrics=metrics)
    state = TransformState(
        group_order=graph.group_order,
        matrices={"mlp/h0": permutation_matrix([2, 0, 3, 1])},
    )
    reference_data = graph.materialize(ref, backend="jax")
    target_data = graph.materialize(target, backend="jax")

    value = float(objective.value(graph, reference_data, target_data, state))
    aligned = graph.materialize(
        graph.apply_transforms(target, state), backend="numpy"
    ).materialize()
    expected = 0.0
    for tensor_id, spec in graph.tensors.items():
        diff = np.asarray(reference_data[tensor_id]) - aligned[tensor_id]
        metric = metrics[tensor_id]
        if metric["input_axes"]:
            flat = diff.reshape((spec.shape[0], -1))
            expected += float(np.einsum("ir,ij,jr->", flat, metric["gram"], flat))
        else:
            expected += float(np.sum(diff**2))
    np.testing.assert_allclose(value, expected, rtol=1e-5)


def test_relative_fisher_batched_value_matches_single_values():
    graph, ref, target_a = two_layer_graph(seed=4, hidden=3)
    _, _, target_b = two_layer_graph(seed=5, hidden=3)
    objective = RelativeFisherObjective(tensor_metrics=_metrics(graph))
    state = TransformState(
        group_order=graph.group_order,
        matrices={"mlp/h0": jnp.asarray(permutation_matrix([1, 2, 0]))},
    )
    reference_data = graph.materialize(ref, backend="jax")
    targets = [
        graph.materialize(target, backend="jax") for target in (target_a, target_b)
    ]
    batched = {
        tensor_id: jnp.stack([target[tensor_id] for target in targets])
        for tensor_id in graph.tensors
    }
    batch_values = np.asarray(objective.value(graph, reference_data, batched, state))
    single_values = [
        float(objective.value(graph, reference_data, target, state))
        for target in targets
    ]
    np.testing.assert_allclose(batch_values, single_values, rtol=1e-5)


def test_identity_grams_reduce_to_euclidean_including_bias_terms():
    graph, ref, target = two_layer_graph(seed=9, hidden=3)
    metrics = {
        tensor_id: {
            "input_axes": (0,) if tensor_id.endswith("kernel") else (),
            "gram": np.eye(spec.shape[0])
            if tensor_id.endswith("kernel")
            else np.ones((1, 1)),
        }
        for tensor_id, spec in graph.tensors.items()
    }
    reference_data = graph.materialize(ref, backend="jax")
    target_data = graph.materialize(target, backend="jax")
    state = TransformState.identity(graph, backend="numpy")
    actual = RelativeFisherObjective(tensor_metrics=metrics).value(
        graph, reference_data, target_data, state
    )
    expected = EuclideanObjective().value(graph, reference_data, target_data, state)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_relative_fisher_declines_lap_linearization():
    graph, ref, target = two_layer_graph(seed=0, hidden=3)
    objective = RelativeFisherObjective(tensor_metrics=_metrics(graph))
    state = TransformState.identity(graph, backend="numpy")
    with pytest.raises(ValueError, match="Sinkhorn-only"):
        objective.linearize_group(
            graph,
            graph.materialize(ref, backend="numpy"),
            graph.materialize(target, backend="numpy"),
            state,
            "mlp/h0",
        )


def test_activation_gram_estimation_matches_second_moment_and_normalization():
    graph, ref, _ = two_layer_graph(seed=6, hidden=3)
    inputs = jnp.asarray(np.arange(15, dtype=np.float32).reshape(5, 3) / 7.0)
    metrics = estimate_activation_gram_metrics(
        graph,
        ref,
        mlp_activation_samples,
        inputs,
        damping=0.0,
        normalize=False,
    )
    first = metrics["fcn/Dense_0/kernel"]
    np.testing.assert_allclose(
        first["gram"], np.asarray(inputs).T @ np.asarray(inputs) / len(inputs)
    )


def test_activation_gram_npz_roundtrip(tmp_path):
    graph, _, _ = two_layer_graph(seed=1, hidden=3)
    metrics = _metrics(graph)
    path = tmp_path / "rfim.npz"
    save_activation_gram_metrics_npz(metrics, path)
    loaded = load_activation_gram_metrics_npz(path)
    assert set(loaded) == set(metrics)
    for tensor_id in metrics:
        assert loaded[tensor_id]["input_axes"] == metrics[tensor_id]["input_axes"]
        np.testing.assert_array_equal(
            loaded[tensor_id]["gram"], metrics[tensor_id]["gram"]
        )


def test_relative_fisher_rejects_invalid_input_axis():
    graph, ref, target = two_layer_graph(seed=2, hidden=3)
    metrics = _metrics(graph)
    first = next(iter(metrics))
    metrics[first] = {"input_axes": (99,), "gram": np.ones((1, 1))}
    objective = RelativeFisherObjective(tensor_metrics=metrics)
    with pytest.raises(ValueError, match="out of bounds"):
        objective.value(
            graph,
            graph.materialize(ref, backend="jax"),
            graph.materialize(target, backend="jax"),
            TransformState.identity(graph),
        )


def test_relative_fisher_calibration_runs_through_sinkhorn():
    graph, ref, target = two_layer_graph(seed=8, hidden=3)
    inputs = jnp.asarray(
        np.random.default_rng(0).standard_normal((7, 3)), dtype=jnp.float32
    )
    kwargs = resolve_calibration_kwargs(
        {"calibration": "relative_fisher", "calibration_damping": 0.1},
        graph=graph,
        params=ref,
        apply_fn=lambda params, x: x,
        activation_fn=mlp_activation_samples,
        inputs=inputs,
    )
    aligned, transforms, aux = match_sample(
        graph,
        ref,
        target,
        objective="relative_fisher",
        objective_kwargs=kwargs,
        schedule=[
            {
                "solver": "sinkhorn",
                "max_steps": 2,
                "sinkhorn_iterations": 10,
                "init_scale": 0.0,
            }
        ],
    )
    assert aligned is not None
    assert set(transforms) == {"mlp/h0"}
    assert aux["steps"][0]["solver"] == "sinkhorn"
