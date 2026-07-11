"""Tests for graph matching APIs, state handling, and validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_util

from align.architectures import MLPRecipe, ResidualConvNetRecipe
from align.canonicalization import ScaleState
from align.matching import (
    SolverStep,
    TransformState,
    as_permutation_matrix,
    build_solver_sequence,
    match_batch,
    match_sample,
    sinkhorn_operator,
)
from align.symmetry import (
    AxisBinding,
    ResidualChannelTie,
    SymmetryGraph,
    SymmetryGroup,
    TensorSpec,
)


def _make_params_tree():
    return {
        "params": {
            "fcn": {
                "layer0": {
                    "kernel": jnp.array([[1.0, -0.5], [0.3, -1.2]], dtype=jnp.float32),
                    "bias": jnp.array([0.2, -0.3], dtype=jnp.float32),
                },
                "layer1": {
                    "kernel": jnp.array([[0.4, 0.6], [-0.7, 1.1]], dtype=jnp.float32),
                    "bias": jnp.array([0.0, 1.0], dtype=jnp.float32),
                },
                "layer2": {
                    "kernel": jnp.array([[0.9, -0.2], [0.1, 0.8]], dtype=jnp.float32),
                    "bias": jnp.array([0.5, -0.4], dtype=jnp.float32),
                },
            }
        }
    }


def _flatten_tree(tree):
    leaves, _ = tree_util.tree_flatten(tree)
    return [np.asarray(leaf) for leaf in leaves]


def test_graph_group_order_uses_validated_metadata_order():
    graph = SymmetryGraph(
        groups={
            "b": SymmetryGroup(id="b", size=1),
            "a": SymmetryGroup(id="a", size=1),
        },
        tensors={},
        metadata={"group_order": ("a", "b")},
    )

    assert graph.group_order == ("a", "b")


def test_graph_group_order_rejects_duplicates():
    graph = SymmetryGraph(
        groups={"a": SymmetryGroup(id="a", size=2)},
        tensors={},
        metadata={"group_order": ("a", "a")},
    )
    with pytest.raises(ValueError, match="exactly the graph groups"):
        graph.validate()


def test_lap_matching_recovers_dense_permutation():
    recipe = MLPRecipe(parameter_root="params.fcn")
    reference_params = _make_params_tree()
    graph = recipe.build_graph(reference_params)
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    target = graph.apply_transforms(
        reference_params,
        TransformState.from_transforms(graph, {"mlp/h0": swap, "mlp/h1": swap}),
    )

    aligned, perms, aux = match_sample(
        graph,
        reference_params,
        target,
        schedule=[{"solver": "lap", "max_sweeps": 50, "tolerance": 0.0}],
    )

    assert aux is not None
    np.testing.assert_array_equal(perms["mlp/h0"], np.array([1, 0]))
    np.testing.assert_array_equal(perms["mlp/h1"], np.array([1, 0]))
    for ref_leaf, aligned_leaf in zip(
        _flatten_tree(reference_params), _flatten_tree(aligned), strict=True
    ):
        np.testing.assert_allclose(ref_leaf, aligned_leaf, atol=1e-6)


def test_matching_sinkhorn_batch_matches_single_call():
    reference_params = _make_params_tree()
    recipe = MLPRecipe(parameter_root="params.fcn")
    graph = recipe.build_graph(reference_params)

    swap = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    target_params = graph.apply_transforms(
        reference_params,
        TransformState.from_transforms(graph, {"mlp/h0": swap, "mlp/h1": swap}),
    )

    schedule = [
        {
            "solver": "sinkhorn",
            "max_steps": 50,
            "tolerance": 1e-4,
            "sinkhorn_iterations": 30,
            "init_scale": 0.0,
        }
    ]
    rng_key = jax.random.PRNGKey(0)
    solver_sequence = build_solver_sequence(schedule=schedule)
    reference_data = graph.materialize(
        reference_params, backend=solver_sequence.backend, cache=True
    )
    aligned_single = match_sample(
        graph,
        reference_params,
        target_params,
        solver_sequence=solver_sequence,
        rng_key=rng_key,
        reference_data=reference_data,
        reference_backend=solver_sequence.backend,
        is_reference=False,
    )

    batch_results = match_batch(
        graph,
        reference_params,
        [target_params, target_params],
        solver_sequence=solver_sequence,
        rng_key=rng_key,
        reference_data=reference_data,
        reference_backend=solver_sequence.backend,
    )

    assert len(batch_results) == 2
    single_aligned, single_perms, _ = aligned_single
    batch_folded, batch_perms, _ = batch_results[0]

    for a, b in zip(
        _flatten_tree(single_aligned), _flatten_tree(batch_folded), strict=True
    ):
        np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(single_perms["mlp/h0"]),
        np.asarray(batch_perms["mlp/h0"]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(single_perms["mlp/h1"]),
        np.asarray(batch_perms["mlp/h1"]),
        atol=1e-5,
    )
    second_folded, second_perms, second_aux = batch_results[1]
    for a, b in zip(
        _flatten_tree(single_aligned), _flatten_tree(second_folded), strict=True
    ):
        np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(single_perms["mlp/h0"]),
        np.asarray(second_perms["mlp/h0"]),
        atol=1e-5,
    )
    assert second_aux is not None
    assert isinstance(second_aux["steps"][0]["loss_final"], float)


def test_sinkhorn_batch_rng_is_invariant_to_batch_order():
    reference = _make_params_tree()
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(reference)
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    target = graph.apply_transforms(
        reference,
        TransformState.from_transforms(graph, {"mlp/h0": swap, "mlp/h1": swap}),
    )
    sequence = build_solver_sequence(
        schedule=[
            SolverStep(
                solver="sinkhorn",
                max_steps=2,
                tolerance=0.0,
                init_scale=0.2,
            )
        ]
    )
    key_a = jax.random.PRNGKey(11)
    key_b = jax.random.PRNGKey(29)
    forward = match_batch(
        graph,
        reference,
        [target, reference],
        solver_sequence=sequence,
        rng_keys=jnp.stack([key_a, key_b]),
    )
    reverse = match_batch(
        graph,
        reference,
        [reference, target],
        solver_sequence=sequence,
        rng_keys=jnp.stack([key_b, key_a]),
    )
    for left, right in ((forward[0], reverse[1]), (forward[1], reverse[0])):
        np.testing.assert_array_equal(left[1]["mlp/h0"], right[1]["mlp/h0"])
        assert left[2]["steps"][0]["loss_initial"] == pytest.approx(
            right[2]["steps"][0]["loss_initial"]
        )


def _resnet_params():
    return {
        "core": {
            "Conv_0": {
                "kernel": jnp.array([[[[1.0, 0.0]]]], dtype=jnp.float32),
                "bias": jnp.array([0.1, -0.1], dtype=jnp.float32),
            },
            "Conv_1": {
                "kernel": jnp.array([[[[0.5, -0.5], [1.0, 0.0]]]], dtype=jnp.float32),
                "bias": jnp.array([0.0, 0.2], dtype=jnp.float32),
            },
            "Dense_0": {
                "kernel": jnp.array([[1.0], [-1.0]], dtype=jnp.float32),
                "bias": jnp.array([0.0], dtype=jnp.float32),
            },
        }
    }


def test_sinkhorn_resnet_identity_perm():
    params = _resnet_params()
    recipe = ResidualConvNetRecipe(parameter_root="core")
    graph = recipe.build_graph(params)
    solver_sequence = build_solver_sequence(
        schedule=[
            SolverStep(
                solver="sinkhorn",
                max_steps=10,
                tolerance=1e-4,
                sinkhorn_iterations=20,
            )
        ]
    )
    state, aux = solver_sequence.solve(
        graph,
        graph.materialize(params, backend="jax"),
        graph.materialize(params, backend="jax"),
        rng_key=jax.random.PRNGKey(0),
    )
    assert set(state.transforms) == set(graph.groups)
    assert aux["steps"][0]["steps"] >= 1
    for gid in state.transforms:
        np.testing.assert_allclose(
            np.asarray(state.matrix(gid)), np.eye(graph.groups[gid].size), atol=1e-3
        )


def test_reference_matching_returns_identity_artifacts():
    params = _make_params_tree()
    recipe = MLPRecipe(parameter_root="params.fcn")
    graph = recipe.build_graph(params)
    aligned, perms, aux = match_sample(
        graph,
        params,
        params,
        is_reference=True,
    )

    assert aligned is params
    assert aux == {"reference": True}
    for gid, group in graph.groups.items():
        np.testing.assert_array_equal(perms[gid], np.arange(group.size, dtype=np.int32))


def test_index_vector_state_applies_and_serializes_like_matrix():
    params = _make_params_tree()
    recipe = MLPRecipe(parameter_root="params.fcn")
    graph = recipe.build_graph(params)
    swap_indices = np.array([1, 0], dtype=np.int64)
    swap_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    index_state = TransformState.from_transforms(
        graph,
        {"mlp/h0": swap_indices, "mlp/h1": np.arange(2, dtype=np.int64)},
    )
    matrix_state = TransformState.from_transforms(
        graph,
        {"mlp/h0": swap_matrix, "mlp/h1": np.eye(2, dtype=np.float32)},
    )

    index_state.validate(graph, hard=True)
    index_applied = graph.apply_transforms(params, index_state)
    matrix_applied = graph.apply_transforms(params, matrix_state)
    for lhs, rhs in zip(
        _flatten_tree(index_applied), _flatten_tree(matrix_applied), strict=True
    ):
        np.testing.assert_allclose(lhs, rhs)

    artifacts = index_state.to_artifacts()
    np.testing.assert_array_equal(artifacts["mlp/h0"], swap_indices.astype(np.int32))
    np.testing.assert_array_equal(artifacts["mlp/h1"], np.arange(2, dtype=np.int32))


def test_harden_projects_relaxed_matrices():
    params = _make_params_tree()
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    matrices = {
        "mlp/h0": jnp.array([[0.0, 8.0], [8.0, 0.0]], dtype=jnp.float32),
        "mlp/h1": jnp.array([[8.0, 0.0], [0.0, 8.0]], dtype=jnp.float32),
    }
    state = TransformState.identity(graph).with_relaxation(matrices, matrices)

    hardened = state.harden(groups=graph.group_order)

    np.testing.assert_allclose(
        hardened.matrix("mlp/h0"),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(hardened.matrix("mlp/h1"), np.eye(2), atol=1e-6)


def test_graph_validation_rejects_duplicate_tensor_axis_bindings():
    graph = SymmetryGraph(
        groups={
            "g0": SymmetryGroup(id="g0", size=2),
            "g1": SymmetryGroup(id="g1", size=2),
        },
        tensors={
            "w": TensorSpec(id="w", path=("w",), shape=(2, 2)),
        },
        axis_bindings=(
            AxisBinding(tensor_id="w", axis=0, group="g0"),
            AxisBinding(tensor_id="w", axis=-2, group="g1"),
        ),
    )

    with pytest.raises(ValueError, match="overlaps"):
        graph.validate({"w": np.zeros((2, 2), dtype=np.float32)})


def test_graph_validation_accepts_disjoint_half_open_axis_bindings():
    graph = SymmetryGraph(
        groups={
            "g0": SymmetryGroup(id="g0", size=2),
            "g1": SymmetryGroup(id="g1", size=3),
        },
        tensors={"w": TensorSpec(id="w", path=("w",), shape=(5, 2))},
        axis_bindings=(
            AxisBinding(tensor_id="w", axis=0, group="g0", start=0, stop=2),
            AxisBinding(tensor_id="w", axis=0, group="g1", start=2, stop=5),
        ),
        metadata={"group_order": ("g0", "g1")},
    )
    params = {"w": np.arange(10, dtype=np.float32).reshape(5, 2)}

    graph.validate(params)
    state = TransformState.from_transforms(
        graph,
        {
            "g0": np.array([1, 0], dtype=np.int64),
            "g1": np.array([2, 1, 0], dtype=np.int64),
        },
        backend="numpy",
    )
    applied = graph.apply_transforms(params, state)

    np.testing.assert_allclose(applied["w"], params["w"][[1, 0, 4, 3, 2], :])


def test_graph_validation_rejects_invalid_axis_role():
    graph = SymmetryGraph(
        groups={"g0": SymmetryGroup(id="g0", size=2)},
        tensors={"w": TensorSpec(id="w", path=("w",), shape=(2, 2))},
        axis_bindings=(AxisBinding(tensor_id="w", axis=0, group="g0", role="middle"),),
    )

    with pytest.raises(ValueError, match="invalid role"):
        graph.validate({"w": np.zeros((2, 2), dtype=np.float32)})


def test_graph_reports_repeated_group_tensor_terms():
    graph = SymmetryGraph(
        groups={"g0": SymmetryGroup(id="g0", size=2)},
        tensors={"w": TensorSpec(id="w", path=("w",), shape=(2, 2))},
        axis_bindings=(
            AxisBinding(tensor_id="w", axis=0, group="g0", role="in"),
            AxisBinding(tensor_id="w", axis=1, group="g0", role="out"),
        ),
    )

    graph.validate({"w": np.zeros((2, 2), dtype=np.float32)})
    assert graph.repeated_group_terms() == {"g0": ("w",)}


def test_graph_apply_scales_uses_axis_roles():
    params = _make_params_tree()
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    scales = {
        "mlp/h0": np.array([2.0, 4.0], dtype=np.float32),
        "mlp/h1": np.array([3.0, 5.0], dtype=np.float32),
    }
    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))

    np.testing.assert_allclose(
        np.asarray(scaled["params"]["fcn"]["layer0"]["kernel"]),
        np.asarray(params["params"]["fcn"]["layer0"]["kernel"]) / scales["mlp/h0"],
    )
    np.testing.assert_allclose(
        np.asarray(scaled["params"]["fcn"]["layer0"]["bias"]),
        np.asarray(params["params"]["fcn"]["layer0"]["bias"]) / scales["mlp/h0"],
    )
    np.testing.assert_allclose(
        np.asarray(scaled["params"]["fcn"]["layer1"]["kernel"]),
        np.asarray(params["params"]["fcn"]["layer1"]["kernel"])
        * scales["mlp/h0"][:, None]
        / scales["mlp/h1"],
    )
    np.testing.assert_allclose(
        np.asarray(scaled["params"]["fcn"]["layer2"]["kernel"]),
        np.asarray(params["params"]["fcn"]["layer2"]["kernel"])
        * scales["mlp/h1"][:, None],
    )


def test_graph_validation_rejects_uncanonicalized_residual_channel_tie():
    graph = SymmetryGraph(
        groups={
            "g0": SymmetryGroup(id="g0", size=2),
            "g1": SymmetryGroup(id="g1", size=2),
        },
        tensors={"w": TensorSpec(id="w", path=("w",), shape=(2,))},
        constraints=(
            ResidualChannelTie(
                groups=("g0", "g1"),
                tensors=("w",),
                members=("left", "right"),
                source="test",
            ),
        ),
    )

    with pytest.raises(ValueError, match="canonicalized"):
        graph.validate({"w": np.zeros((2,), dtype=np.float32)})


def test_as_permutation_matrix_round_trips_indices_and_matrices():
    indices = np.array([2, 0, 1], dtype=np.int64)
    matrix = as_permutation_matrix(indices)

    expected = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    np.testing.assert_allclose(matrix, expected)
    # Reading the matrix back yields the same one-hot rows, and the
    # column-argmax recovers the original index vector.
    np.testing.assert_allclose(as_permutation_matrix(matrix), expected)
    np.testing.assert_array_equal(np.argmax(matrix, axis=1), indices)


def test_as_permutation_matrix_rejects_non_permutation_indices():
    with pytest.raises(ValueError, match="rearrangement"):
        as_permutation_matrix(np.array([0, 0, 1], dtype=np.int64))


def test_harden_recovers_nearest_permutation_from_doubly_stochastic_matrix():
    """Hardening a Sinkhorn projection returns the Hungarian-optimal permutation."""

    params = _make_params_tree()
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    target = np.array([1, 0], dtype=np.int64)
    soft = sinkhorn_operator(
        jnp.asarray(8.0 * np.eye(2, dtype=np.float32)[target]),
        tau=0.1,
        n_iters=100,
    )
    state = TransformState.identity(graph).with_relaxation(
        {"mlp/h1": soft}, {"mlp/h1": soft}
    )

    hardened = state.harden(groups=("mlp/h1",))

    assert np.allclose(np.sum(np.asarray(hardened.matrix("mlp/h1")), axis=1), 1.0)
    np.testing.assert_allclose(
        np.asarray(hardened.matrix("mlp/h1")),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
    )
    hardened.validate(graph, hard=True)


def test_state_validate_rejects_non_permutation_hard_matrix():
    params = _make_params_tree()
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    not_a_permutation = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    with pytest.raises(ValueError, match="not a hard permutation"):
        TransformState.from_transforms(
            graph,
            {"mlp/h0": not_a_permutation, "mlp/h1": np.eye(2, dtype=np.float32)},
        )


def test_to_artifacts_emits_int32_permutation_indices():
    params = _make_params_tree()
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    state = TransformState.from_transforms(
        graph,
        {
            "mlp/h0": np.array([1, 0], dtype=np.int64),
            "mlp/h1": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        },
    )

    artifacts = state.to_artifacts()

    for group_id in graph.group_order:
        artifact = artifacts[group_id]
        assert artifact.dtype == np.int32
        np.testing.assert_array_equal(artifact, np.array([1, 0], dtype=np.int32))


def test_transform_artifacts_round_trip_all_families(tmp_path):
    """Every transform family survives write -> load exactly."""

    from align.runtime.artifacts import (
        read_transforms_artifact,
        write_transforms_artifact,
    )

    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    orthogonal, _ = np.linalg.qr(
        np.array([[0.3, 1.2, -0.5], [1.1, -0.4, 0.9], [-0.7, 0.2, 1.3]])
    )
    families = {
        "permutation": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "signed_permutation": np.array([[0.0, -1.0], [1.0, 0.0]]),
        "rotation_pairs": rotation,
        "orthogonal": orthogonal,
    }
    graph = SymmetryGraph(
        groups={
            key: SymmetryGroup(id=key, size=value.shape[0], transform_family=key)
            for key, value in families.items()
        },
        tensors={},
        metadata={"group_order": tuple(families)},
    )
    state = TransformState.from_transforms(graph, families)
    artifacts = state.to_artifacts()

    assert artifacts["permutation"].dtype == np.int32
    assert artifacts["signed_permutation"].dtype == np.int32
    for key in ("rotation_pairs", "orthogonal"):
        assert artifacts[key].dtype == np.float32

    path = write_transforms_artifact(
        tmp_path / "transforms.npz",
        artifacts,
        transform_families={key: key for key in families},
    )
    loaded, metadata = read_transforms_artifact(path)
    assert metadata["transform_families"] == {key: key for key in families}
    assert metadata["representations"] == {
        "permutation": "indices",
        "signed_permutation": "signed_indices",
        "rotation_pairs": "matrix",
        "orthogonal": "matrix",
    }
    np.testing.assert_array_equal(loaded["permutation"], np.array([1, 0]))
    np.testing.assert_array_equal(
        loaded["signed_permutation"], np.array([[1, 0], [-1, 1]])
    )
    for key in ("rotation_pairs", "orthogonal"):
        np.testing.assert_allclose(loaded[key], families[key], atol=1e-6)

    identity_artifacts = TransformState.identity(graph, backend="numpy").to_artifacts()
    identity_path = write_transforms_artifact(
        tmp_path / "identity_transforms.npz",
        identity_artifacts,
        transform_families={key: key for key in families},
    )
    _, identity_metadata = read_transforms_artifact(identity_path)
    assert identity_metadata["representations"]["orthogonal"] == "signed_indices"


def test_transform_artifact_requires_metadata(tmp_path):
    from align.runtime.artifacts import read_transforms_artifact

    path = tmp_path / "transforms.npz"
    np.savez(path, group=np.eye(2, dtype=np.uint8))

    with pytest.raises(ValueError, match="no metadata"):
        read_transforms_artifact(path)


def test_transform_artifact_requires_family_for_every_group(tmp_path):
    from align.runtime.artifacts import write_transforms_artifact

    with pytest.raises(ValueError, match="exactly the persisted groups"):
        write_transforms_artifact(
            tmp_path / "transforms.npz",
            {"group": np.eye(2, dtype=np.uint8)},
        )
