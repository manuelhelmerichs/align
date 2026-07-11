"""Plain convnet recipe: conv stack -> pool -> flatten -> dense chain.

The flatten boundary binds the last conv's channel group to the first dense
kernel's input axis once per spatial position (channel-fastest flatten
order). These tests verify that the permutation action through that boundary
preserves a real conv/pool/flatten forward, that LAP recovers exact
permutation orbits (multi-slice bindings keep the exact linearization), and
that the scale plans dispatch correctly (producer-energy for ReLU, loud
rejection for GELU).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures import ConvNetRecipe, get_recipe
from align.canonicalization import ScaleCanonicalizer, ScaleState
from align.matching import TransformState, match_sample


def _make_convnet_params(seed=0, channels=(2, 4), hidden=(6, 5), num_classes=3, side=6):
    """Conv(3x3, c0->c1) -> relu -> maxpool(2) -> flatten -> dense chain."""

    rng = np.random.default_rng(seed)
    pooled_side = side // 2
    flat_dim = pooled_side * pooled_side * channels[1]
    core = {
        "conv1": {
            "kernel": rng.standard_normal((3, 3, channels[0], channels[1])).astype(
                np.float32
            ),
            "bias": 0.1 * rng.standard_normal(channels[1]).astype(np.float32),
        },
        "fc1": {
            "kernel": rng.standard_normal((flat_dim, hidden[0])).astype(np.float32)
            / np.sqrt(flat_dim),
            "bias": 0.1 * rng.standard_normal(hidden[0]).astype(np.float32),
        },
        "fc2": {
            "kernel": rng.standard_normal((hidden[0], hidden[1])).astype(np.float32),
            "bias": 0.1 * rng.standard_normal(hidden[1]).astype(np.float32),
        },
        "fc3": {
            "kernel": rng.standard_normal((hidden[1], num_classes)).astype(np.float32),
            "bias": 0.1 * rng.standard_normal(num_classes).astype(np.float32),
        },
    }
    return {"core": core}


def _cnn_apply(params, x, *, activation=jax.nn.relu):
    """NHWC conv (SAME padding) -> act -> 2x2 maxpool -> flatten -> dense chain."""

    core = params["core"]
    h = jax.lax.conv_general_dilated(
        x,
        jnp.asarray(core["conv1"]["kernel"]),
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + jnp.asarray(core["conv1"]["bias"])
    h = activation(h)
    h = jax.lax.reduce_window(
        h, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID"
    )
    h = h.reshape(h.shape[0], -1)
    for name in ("fc1", "fc2"):
        h = activation(h @ jnp.asarray(core[name]["kernel"]) + core[name]["bias"])
    return h @ jnp.asarray(core["fc3"]["kernel"]) + core["fc3"]["bias"]


def _inputs(seed=1, side=6, channels=2, batch=4):
    return jax.random.normal(jax.random.PRNGKey(seed), (batch, side, side, channels))


def _random_perm_state(graph, seed):
    rng = np.random.default_rng(seed)
    perms = {
        gid: np.eye(group.size, dtype=np.uint8)[rng.permutation(group.size)]
        for gid, group in graph.groups.items()
    }
    return TransformState.from_transforms(graph, perms)


def test_structure_and_registration():
    params = _make_convnet_params()
    graph = get_recipe("convnet", parameter_root="core").build_graph(params)

    assert set(graph.components) == {"features", "classifier"}
    assert graph.components["features"].kind == "convnet"
    assert "core/conv1" in graph.groups
    assert {"classifier/h0", "classifier/h1"} <= set(graph.groups)
    # One flatten binding per spatial position on the first dense kernel.
    flatten_bindings = [
        binding
        for binding in graph.bindings_for_tensor("core/fc1/kernel")
        if binding.group == "core/conv1"
    ]
    assert len(flatten_bindings) == 9  # (6/2)^2 positions
    assert graph.metadata["spatial_positions"] == 9
    # Same-axis multi-bindings must stay LAP-linearizable (not a QAP).
    assert "core/conv1" not in graph.repeated_group_terms()


def test_permutation_action_preserves_function():
    params = _make_convnet_params(seed=2)
    graph = ConvNetRecipe(parameter_root="core").build_graph(params)
    state = _random_perm_state(graph, seed=3)
    permuted = graph.apply_transforms(params, state)

    x = _inputs(seed=4)
    np.testing.assert_allclose(
        np.asarray(_cnn_apply(permuted, x)),
        np.asarray(_cnn_apply(params, x)),
        atol=1e-5,
    )


def test_mixed_schedule_recovers_exact_permutation_orbit():
    """Exact orbit recovery through the flatten boundary.

    Plain LAP coordinate descent stalls on some permutation seeds (the conv
    group couples to the dense chain through the flatten slices — the same
    stall class as the FRN residual cycle); the mixed schedule recovers all
    tested seeds exactly.
    """

    params = _make_convnet_params(seed=5)
    graph = ConvNetRecipe(parameter_root="core").build_graph(params)
    target = graph.apply_transforms(params, _random_perm_state(graph, seed=6))

    aligned, _, _ = match_sample(
        graph,
        params,
        target,
        objective="euclidean",
        schedule=[
            {"solver": "lap", "max_sweeps": 25, "tolerance": 0.0},
            {"solver": "sinkhorn", "max_steps": 200, "tau": 0.1, "learning_rate": 0.01},
            {"solver": "lap", "max_sweeps": 5, "tolerance": 0.0},
        ],
        rng_key=jax.random.PRNGKey(0),
    )
    diff = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(
            jax.tree_util.tree_leaves(params),
            jax.tree_util.tree_leaves(aligned),
            strict=True,
        )
    )
    assert diff < 1e-5


def test_multi_binding_linearization_is_exact():
    """The flatten slices sum to an exact linear cost for the conv group."""

    from scipy.optimize import linear_sum_assignment

    from align.matching.objectives import get_objective

    params = _make_convnet_params(seed=5)
    graph = ConvNetRecipe(parameter_root="core").build_graph(params)
    true_state = _random_perm_state(graph, seed=6)
    target = graph.apply_transforms(params, true_state)

    def invert(perm):
        indices = np.asarray(perm).argmax(axis=1)
        inverse = np.zeros_like(np.asarray(perm))
        inverse[indices, np.arange(len(indices))] = 1
        return inverse

    align_perms = {gid: invert(p) for gid, p in true_state.matrices.items()}
    objective = get_objective("euclidean")
    reference_data = graph.materialize(params, backend="numpy")
    target_data = graph.materialize(target, backend="numpy")

    truth = TransformState.from_transforms(graph, align_perms)
    assert float(objective.value(graph, reference_data, target_data, truth)) < 1e-8

    for group_id in graph.groups:
        state = TransformState.from_transforms(
            graph,
            {
                gid: (
                    align_perms[gid]
                    if gid != group_id
                    else np.eye(graph.groups[gid].size, dtype=np.uint8)
                )
                for gid in align_perms
            },
        )
        cost = objective.linearize_group(
            graph, reference_data, target_data, state, group_id
        )
        row, col = linear_sum_assignment(-cost)
        found = np.zeros_like(align_perms[group_id])
        found[row, col] = 1
        assert (found == align_perms[group_id]).all(), group_id


def test_scale_canonicalization_relu_collapses_scale_orbit():
    params = _make_convnet_params(seed=7)
    graph = ConvNetRecipe(parameter_root="core").build_graph(params)
    rng = np.random.default_rng(8)
    scales = {
        gid: np.exp(rng.uniform(-0.5, 0.5, size=group.size)).astype(np.float32)
        for gid, group in graph.groups.items()
    }
    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))

    x = _inputs(seed=9)
    np.testing.assert_allclose(
        np.asarray(_cnn_apply(scaled, x)),
        np.asarray(_cnn_apply(params, x)),
        atol=1e-4,
    )

    canonicalizer = ScaleCanonicalizer()
    canonical_a, _, aux = canonicalizer.canonicalize(graph, params, activation="relu")
    canonical_b, _, _ = canonicalizer.canonicalize(graph, scaled, activation="relu")
    assert aux["plan"] == "conv_balanced"
    diff = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(
            jax.tree_util.tree_leaves(canonical_a),
            jax.tree_util.tree_leaves(canonical_b),
            strict=True,
        )
    )
    assert diff < 1e-4


def test_scale_canonicalization_rejects_gelu():
    params = _make_convnet_params(seed=10)
    graph = ConvNetRecipe(parameter_root="core").build_graph(params)
    with pytest.raises(ValueError, match="positively.*homogeneous|homogeneous"):
        ScaleCanonicalizer().canonicalize(graph, params, activation="gelu")


def test_flatten_mismatch_fails_loud():
    params = _make_convnet_params(seed=11)
    fc1 = np.asarray(params["core"]["fc1"]["kernel"])
    params["core"]["fc1"]["kernel"] = fc1[:-1]  # break divisibility
    with pytest.raises(ValueError, match="Flatten boundary mismatch"):
        ConvNetRecipe(parameter_root="core").build_graph(params)


def test_pure_dense_tree_is_rejected():
    params = {"core": {"fc1": {"kernel": np.zeros((4, 3)), "bias": np.zeros(3)}}}
    with pytest.raises(ValueError, match="No conv layers"):
        ConvNetRecipe(parameter_root="core").build_graph(params)
