"""Tests that graph actions preserve model functions on synthetic networks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures.mlp import MLPRecipe
from align.architectures.residual_convnet import ResidualConvNetRecipe
from align.matching import TransformState, match_sample
from align.samples import tree_leaves_with_names


def calculate_r_hat(
    flattened: np.ndarray,
    chain_labels: np.ndarray,
    num_chains: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Gelman-Rubin R-hat without pulling in visualization deps."""
    overall_mean = np.mean(flattened, axis=0)
    within_chain_vars = []
    chain_means = []
    for chain_id in range(num_chains):
        chain_samples = flattened[chain_labels == chain_id]
        if len(chain_samples) < 2:
            continue
        chain_var = np.var(chain_samples, axis=0, ddof=1)
        within_chain_vars.append(chain_var)
        chain_mean = np.mean(chain_samples, axis=0)
        chain_means.append((chain_mean - overall_mean) ** 2)
    within_chain_var = np.mean(within_chain_vars, axis=0)
    chain_means = np.array(chain_means)
    M = num_chains
    N = len(flattened) / M
    between_chain_var = np.sum(chain_means, axis=0) * (N / max(M - 1, 1))
    v_hat = ((N - 1) / N) * within_chain_var + ((M + 1) / (M * N)) * between_chain_var
    r_hat = np.sqrt(v_hat / within_chain_var)
    return within_chain_var, between_chain_var, r_hat


def _flatten_params(params) -> np.ndarray:
    _, leaves, _ = tree_leaves_with_names(params)
    flat = np.concatenate([np.ravel(np.asarray(leaf)) for leaf in leaves], axis=0)
    return flat


def _perm_matrix(indices: np.ndarray) -> np.ndarray:
    """Return a permutation matrix P with P[i, indices[i]] = 1."""
    indices = np.asarray(indices, dtype=int)
    n = int(indices.shape[0])
    mat = np.zeros((n, n), dtype=np.float64)
    mat[np.arange(n), indices] = 1.0
    return mat


def _mlp_apply(params, x: jnp.ndarray) -> jnp.ndarray:
    subtree = params["params"]["fcn"]
    h = x
    layer_names = sorted(subtree.keys())
    for name in layer_names[:-1]:
        w = jnp.asarray(subtree[name]["kernel"])
        b = jnp.asarray(subtree[name]["bias"])
        # Match align's convention (align.md): z = W^T h + b.
        z = (w.T @ h.T).T + b
        h = jnp.maximum(0.0, z)
    last = layer_names[-1]
    w = jnp.asarray(subtree[last]["kernel"])
    b = jnp.asarray(subtree[last]["bias"])
    return (w.T @ h.T).T + b


def _make_mlp_params(
    *, key: jax.Array, in_dim: int, hidden: tuple[int, ...], out_dim: int
):
    sizes = (in_dim, *hidden, out_dim)
    keys = jax.random.split(key, len(sizes) - 1)
    fcn = {}
    for idx, (k, din, dout) in enumerate(zip(keys, sizes[:-1], sizes[1:], strict=True)):
        w_key, b_key = jax.random.split(k)
        # Make weights fairly distinct to make matching stable.
        w = jax.random.normal(w_key, (din, dout), dtype=jnp.float32) * 0.7
        b = jax.random.normal(b_key, (dout,), dtype=jnp.float32) * 0.1
        fcn[f"dense{idx}"] = {"kernel": w, "bias": b}
    return {"params": {"fcn": fcn}}


def _rhat_from_predictions(
    preds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # preds: [C, S, N, D]
    # Flatten prediction space into features so calculate_r_hat sees shape:
    #   flattened: [C*S, N*D]
    values = preds.reshape(preds.shape[0] * preds.shape[1], -1)
    chain_labels = np.repeat(np.arange(preds.shape[0]), preds.shape[1])
    return calculate_r_hat(values, chain_labels=chain_labels, num_chains=preds.shape[0])


def test_residual_dag_action_preserves_fanout_projection_function_when_reordered():
    rng = np.random.default_rng(91)

    def module(din, dout):
        return {
            "kernel": rng.normal(size=(1, 1, din, dout)).astype(np.float32),
            "bias": rng.normal(size=(dout,)).astype(np.float32),
        }

    # Deliberately unrelated insertion/name order: topology, not mapping order,
    # defines the stem -> two-branch fan-out -> residual add dataflow.
    params = {
        "core": {
            "Conv_10": module(3, 3),
            "Dense_0": {
                "kernel": rng.normal(size=(3, 2)).astype(np.float32),
                "bias": rng.normal(size=(2,)).astype(np.float32),
            },
            "Conv_0": module(2, 3),
            "Conv_2": module(3, 3),
        }
    }
    topology = {
        "nodes": [
            {"id": "input", "kind": "input"},
            {"id": "core/Conv_0", "kind": "conv"},
            {"id": "core/Conv_10", "kind": "conv"},
            {"id": "core/Conv_2", "kind": "conv"},
            {"id": "residual_add", "kind": "add"},
            {"id": "core/Dense_0", "kind": "dense"},
        ],
        "edges": [
            {"source": "input", "target": "core/Conv_0"},
            {"source": "core/Conv_0", "target": "core/Conv_10"},
            {"source": "core/Conv_0", "target": "core/Conv_2"},
            {"source": "core/Conv_10", "target": "residual_add"},
            {"source": "core/Conv_2", "target": "residual_add"},
            {"source": "residual_add", "target": "core/Dense_0"},
        ],
    }
    graph = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=topology
    ).build_graph(params)

    def apply(tree, x):
        core = tree["core"]

        def conv(name, value):
            layer = core[name]
            return value @ jnp.asarray(layer["kernel"])[0, 0] + jnp.asarray(
                layer["bias"]
            )

        stem = jax.nn.relu(conv("Conv_0", x))
        merged = jax.nn.relu(conv("Conv_10", stem) + conv("Conv_2", stem))
        dense = core["Dense_0"]
        return merged @ jnp.asarray(dense["kernel"]) + jnp.asarray(dense["bias"])

    transforms = {
        group_id: _perm_matrix(rng.permutation(group.size))
        for group_id, group in graph.groups.items()
    }
    transformed = graph.apply_transforms(
        params, TransformState.from_transforms(graph, transforms)
    )
    x = jnp.asarray(rng.normal(size=(7, 2)), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(apply(transformed, x)), np.asarray(apply(params, x)), atol=1e-5
    )


def test_residual_dag_with_more_than_ten_convs_uses_topological_order():
    rng = np.random.default_rng(92)
    modules = {
        f"Conv_{index}": {
            "kernel": rng.normal(size=(1, 1, 2, 2)).astype(np.float32),
            "bias": rng.normal(size=(2,)).astype(np.float32),
        }
        for index in reversed(range(12))
    }
    modules["Dense_0"] = {
        "kernel": rng.normal(size=(2, 1)).astype(np.float32),
        "bias": rng.normal(size=(1,)).astype(np.float32),
    }
    nodes = (
        [{"id": "input", "kind": "input"}]
        + [{"id": f"core/Conv_{index}", "kind": "conv"} for index in range(12)]
        + [{"id": "core/Dense_0", "kind": "dense"}]
    )
    edges = (
        [{"source": "input", "target": "core/Conv_0"}]
        + [
            {"source": f"core/Conv_{index}", "target": f"core/Conv_{index + 1}"}
            for index in range(11)
        ]
        + [{"source": "core/Conv_11", "target": "core/Dense_0"}]
    )
    graph = ResidualConvNetRecipe(
        parameter_root="core",
        residual_topology={
            "nodes": nodes,
            "edges": edges,
        },
    ).build_graph({"core": modules})
    assert graph.group_order == tuple(f"core/Conv_{index}" for index in range(12))


@pytest.mark.parametrize(
    "schedule",
    [
        [{"solver": "lap", "max_sweeps": 25, "tolerance": 0.0}],
        [
            {
                "solver": "sinkhorn",
                "max_steps": 25,
                "tolerance": 1e-4,
                "init_scale": 0.0,
            },
            {"solver": "lap", "max_sweeps": 5, "tolerance": 0.0},
        ],
    ],
)
def test_mlp_alignment_preserves_predictions_and_function_rhat(schedule):

    num_chains = 2
    num_samples = 6
    in_dim = 3
    hidden = (5, 4)
    out_dim = 2

    key = jax.random.PRNGKey(0)
    base_key, noise_key, x_key = jax.random.split(key, 3)

    reference_params = _make_mlp_params(
        key=base_key, in_dim=in_dim, hidden=hidden, out_dim=out_dim
    )

    recipe = MLPRecipe()
    graph = recipe.build_graph(reference_params)

    # Create per-sample perturbations around ref; chain1 is a fixed permutation of chain0.
    perm0 = np.array([2, 4, 1, 0, 3], dtype=np.int32)  # size=5
    pmat0 = _perm_matrix(perm0)

    # X for prediction checks
    x = jax.random.normal(x_key, (11, in_dim), dtype=jnp.float32)

    chain0 = []
    chain1 = []
    for s in range(num_samples):
        k = jax.random.fold_in(noise_key, s)
        # small noise so weight-matching remains stable
        delta = jax.tree_util.tree_map(
            lambda a, key=k: jax.random.normal(key, a.shape, dtype=jnp.float32) * 0.02,
            reference_params,
        )
        sample0 = jax.tree_util.tree_map(lambda a, d: a + d, reference_params, delta)
        sample1 = graph.apply_transforms(
            sample0, TransformState.from_transforms(graph, {"mlp/h0": pmat0})
        )
        chain0.append(sample0)
        chain1.append(sample1)

    # Predictions before alignment
    preds_before = []
    for chain in (chain0, chain1):
        chain_preds = [np.asarray(_mlp_apply(p, x)) for p in chain]
        preds_before.append(np.stack(chain_preds, axis=0))
    preds_before = np.stack(preds_before, axis=0)  # [C,S,N,D]

    # Align chain1 samples to ref.
    aligned_chain0 = chain0
    aligned_chain1 = []
    for p in chain1:
        aligned, _, _ = match_sample(
            graph,
            reference_params,
            p,
            schedule=schedule,
            rng_key=jax.random.PRNGKey(123),
        )
        aligned_chain1.append(aligned)

    # Predictions after alignment must match per-sample.
    preds_after = []
    for chain in (aligned_chain0, aligned_chain1):
        chain_preds = [np.asarray(_mlp_apply(p, x)) for p in chain]
        preds_after.append(np.stack(chain_preds, axis=0))
    preds_after = np.stack(preds_after, axis=0)

    np.testing.assert_allclose(preds_after, preds_before, rtol=0, atol=1e-5)

    # Function-space R-hat must be identical.
    w0, b0, r0 = _rhat_from_predictions(preds_before)
    w1, b1, r1 = _rhat_from_predictions(preds_after)
    # Use a small tolerance because calculate_r_hat operates in float32 here,
    # and minor numerical differences are expected even when predictions match.
    np.testing.assert_allclose(w1, w0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(b1, b0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(r1, r0, rtol=0, atol=1e-7)

    # Weight-space R-hat should improve after alignment when chains differ by permutation.
    flat_before = np.stack([_flatten_params(p) for p in (chain0 + chain1)], axis=0)
    flat_after = np.stack(
        [_flatten_params(p) for p in (aligned_chain0 + aligned_chain1)], axis=0
    )
    chain_labels = np.array([0] * num_samples + [1] * num_samples)

    within0, between0, rhat0 = calculate_r_hat(
        flat_before, chain_labels, num_chains=num_chains
    )
    within1, between1, rhat1 = calculate_r_hat(
        flat_after, chain_labels, num_chains=num_chains
    )

    assert float(np.mean(between1)) < float(np.mean(between0))
    assert float(np.mean(rhat1)) < float(np.mean(rhat0))


def _conv1x1(x: jnp.ndarray, kernel: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    # x: [B,H,W,Cin], kernel: [1,1,Cin,Cout], bias: [Cout]
    w = kernel[0, 0, :, :]
    # Match align's convention: y = W^T x + b (per pixel).
    y = jnp.einsum("oc,bhwc->bhwo", w.T, x) + bias
    return y


def _tiny_residual_convnet_apply(params, x: jnp.ndarray) -> jnp.ndarray:
    core = params["core"]
    h = _conv1x1(
        x, jnp.asarray(core["Conv_0"]["kernel"]), jnp.asarray(core["Conv_0"]["bias"])
    )
    h = jnp.maximum(0.0, h)
    h = _conv1x1(
        h, jnp.asarray(core["Conv_1"]["kernel"]), jnp.asarray(core["Conv_1"]["bias"])
    )
    h = jnp.maximum(0.0, h)
    h = jnp.mean(h, axis=(1, 2))
    w = jnp.asarray(core["Dense_0"]["kernel"])
    b = jnp.asarray(core["Dense_0"]["bias"])
    return (w.T @ h.T).T + b


@pytest.mark.parametrize(
    "schedule",
    [
        [{"solver": "lap", "max_sweeps": 25, "tolerance": 0.0}],
        [
            {
                "solver": "sinkhorn",
                "max_steps": 25,
                "tolerance": 1e-4,
                "init_scale": 0.0,
            },
            {"solver": "lap", "max_sweeps": 5, "tolerance": 0.0},
        ],
    ],
)
def test_residual_convnet_matching_preserves_predictions(schedule):

    # Match the tiny residual_convnet tree shape used by tests/implementation/test_stages.py.
    key = jax.random.PRNGKey(7)
    k0, k1, kd, xk = jax.random.split(key, 4)

    ref = {
        "core": {
            "Conv_0": {
                "kernel": jax.random.normal(k0, (1, 1, 1, 2), dtype=jnp.float32),
                "bias": jax.random.normal(k0, (2,), dtype=jnp.float32) * 0.1,
            },
            "Conv_1": {
                "kernel": jax.random.normal(k1, (1, 1, 2, 2), dtype=jnp.float32),
                "bias": jax.random.normal(k1, (2,), dtype=jnp.float32) * 0.1,
            },
            "Dense_0": {
                "kernel": jax.random.normal(kd, (2, 1), dtype=jnp.float32),
                "bias": jax.random.normal(kd, (1,), dtype=jnp.float32) * 0.1,
            },
        }
    }

    recipe = ResidualConvNetRecipe(linear_residual_free=True)
    graph = recipe.build_graph(ref)

    # Create a target that's a pure channel permutation of Conv_0 outputs (size=2)
    perm = np.array([1, 0], dtype=np.int32)
    group_id = graph.metadata["group_order"][0]
    target = graph.apply_transforms(
        ref, TransformState.from_transforms(graph, {group_id: _perm_matrix(perm)})
    )

    x = jax.random.normal(xk, (5, 3, 3, 1), dtype=jnp.float32)

    pred_before = np.asarray(_tiny_residual_convnet_apply(target, x))

    aligned, _, _ = match_sample(
        graph,
        ref,
        target,
        schedule=schedule,
        rng_key=jax.random.PRNGKey(9),
    )

    pred_after = np.asarray(_tiny_residual_convnet_apply(aligned, x))
    np.testing.assert_allclose(pred_after, pred_before, rtol=0, atol=1e-5)
