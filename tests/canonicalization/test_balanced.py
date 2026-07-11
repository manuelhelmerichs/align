"""Balanced (minimum-norm) dense scale canonicalization.

``mode="balanced"`` (the dense-chain default) picks the minimum-norm
representative of the positive scale orbit by equalizing each hidden unit's
incoming and outgoing energies — the same canonical-form philosophy as
attention circuit balancing. Unlike ``unit_norm``, it keeps canonical weights
near the data scale, so posterior sample clouds are not inflated when true
unit norms are far from 1.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures import MLPRecipe
from align.canonicalization import ScaleCanonicalizer, ScaleState
from benchmarks.synthetic import (
    make_dense_params,
    make_frn_residual_conv_orbit_case,
    make_layernorm_mha_transformer_orbit_case,
    mlp_apply,
)


def _graph_and_params(seed=0, sizes=(3, 5, 4, 2)):
    params = make_dense_params(key=jax.random.PRNGKey(seed), sizes=sizes)
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    return graph, params


def _tree_max_abs_diff(a, b):
    return max(
        float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
        for x, y in zip(
            jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
        )
    )


def _total_squared_norm(params):
    return sum(
        float(np.sum(np.square(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(params)
    )


def _random_scales(graph, seed, low=-1.0, high=1.0):
    rng = np.random.default_rng(seed)
    return {
        gid: np.exp(rng.uniform(low, high, size=group.size)).astype(np.float32)
        for gid, group in graph.groups.items()
    }


def test_balanced_is_default_and_preserves_function():
    graph, params = _graph_and_params(seed=0)
    canonicalized, _, aux = ScaleCanonicalizer().canonicalize(graph, params)

    assert aux["strategy"] == "balanced"
    assert aux["convergence"]["converged"] is True
    assert aux["convergence"]["iterations"] >= 1
    assert aux["convergence"]["residual"] < 1e-9
    x = jax.random.normal(jax.random.PRNGKey(1), (16, 3))
    np.testing.assert_allclose(
        np.asarray(mlp_apply(canonicalized, x)),
        np.asarray(mlp_apply(params, x)),
        atol=1e-5,
    )


def test_balanced_equalizes_in_out_energies():
    graph, params = _graph_and_params(seed=2)
    canonicalized, _, _ = ScaleCanonicalizer().canonicalize(graph, params)

    fcn = canonicalized["params"]["fcn"]
    names = sorted(fcn, key=lambda n: int(n.split("_")[-1]))
    for producer, consumer in zip(names[:-1], names[1:], strict=True):
        kernel = np.asarray(fcn[producer]["kernel"])
        bias = np.asarray(fcn[producer]["bias"])
        next_kernel = np.asarray(fcn[consumer]["kernel"])
        energy_out = (kernel**2).sum(axis=0) + bias**2
        energy_in = (next_kernel**2).sum(axis=1)
        np.testing.assert_allclose(energy_out, energy_in, rtol=1e-4)


def test_balanced_is_idempotent():
    graph, params = _graph_and_params(seed=3)
    once, _, _ = ScaleCanonicalizer().canonicalize(graph, params)
    twice, _, _ = ScaleCanonicalizer().canonicalize(graph, once)
    assert _tree_max_abs_diff(once, twice) < 1e-6


def test_balanced_collapses_the_scale_orbit():
    graph, params = _graph_and_params(seed=4)
    scaled = graph.apply_scales(
        params, ScaleState.from_scales(graph, _random_scales(graph, seed=5))
    )

    canonical_a, _, _ = ScaleCanonicalizer().canonicalize(graph, params)
    canonical_b, _, _ = ScaleCanonicalizer().canonicalize(graph, scaled)
    assert _tree_max_abs_diff(canonical_a, canonical_b) < 1e-5


def test_balanced_is_minimum_norm():
    """The balanced representative has the smallest total norm in its orbit."""

    graph, params = _graph_and_params(seed=6)
    balanced, _, _ = ScaleCanonicalizer().canonicalize(graph, params)
    unit, _, _ = ScaleCanonicalizer().canonicalize(graph, params, strategy="unit_norm")

    balanced_norm = _total_squared_norm(balanced)
    assert balanced_norm <= _total_squared_norm(params) + 1e-6
    assert balanced_norm <= _total_squared_norm(unit) + 1e-6
    # And it beats random orbit points, not just the two canonical ones.
    for seed in (7, 8):
        other = graph.apply_scales(
            balanced,
            ScaleState.from_scales(
                graph, _random_scales(graph, seed=seed, low=-0.5, high=0.5)
            ),
        )
        assert balanced_norm <= _total_squared_norm(other) + 1e-6


def test_balanced_reported_scales_reproduce_the_action():
    graph, params = _graph_and_params(seed=9)
    canonicalized, scales, _ = ScaleCanonicalizer().canonicalize(graph, params)

    action_scales = scales
    via_action = graph.apply_scales(
        params, ScaleState.from_scales(graph, action_scales)
    )
    assert _tree_max_abs_diff(canonicalized, via_action) < 1e-6


def test_balanced_preserves_degenerate_units():
    graph, params = _graph_and_params(seed=10)
    fcn = params["params"]["fcn"]
    kernel = np.asarray(fcn["Dense_0"]["kernel"]).copy()
    bias = np.asarray(fcn["Dense_0"]["bias"]).copy()
    kernel[:, 1] = 0.0
    bias[1] = 0.0
    fcn["Dense_0"]["kernel"] = jnp.asarray(kernel)
    fcn["Dense_0"]["bias"] = jnp.asarray(bias)

    _, scales, _ = ScaleCanonicalizer().canonicalize(graph, params)
    assert float(np.asarray(scales["mlp/h0"])[1]) == pytest.approx(1.0)


def test_unknown_strategy_is_rejected():
    graph, params = _graph_and_params(seed=11)
    with pytest.raises(ValueError, match="Unknown canonicalize strategy"):
        ScaleCanonicalizer().canonicalize(graph, params, strategy="bogus")


def test_convnet_accepts_balanced_strategy():
    case = make_frn_residual_conv_orbit_case(seed=0)
    _, _, aux = ScaleCanonicalizer().canonicalize(
        case.graph, case.reference, strategy="balanced"
    )
    assert aux["plan"] == "conv_balanced"
    assert aux["strategy"] == "balanced"


def test_attention_rejects_unit_norm_strategy():
    case = make_layernorm_mha_transformer_orbit_case(seed=0)
    with pytest.raises(ValueError, match="undefined for attention"):
        ScaleCanonicalizer().canonicalize(
            case.graph, case.reference, strategy="unit_norm", activation="gelu"
        )
