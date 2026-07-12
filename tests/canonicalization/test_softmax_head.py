"""Softmax-head translation centering: probability-preserving post-processing.

Predictive probabilities are invariant under adding one vector to every class
column of the head kernel and one constant to every bias entry; logits are
not. Centering to zero-mean class columns is the minimum-norm representative
of that translation orbit, exactly idempotent and translation-equivariant.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures import MLPRecipe, ResidualConvNetRecipe
from align.canonicalization.softmax_head import (
    center_softmax_head,
    detect_softmax_head,
)
from benchmarks.synthetic import make_dense_params, mlp_apply


def _case(seed=0, sizes=(3, 5, 4, 3)):
    params = make_dense_params(key=jax.random.PRNGKey(seed), sizes=sizes)
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    return graph, params


def _translate_head(params, head_path, v, c):
    import copy

    translated = copy.deepcopy(params)
    module = translated
    for key in head_path:
        module = module[key]
    module["kernel"] = jnp.asarray(module["kernel"]) + v[:, None]
    module["bias"] = jnp.asarray(module["bias"]) + c
    return translated


def test_centering_preserves_probabilities_but_not_logits():
    graph, params = _case(seed=0)
    head_path = detect_softmax_head(graph)

    centered, diagnostics = center_softmax_head(params, head_path)
    x = jax.random.normal(jax.random.PRNGKey(1), (16, 3))
    logits_before = np.asarray(mlp_apply(params, x))
    logits_after = np.asarray(mlp_apply(centered, x))

    np.testing.assert_allclose(
        np.asarray(jax.nn.softmax(logits_after, axis=-1)),
        np.asarray(jax.nn.softmax(logits_before, axis=-1)),
        atol=1e-6,
    )
    assert np.max(np.abs(logits_before - logits_after)) > 1e-3
    assert diagnostics["plan"] == "softmax_head_translation"
    assert diagnostics["kernel_translation_norm"] > 0.0


def test_centering_zero_means_idempotent_and_translation_equivariant():
    graph, params = _case(seed=2)
    head_path = detect_softmax_head(graph)
    centered, _ = center_softmax_head(params, head_path)

    module = centered
    for key in head_path:
        module = module[key]
    np.testing.assert_allclose(
        np.mean(np.asarray(module["kernel"]), axis=1),
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.mean(np.asarray(module["bias"])), 0.0, atol=1e-6)

    twice, diagnostics = center_softmax_head(centered, head_path)
    for before, after in zip(
        jax.tree_util.tree_leaves(centered),
        jax.tree_util.tree_leaves(twice),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(before), np.asarray(after), atol=1e-6)
    assert diagnostics["kernel_translation_norm"] < 1e-6

    rng = np.random.default_rng(3)
    features = np.asarray(module["kernel"]).shape[0]
    translated = _translate_head(
        params,
        head_path,
        jnp.asarray(rng.standard_normal(features), jnp.float32),
        jnp.float32(rng.standard_normal()),
    )
    centered_b, _ = center_softmax_head(translated, head_path)
    for left, right in zip(
        jax.tree_util.tree_leaves(centered),
        jax.tree_util.tree_leaves(centered_b),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-5)


def test_centering_is_minimum_norm_in_the_translation_orbit():
    graph, params = _case(seed=4)
    head_path = detect_softmax_head(graph)
    centered, _ = center_softmax_head(params, head_path)

    def _head_norm(tree):
        module = tree
        for key in head_path:
            module = module[key]
        return float(np.sum(np.asarray(module["kernel"]) ** 2)) + float(
            np.sum(np.asarray(module["bias"]) ** 2)
        )

    canonical_norm = _head_norm(centered)
    assert canonical_norm <= _head_norm(params) + 1e-6
    rng = np.random.default_rng(5)
    module = params
    for key in head_path:
        module = module[key]
    n_features = np.asarray(module["kernel"]).shape[0]
    for _ in range(5):
        other = _translate_head(
            centered,
            head_path,
            jnp.asarray(0.5 * rng.standard_normal(n_features), jnp.float32),
            jnp.float32(0.5 * rng.standard_normal()),
        )
        assert canonical_norm < _head_norm(other)


def test_detection_finds_the_unique_head():
    graph, params = _case(seed=6)
    head_path = detect_softmax_head(graph)
    fcn = params["params"]["fcn"]
    last = sorted(fcn, key=lambda name: int(name.split("_")[-1]))[-1]
    assert head_path == ("params", "fcn", last)


def test_detection_rejects_ambiguous_heads():
    rng = np.random.default_rng(7)

    def _conv(cin, cout):
        return {
            "kernel": jnp.asarray(
                rng.standard_normal((1, 1, cin, cout)), dtype=jnp.float32
            ),
            "bias": jnp.asarray(0.1 * rng.standard_normal((cout,)), jnp.float32),
        }

    def _dense(cin, cout):
        return {
            "kernel": jnp.asarray(rng.standard_normal((cin, cout)), jnp.float32),
            "bias": jnp.asarray(0.1 * rng.standard_normal((cout,)), jnp.float32),
        }

    params = {
        "core": {
            "Conv_0": _conv(2, 4),
            "Dense_0": _dense(4, 3),
            "Dense_1": _dense(4, 3),
        }
    }
    graph = ResidualConvNetRecipe(
        parameter_root="core", linear_residual_free=True
    ).build_graph(params)
    with pytest.raises(ValueError, match="found 2"):
        detect_softmax_head(graph)


def test_center_softmax_head_validates_module():
    graph, params = _case(seed=8)
    with pytest.raises(ValueError, match="no 'kernel'"):
        center_softmax_head(params, ("params", "fcn"))
