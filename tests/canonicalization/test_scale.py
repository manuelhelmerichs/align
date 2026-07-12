"""Tests for the positive-scale graph symmetry and its canonicalization.

The scale action (``SymmetryGraph.apply_scales`` / ``ScaleCanonicalizer``) is the
deterministic counterpart of matching: dividing producer (``out``) axes and
multiplying consumer (``in``) axes by a positive per-channel scale is an exact
function-preserving symmetry of positively-homogeneous (ReLU) MLPs. These tests
pin that invariance, the canonicalizer's idempotence and scale-invariance, and
``ScaleState`` validation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures import MLPRecipe
from align.canonicalization import ScaleCanonicalizer, ScaleState
from align.runtime.artifacts import read_scales_artifact, write_scales_artifact


def _mlp_params(seed: int, sizes=(3, 5, 4, 2)):
    rng = np.random.default_rng(seed)
    fcn = {}
    for idx, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        fcn[f"Dense_{idx}"] = {
            "kernel": jnp.asarray(
                0.7 * rng.standard_normal((din, dout)), dtype=jnp.float32
            ),
            "bias": jnp.asarray(0.4 * rng.standard_normal((dout,)), dtype=jnp.float32),
        }
    return {"params": {"fcn": fcn}}


def _forward(params, x):
    """ReLU MLP with align's ``z = h @ W + b`` convention."""

    fcn = params["params"]["fcn"]
    names = sorted(fcn)
    h = x
    for name in names[:-1]:
        h = jnp.maximum(0.0, h @ fcn[name]["kernel"] + fcn[name]["bias"])
    last = fcn[names[-1]]
    return h @ last["kernel"] + last["bias"]


def _positive_scales(graph, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        gid: (0.3 + np.abs(rng.standard_normal(group.size))).astype(np.float32)
        for gid, group in graph.groups.items()
    }


def _tree_max_abs_diff(left, right) -> float:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return max(
        float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _canonicalize(graph, params, **kwargs):
    kwargs.setdefault("activation", "relu")
    return ScaleCanonicalizer().canonicalize(graph, params, **kwargs)


def test_apply_scales_preserves_relu_mlp_function():
    params = _mlp_params(seed=0)
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    scales = _positive_scales(graph, seed=1)

    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))

    x = jnp.asarray(np.random.default_rng(2).standard_normal((9, 3)), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(_forward(scaled, x)),
        np.asarray(_forward(params, x)),
        atol=1e-5,
    )


def test_apply_scales_inverse_round_trips_to_identity():
    params = _mlp_params(seed=3)
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    scales = _positive_scales(graph, seed=4)
    inverse = {gid: 1.0 / vec for gid, vec in scales.items()}

    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))
    restored = graph.apply_scales(scaled, ScaleState.from_scales(graph, inverse))

    assert _tree_max_abs_diff(params, restored) < 1e-5


def test_scale_canonicalizer_preserves_function_and_is_idempotent():
    params = _mlp_params(seed=5)
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)

    canonicalized, _, _ = _canonicalize(graph, params)
    twice, _, _ = _canonicalize(graph, canonicalized)

    x = jnp.asarray(np.random.default_rng(6).standard_normal((8, 3)), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(_forward(canonicalized, x)),
        np.asarray(_forward(params, x)),
        atol=1e-5,
    )
    # Canonical form is a fixed point of the canonicalizer.
    assert _tree_max_abs_diff(canonicalized, twice) < 1e-5


def test_scale_canonicalizer_is_invariant_to_prior_scaling():
    """Two scale-equivalent networks canonicalize to the same canonical weights."""

    params = _mlp_params(seed=7)
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    scales = _positive_scales(graph, seed=8)
    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))

    canonical_a, _, _ = _canonicalize(graph, params)
    canonical_b, _, _ = _canonicalize(graph, scaled)

    assert _tree_max_abs_diff(canonical_a, canonical_b) < 1e-5


def test_scale_canonicalizer_unit_norm_hidden_neurons():
    params = _mlp_params(seed=9)
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)

    canonicalized, _, _ = _canonicalize(graph, params, strategy="unit_norm")

    fcn = canonicalized["params"]["fcn"]
    for name in ("Dense_0", "Dense_1"):  # hidden producers only
        kernel = np.asarray(fcn[name]["kernel"])
        bias = np.asarray(fcn[name]["bias"])
        norms = np.sqrt(np.sum(kernel**2, axis=0) + bias**2)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


def test_scale_canonicalizer_matches_explicit_scale_action_when_non_degenerate():
    params = _mlp_params(seed=14)
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)

    canonicalized, scales, _ = _canonicalize(graph, params)
    assert all(np.all(np.asarray(scale) > 1e-4) for scale in scales.values())

    action_scales = scales
    via_action = graph.apply_scales(
        params,
        ScaleState.from_scales(graph, action_scales),
    )

    assert _tree_max_abs_diff(canonicalized, via_action) < 1e-6


def test_scale_artifact_retains_ids_and_reconstructs_action(tmp_path):
    params = _mlp_params(seed=15)
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    canonicalized, scales, _ = _canonicalize(graph, params)

    path = write_scales_artifact(tmp_path / "scales.npz", scales)
    loaded, metadata = read_scales_artifact(path)

    assert metadata["scale_ids"] == list(scales)
    assert set(loaded) == set(scales) == set(graph.group_order)
    reconstructed = graph.apply_scales(
        params,
        ScaleState(group_order=graph.group_order, scales=loaded),
    )
    assert _tree_max_abs_diff(canonicalized, reconstructed) < 1e-6


class TestScaleStateValidation:
    def _graph(self):
        params = _mlp_params(seed=10)
        return MLPRecipe(parameter_root="params.fcn").build_graph(params)

    def test_identity_state_validates(self):
        graph = self._graph()
        ScaleState.identity(graph).validate(graph)

    def test_rejects_non_positive_scale(self):
        graph = self._graph()
        scales = _positive_scales(graph, seed=11)
        first = next(iter(scales))
        scales[first] = scales[first].copy()
        scales[first][0] = 0.0
        state = ScaleState.from_scales(graph, scales)
        with pytest.raises(ValueError, match="strictly positive"):
            state.validate(graph)

    def test_rejects_wrong_shape_scale(self):
        graph = self._graph()
        scales = _positive_scales(graph, seed=12)
        first = next(iter(scales))
        scales[first] = np.ones(scales[first].shape[0] + 1, dtype=np.float32)
        state = ScaleState.from_scales(graph, scales)
        with pytest.raises(ValueError, match="shape"):
            state.validate(graph)

    def test_rejects_missing_group(self):
        graph = self._graph()
        scales = _positive_scales(graph, seed=13)
        scales.pop(next(iter(scales)))
        state = ScaleState(group_order=graph.group_order, scales=scales)
        with pytest.raises(ValueError, match="exactly the graph groups"):
            state.validate(graph)
