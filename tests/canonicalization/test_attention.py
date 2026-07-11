"""Tests for the attention qk/vo diagonal scale symmetry and its balancing.

Attention scores depend on q/k only through ``QK_i = W_i^Q (W_i^K)^T`` and the
output on v/out only through ``OV_i = W_i^V W_i^O``, so per-intra-head-dim
positive diagonal rescalings (query/value divided, key/out multiplied) are an
exact, activation-independent function symmetry. These tests pin that the
graph action realizes the symmetry (binding roles), and that the balancing
canonicalization is function-preserving, idempotent, orbit-collapsing, and
actually balances the circuit energies.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures import LayerNormMHATransformerRecipe
from align.canonicalization import ScaleCanonicalizer, ScaleState
from align.canonicalization.mha import (
    attention_balancing_scales,
    mha_circuit_constraints,
)
from benchmarks.synthetic import (
    layernorm_mha_transformer_apply,
    make_layernorm_mha_transformer_params,
    make_layernorm_mha_transformer_scaled_orbit_case,
    run_alignment_benchmark,
)


def _case(seed: int = 0):
    params = make_layernorm_mha_transformer_params(key=jax.random.PRNGKey(seed))
    graph = LayerNormMHATransformerRecipe().build_graph(params)
    tokens = jax.random.randint(
        jax.random.PRNGKey(seed + 100), (4, 5), 0, 7, dtype=jnp.int32
    )
    return graph, params, tokens


def _intra_scales(graph, seed: int) -> dict[str, np.ndarray]:
    """Random positive scales on qk/vo groups only (identity elsewhere)."""

    rng = np.random.default_rng(seed)
    scales: dict[str, np.ndarray] = {}
    for constraint in mha_circuit_constraints(graph):
        for group_id in (*constraint.qk_groups, *constraint.vo_groups):
            size = graph.groups[group_id].size
            scales[group_id] = np.exp(rng.uniform(-1.5, 1.5, size=size)).astype(
                np.float32
            )
    return scales


def _tree_max_abs_diff(left, right) -> float:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return max(
        float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def test_intra_head_binding_roles_pair_divide_and_multiply():
    """query/value intra bindings are producers, key/out are consumers."""

    graph, _, _ = _case()
    (constraint, *_rest) = mha_circuit_constraints(graph)
    tensor_ids = {
        role: getattr(constraint, role) for role in ("query", "key", "value", "out")
    }
    intra_groups = set(constraint.qk_groups) | set(constraint.vo_groups)
    expected_roles = {"query": "out", "value": "out", "key": "in", "out": "in"}
    for name, expected in expected_roles.items():
        roles = {
            binding.role
            for binding in graph.bindings_for_tensor(tensor_ids[name])
            if binding.group in intra_groups
        }
        assert roles == {expected}, (name, roles)


def test_apply_scales_on_intra_groups_preserves_transformer_function():
    graph, params, tokens = _case(seed=0)
    scales = _intra_scales(graph, seed=1)

    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))

    np.testing.assert_allclose(
        np.asarray(layernorm_mha_transformer_apply(scaled, tokens)),
        np.asarray(layernorm_mha_transformer_apply(params, tokens)),
        atol=1e-4,
    )
    # The action must actually change the weights (guard against a no-op).
    assert _tree_max_abs_diff(scaled, params) > 1e-3


def test_balancing_canonicalizer_preserves_function_and_is_idempotent():
    graph, params, tokens = _case(seed=2)

    canonicalized, factors, aux = ScaleCanonicalizer().canonicalize(
        graph, params, activation="gelu"
    )
    twice, _, _ = ScaleCanonicalizer().canonicalize(
        graph, canonicalized, activation="gelu"
    )

    assert aux["plan"] == "attention_balance"
    assert factors, "expected balancing factors for qk/vo groups"
    np.testing.assert_allclose(
        np.asarray(layernorm_mha_transformer_apply(canonicalized, tokens)),
        np.asarray(layernorm_mha_transformer_apply(params, tokens)),
        atol=1e-4,
    )
    assert _tree_max_abs_diff(canonicalized, twice) < 1e-5


def test_balancing_canonicalizer_collapses_scale_orbit():
    """Two diagonal-scale copies of one network canonicalize to identical trees."""

    graph, params, _ = _case(seed=3)
    scales = _intra_scales(graph, seed=4)
    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))

    canonical_a, _, _ = ScaleCanonicalizer().canonicalize(
        graph, params, activation="gelu"
    )
    canonical_b, _, _ = ScaleCanonicalizer().canonicalize(
        graph, scaled, activation="gelu"
    )

    assert _tree_max_abs_diff(canonical_a, canonical_b) < 1e-4


def test_balancing_canonicalizer_equalizes_circuit_energies():
    graph, params, _ = _case(seed=5)

    canonicalized, _, _ = ScaleCanonicalizer().canonicalize(
        graph, params, activation="gelu"
    )

    residual = attention_balancing_scales(graph, canonicalized)
    for group_id, factors in residual.items():
        np.testing.assert_allclose(
            np.asarray(factors),
            np.ones_like(np.asarray(factors)),
            atol=1e-5,
            err_msg=group_id,
        )


def test_balancing_canonicalizer_leaves_non_circuit_groups_untouched():
    """Stream/head/FFN tensors without intra-head bindings must be unchanged."""

    graph, params, _ = _case(seed=6)
    canonicalized, _, _ = ScaleCanonicalizer().canonicalize(
        graph, params, activation="gelu"
    )

    embedding = "TokenEmbedding_0/Embedding/embedding"
    path = graph.tensors[embedding].path
    node_before, node_after = params, canonicalized
    for key in path:
        node_before = node_before[key]
        node_after = node_after[key]
    np.testing.assert_array_equal(np.asarray(node_before), np.asarray(node_after))


def test_attention_plan_rejects_dense_only_post_passes():
    graph, params, _ = _case(seed=7)
    with pytest.raises(ValueError, match="degenerate_channels"):
        ScaleCanonicalizer().canonicalize(
            graph, params, activation="gelu", degenerate_channels="zero_outgoing"
        )


def test_scaled_orbit_needs_balancing_before_matching():
    """The measurable gain: permutation-only matching cannot collapse circuit
    scales; balancing canonicalization first recovers the exact orbit."""

    case = make_layernorm_mha_transformer_scaled_orbit_case(seed=0)
    schedule = [{"solver": "lap", "max_sweeps": 25, "tolerance": 0.0}]

    without = run_alignment_benchmark(case, schedule=schedule, canonicalize=False)
    with_canonicalization = run_alignment_benchmark(
        case, schedule=schedule, canonicalize=True
    )

    # Both treatments are exact symmetries: the function never drifts.
    assert without.metrics.function_drift_max < 1e-4
    assert with_canonicalization.metrics.function_drift_max < 1e-4
    # Matching alone leaves the diagonal-scale offset in weight space.
    assert (
        without.metrics.distance_after
        > 10 * with_canonicalization.metrics.distance_after
    )
    # Balancing first collapses the orbit and recovers the permutations.
    assert with_canonicalization.metrics.distance_after < 1e-4
    assert with_canonicalization.metrics.recovered_transform_error == 0.0
    assert with_canonicalization.metrics.transform_validity_error == 0.0


def test_dense_chain_rejects_non_homogeneous_activation():
    from align.architectures import MLPRecipe

    rng = np.random.default_rng(0)
    fcn = {
        f"Dense_{idx}": {
            "kernel": jnp.asarray(rng.standard_normal((3, 3)), dtype=jnp.float32),
            "bias": jnp.asarray(rng.standard_normal((3,)), dtype=jnp.float32),
        }
        for idx in range(3)
    }
    params = {"params": {"fcn": fcn}}
    graph = MLPRecipe(parameter_root="params.fcn").build_graph(params)
    with pytest.raises(ValueError, match="homogeneous"):
        ScaleCanonicalizer().canonicalize(graph, params, activation="gelu")
