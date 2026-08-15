"""Tests for conv-stack scale symmetries and their canonicalization.

Covers the exact post-normalization affine symmetry of FRN/TLU stacks (norm
affine parameters divided, consumers multiplied, conv kernels / eps / running
stats untouched via ``scale_power == 0``), the bare-conv ReLU chain symmetry,
and both canonical forms: ``balanced`` (default; the minimum-norm orbit
representative, which also handles bare-conv residual cycles via the convex
fixed point) and ``unit_norm`` (joint producer energy 1, which rejects
bare-conv residual cycles because no one-shot canonical scale exists).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures import ResidualConvNetRecipe
from align.benchmarks.synthetic import (
    frn_residual_conv_apply,
    make_frn_residual_conv_orbit_case,
    make_frn_residual_conv_params,
    make_residual_conv_orbit_case,
    residual_conv_apply,
    run_alignment_benchmark,
)
from align.canonicalization import ScaleCanonicalizer, ScaleState
from align.canonicalization.convnet import (
    convnet_balanced_scales,
    convnet_producer_scales,
)
from align.symmetry.tensor_ops import _descend, binding_axis_interval

_MODULE_GRAPH = {
    "nodes": [
        {"id": "input", "kind": "input"},
        {"id": "core/Conv_0", "kind": "conv", "normalizer": "core/FRN_0"},
        {"id": "core/Conv_1", "kind": "conv", "normalizer": "core/FRN_1"},
        {"id": "core/Conv_2", "kind": "conv", "normalizer": "core/FRN_2"},
        {"id": "residual_add", "kind": "add"},
        {"id": "core/Dense_0", "kind": "dense"},
    ],
    "edges": [
        {"source": "input", "target": "core/Conv_0"},
        {"source": "core/Conv_0", "target": "core/Conv_1"},
        {"source": "core/Conv_1", "target": "core/Conv_2"},
        {"source": "core/Conv_0", "target": "residual_add"},
        {"source": "core/Conv_2", "target": "residual_add"},
        {"source": "residual_add", "target": "core/Dense_0"},
    ],
}


def _frn_case(seed: int = 0):
    params = make_frn_residual_conv_params(jax.random.PRNGKey(seed))
    recipe = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=_MODULE_GRAPH
    )
    graph = recipe.build_graph(params)
    inputs = jax.random.normal(
        jax.random.PRNGKey(seed + 50), (5, 3, 3, 2), dtype=jnp.float32
    )
    return graph, params, inputs


def test_conv_plan_requires_explicit_activation():
    graph, params, _ = _frn_case(seed=31)
    with pytest.raises(ValueError, match="requires an explicit.*activation"):
        ScaleCanonicalizer().canonicalize(graph, params)


def _positive_scales(graph, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        gid: np.exp(rng.uniform(-1.0, 1.0, size=group.size)).astype(np.float32)
        for gid, group in graph.groups.items()
    }


def _tree_max_abs_diff(left, right) -> float:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return max(
        float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def test_apply_scales_preserves_frn_function_exactly():
    graph, params, inputs = _frn_case(seed=0)
    scales = _positive_scales(graph, seed=1)

    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))

    np.testing.assert_allclose(
        np.asarray(frn_residual_conv_apply(scaled, inputs)),
        np.asarray(frn_residual_conv_apply(params, inputs)),
        atol=1e-4,
    )
    assert _tree_max_abs_diff(scaled, params) > 1e-2


def test_apply_scales_exempts_pre_norm_tensors():
    """Producer axes of conv kernels/biases and eps carry scale_power 0: the
    action only multiplies conv *input* axes (consumers) and divides the FRN
    affine parameters."""

    graph, params, _ = _frn_case(seed=2)
    scales = _positive_scales(graph, seed=3)
    tied_group, branch_group = graph.metadata["group_order"]

    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))

    core_before = params["core"]
    core_after = scaled["core"]
    # Conv_0 consumes the network input: fully untouched.
    np.testing.assert_array_equal(
        np.asarray(core_before["Conv_0"]["kernel"]),
        np.asarray(core_after["Conv_0"]["kernel"]),
    )
    # Conv biases and FRN eps are pre-norm producers: untouched.
    for name in ("Conv_0", "Conv_1", "Conv_2"):
        np.testing.assert_array_equal(
            np.asarray(core_before[name]["bias"]),
            np.asarray(core_after[name]["bias"]),
            err_msg=name,
        )
    for name in ("FRN_0", "FRN_1", "FRN_2"):
        np.testing.assert_array_equal(
            np.asarray(core_before[name]["eps"]),
            np.asarray(core_after[name]["eps"]),
            err_msg=name,
        )
    # Downstream conv kernels are consumers: multiplied along the input axis
    # only (no division along the output axis).
    np.testing.assert_allclose(
        np.asarray(core_after["Conv_1"]["kernel"]),
        np.asarray(core_before["Conv_1"]["kernel"])
        * scales[tied_group][None, None, :, None],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(core_after["Conv_2"]["kernel"]),
        np.asarray(core_before["Conv_2"]["kernel"])
        * scales[branch_group][None, None, :, None],
        rtol=1e-6,
    )
    # The FRN affine parameters carry the symmetry: divided by the scale.
    np.testing.assert_allclose(
        np.asarray(core_after["FRN_1"]["gamma"]),
        np.asarray(core_before["FRN_1"]["gamma"]) / scales[branch_group],
        rtol=1e-6,
    )


def _balance_energies(graph, params):
    """Per-group dividing/multiplying energies over power-1 bindings."""

    energies = {}
    for group_id, group in graph.groups.items():
        div = np.zeros(int(group.size))
        mult = np.zeros(int(group.size))
        for binding in graph.bindings_for_group(group_id):
            if binding.scale_power == 0.0:
                continue
            spec = graph.tensors[binding.tensor_id]
            axis, start, stop = binding_axis_interval(spec.shape, binding)
            tensor = np.asarray(_descend(params, spec.path), dtype=np.float64)
            indexer = tuple(
                slice(start, stop) if dim == axis else slice(None)
                for dim in range(tensor.ndim)
            )
            moved = np.moveaxis(tensor[indexer], axis, -1)
            energy = np.sum(moved.reshape(-1, moved.shape[-1]) ** 2, axis=0)
            if binding.role == "out":
                div += energy
            else:
                mult += energy
        energies[group_id] = (div, mult)
    return energies


def _tree_sq_norm(graph, params) -> float:
    """Total squared norm of the graph's scale-carrying tensors."""

    total = 0.0
    for tensor_id, spec in graph.tensors.items():
        if not any(
            binding.scale_power != 0.0
            for binding in graph.bindings_for_tensor(tensor_id)
        ):
            continue
        tensor = np.asarray(_descend(params, spec.path), dtype=np.float64)
        total += float(np.sum(tensor**2))
    return total


def test_frn_canonicalizer_preserves_function_and_is_idempotent():
    graph, params, inputs = _frn_case(seed=4)

    canonicalized, factors, aux = ScaleCanonicalizer().canonicalize(
        graph, params, activation="tlu"
    )
    twice, _, _ = ScaleCanonicalizer().canonicalize(
        graph, canonicalized, activation="tlu"
    )

    assert aux["plan"] == "conv_balanced"
    assert aux["strategy"] == "balanced"
    assert len(factors) == len(graph.group_order)
    np.testing.assert_allclose(
        np.asarray(frn_residual_conv_apply(canonicalized, inputs)),
        np.asarray(frn_residual_conv_apply(params, inputs)),
        atol=1e-4,
    )
    assert _tree_max_abs_diff(canonicalized, twice) < 1e-5


def test_frn_balanced_equalizes_energies_and_minimizes_norm():
    """The default form balances each channel's dividing/multiplying energies
    and is the minimum-norm point of the scale orbit."""

    graph, params, _ = _frn_case(seed=4)
    canonicalized, _, _ = ScaleCanonicalizer().canonicalize(
        graph, params, activation="tlu"
    )

    for group_id, (div, mult) in _balance_energies(graph, canonicalized).items():
        np.testing.assert_allclose(div, mult, rtol=1e-4, err_msg=group_id)

    canonical_norm = _tree_sq_norm(graph, canonicalized)
    assert canonical_norm < _tree_sq_norm(graph, params)
    rng = np.random.default_rng(3)
    for _ in range(5):
        scales = {
            gid: np.exp(rng.uniform(-0.5, 0.5, size=group.size)).astype(np.float32)
            for gid, group in graph.groups.items()
        }
        jittered = graph.apply_scales(
            canonicalized, ScaleState.from_scales(graph, scales)
        )
        assert _tree_sq_norm(graph, jittered) > canonical_norm


def test_frn_canonicalizer_collapses_affine_scale_orbit():
    graph, params, _ = _frn_case(seed=5)
    scales = _positive_scales(graph, seed=6)
    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))

    canonical_a, _, _ = ScaleCanonicalizer().canonicalize(
        graph, params, activation="tlu"
    )
    canonical_b, _, _ = ScaleCanonicalizer().canonicalize(
        graph, scaled, activation="tlu"
    )

    assert _tree_max_abs_diff(canonical_a, canonical_b) < 1e-4


def test_frn_norm_attached_balancing_is_one_shot():
    """Norm-attached groups decouple: the first sweep lands on the fixed
    point for any residual topology (here a residual cycle), so a run capped
    at two sweeps (one working, one confirming, up to the epsilon stabilizer)
    already matches the fully converged scales."""

    graph, params, _ = _frn_case(seed=8)

    converged = convnet_balanced_scales(graph, params)
    one_shot = convnet_balanced_scales(graph, params, max_iter=2, tol=1e-7)
    for group_id in graph.group_order:
        np.testing.assert_allclose(
            np.asarray(one_shot[group_id]),
            np.asarray(converged[group_id]),
            rtol=1e-6,
            err_msg=group_id,
        )


def test_frn_canonicalizer_unit_joint_energy_per_tied_channel():
    """Residual-tied channels canonicalize the *joint* FRN affine energy to 1."""

    graph, params, _ = _frn_case(seed=7)
    canonicalized, _, _ = ScaleCanonicalizer().canonicalize(
        graph, params, activation="tlu", strategy="unit_norm"
    )

    core = canonicalized["core"]
    tied_energy = sum(
        np.asarray(core[name][param]) ** 2
        for name in ("FRN_0", "FRN_2")
        for param in ("gamma", "beta", "tau")
    )
    np.testing.assert_allclose(tied_energy, np.ones_like(tied_energy), atol=1e-5)
    branch_energy = sum(
        np.asarray(core["FRN_1"][param]) ** 2 for param in ("gamma", "beta", "tau")
    )
    np.testing.assert_allclose(branch_energy, np.ones_like(branch_energy), atol=1e-5)


def test_bare_conv_residual_cycle_rejected_by_unit_norm_only():
    case = make_residual_conv_orbit_case(seed=0)
    with pytest.raises(ValueError, match="residual cycle of bare convs"):
        ScaleCanonicalizer().canonicalize(
            case.graph, case.reference, strategy="unit_norm", activation="relu"
        )


def test_bare_conv_residual_cycle_balances_exactly():
    """The coupled fixed point handles the cycle the one-shot fold rejects:
    exact function preservation, per-channel balance, idempotence, and orbit
    collapse."""

    case = make_residual_conv_orbit_case(seed=0)
    graph, params, inputs = case.graph, case.reference, case.inputs

    canonicalized, _, aux = ScaleCanonicalizer().canonicalize(
        graph, params, activation="relu"
    )
    assert aux["plan"] == "conv_balanced"
    np.testing.assert_allclose(
        np.asarray(residual_conv_apply(canonicalized, inputs)),
        np.asarray(residual_conv_apply(params, inputs)),
        atol=1e-4,
    )
    for group_id, (div, mult) in _balance_energies(graph, canonicalized).items():
        np.testing.assert_allclose(div, mult, rtol=1e-4, err_msg=group_id)

    twice, _, _ = ScaleCanonicalizer().canonicalize(
        graph, canonicalized, activation="relu"
    )
    assert _tree_max_abs_diff(canonicalized, twice) < 1e-5

    scales = _positive_scales(graph, seed=12)
    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))
    canonical_b, _, _ = ScaleCanonicalizer().canonicalize(
        graph, scaled, activation="relu"
    )
    assert _tree_max_abs_diff(canonicalized, canonical_b) < 1e-4


def test_bare_conv_self_loop_group_balances_exactly():
    """A single-conv residual branch (``y = relu(h + conv(h))``) couples
    channels within the tied group; the damped update plus the common-shift
    line step still reach the balanced fixed point."""

    rng = np.random.default_rng(15)

    def _conv(cin, cout):
        return {
            "kernel": jnp.asarray(
                rng.standard_normal((1, 1, cin, cout)), dtype=jnp.float32
            ),
            "bias": jnp.asarray(0.1 * rng.standard_normal((cout,)), jnp.float32),
        }

    params = {
        "core": {
            "Conv_0": _conv(2, 4),
            "Conv_1": _conv(4, 4),
            "Dense_0": {
                "kernel": jnp.asarray(rng.standard_normal((4, 3)), jnp.float32),
                "bias": jnp.asarray(0.1 * rng.standard_normal((3,)), jnp.float32),
            },
        }
    }
    topology = {
        "nodes": [
            {"id": "input", "kind": "input"},
            {"id": "core/Conv_0", "kind": "conv"},
            {"id": "core/Conv_1", "kind": "conv"},
            {"id": "add", "kind": "add"},
            {"id": "core/Dense_0", "kind": "dense"},
        ],
        "edges": [
            {"source": "input", "target": "core/Conv_0"},
            {"source": "core/Conv_0", "target": "core/Conv_1"},
            {"source": "core/Conv_0", "target": "add"},
            {"source": "core/Conv_1", "target": "add"},
            {"source": "add", "target": "core/Dense_0"},
        ],
    }
    graph = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=topology
    ).build_graph(params)
    assert len(graph.group_order) == 1

    def _apply(tree, x):
        core = tree["core"]
        h = jnp.einsum(
            "bhwc,co->bhwo", x, jnp.asarray(core["Conv_0"]["kernel"])[0, 0]
        ) + jnp.asarray(core["Conv_0"]["bias"])
        h = jnp.maximum(0.0, h)
        branch = jnp.einsum(
            "bhwc,co->bhwo", h, jnp.asarray(core["Conv_1"]["kernel"])[0, 0]
        ) + jnp.asarray(core["Conv_1"]["bias"])
        h = jnp.maximum(0.0, h + branch)
        pooled = jnp.mean(h, axis=(1, 2))
        return pooled @ jnp.asarray(core["Dense_0"]["kernel"]) + jnp.asarray(
            core["Dense_0"]["bias"]
        )

    inputs = jnp.asarray(rng.standard_normal((5, 3, 3, 2)), jnp.float32)
    canonicalized, _, aux = ScaleCanonicalizer().canonicalize(
        graph, params, activation="relu"
    )
    assert aux["plan"] == "conv_balanced"
    np.testing.assert_allclose(
        np.asarray(_apply(canonicalized, inputs)),
        np.asarray(_apply(params, inputs)),
        atol=1e-4,
    )
    for group_id, (div, mult) in _balance_energies(graph, canonicalized).items():
        np.testing.assert_allclose(div, mult, rtol=1e-4, err_msg=group_id)

    scales = _positive_scales(graph, seed=16)
    scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))
    canonical_b, _, _ = ScaleCanonicalizer().canonicalize(
        graph, scaled, activation="relu"
    )
    assert _tree_max_abs_diff(canonicalized, canonical_b) < 1e-4


def test_bare_conv_chain_canonicalizes_like_dense():
    """Without residual ties, bare convs fold upstream scales sequentially."""

    rng = np.random.default_rng(8)

    def _conv(cin, cout):
        return {
            "kernel": jnp.asarray(
                rng.standard_normal((1, 1, cin, cout)), dtype=jnp.float32
            ),
            "bias": jnp.asarray(0.1 * rng.standard_normal((cout,)), jnp.float32),
        }

    params = {
        "core": {
            "Conv_0": _conv(2, 4),
            "Conv_1": _conv(4, 3),
            "Conv_2": _conv(3, 4),
            "Dense_0": {
                "kernel": jnp.asarray(rng.standard_normal((4, 2)), jnp.float32),
                "bias": jnp.asarray(0.1 * rng.standard_normal((2,)), jnp.float32),
            },
        }
    }
    graph = ResidualConvNetRecipe(
        parameter_root="core", linear_residual_free=True
    ).build_graph(params)

    def _apply(tree, x):
        core = tree["core"]
        h = x
        for name in ("Conv_0", "Conv_1", "Conv_2"):
            kernel = jnp.asarray(core[name]["kernel"])
            h = jnp.einsum("bhwc,co->bhwo", h, kernel[0, 0]) + jnp.asarray(
                core[name]["bias"]
            )
            h = jnp.maximum(0.0, h)
        pooled = jnp.mean(h, axis=(1, 2))
        return pooled @ jnp.asarray(core["Dense_0"]["kernel"]) + jnp.asarray(
            core["Dense_0"]["bias"]
        )

    inputs = jnp.asarray(rng.standard_normal((5, 3, 3, 2)), jnp.float32)
    canonicalized, _, aux = ScaleCanonicalizer().canonicalize(
        graph, params, strategy="unit_norm", activation="relu"
    )
    twice, _, _ = ScaleCanonicalizer().canonicalize(
        graph, canonicalized, strategy="unit_norm", activation="relu"
    )

    assert aux["plan"] == "conv_producer_energy"
    np.testing.assert_allclose(
        np.asarray(_apply(canonicalized, inputs)),
        np.asarray(_apply(params, inputs)),
        atol=1e-4,
    )
    assert _tree_max_abs_diff(canonicalized, twice) < 1e-5

    # Unit incoming energy per channel (kernel column + bias), like dense.
    scales = convnet_producer_scales(graph, canonicalized)
    for group_id, vector in scales.items():
        np.testing.assert_allclose(
            np.asarray(vector),
            np.ones_like(np.asarray(vector)),
            atol=1e-5,
            err_msg=group_id,
        )

    # The balanced default also handles the chain (coupled fixed point).
    balanced, _, balanced_aux = ScaleCanonicalizer().canonicalize(
        graph, params, activation="relu"
    )
    assert balanced_aux["plan"] == "conv_balanced"
    np.testing.assert_allclose(
        np.asarray(_apply(balanced, inputs)),
        np.asarray(_apply(params, inputs)),
        atol=1e-4,
    )
    for group_id, (div, mult) in _balance_energies(graph, balanced).items():
        np.testing.assert_allclose(div, mult, rtol=1e-4, err_msg=group_id)


def test_conv_plan_rejects_dense_only_options():
    graph, params, _ = _frn_case(seed=9)
    with pytest.raises(ValueError, match="homogeneous"):
        ScaleCanonicalizer().canonicalize(graph, params, activation="gelu")
    with pytest.raises(ValueError, match="degenerate_channels"):
        ScaleCanonicalizer().canonicalize(
            graph, params, activation="tlu", degenerate_channels="zero_outgoing"
        )
    with pytest.raises(ValueError, match="include_bias_in_norm"):
        ScaleCanonicalizer().canonicalize(
            graph, params, activation="tlu", include_bias_in_norm=False
        )


def test_frn_scaled_orbit_needs_canonicalization_before_matching():
    """The measurable gain: permutation-only matching cannot collapse affine
    scales; balanced canonicalization first recovers the exact orbit."""

    case = make_frn_residual_conv_orbit_case(seed=0)
    # The residual cycle couples the two channel groups, so LAP coordinate
    # descent can stall in a local optimum on this tiny case (seed 0 does);
    # the mixed schedule's Sinkhorn step escapes it.
    schedule = [
        {"solver": "lap", "max_sweeps": 25, "tolerance": 0.0},
        {
            "solver": "sinkhorn",
            "max_steps": 150,
            "learning_rate": 0.1,
            "tau": 0.3,
            "sinkhorn_iterations": 20,
            "init_scale": 0.01,
            "tolerance": 0.0,
        },
        {"solver": "lap", "max_sweeps": 5, "tolerance": 0.0},
    ]

    without = run_alignment_benchmark(case, schedule=schedule, canonicalize=False)
    with_canonicalization = run_alignment_benchmark(
        case, schedule=schedule, canonicalize=True
    )

    assert without.metrics.function_drift_max < 1e-4
    assert with_canonicalization.metrics.function_drift_max < 1e-4
    assert (
        without.metrics.distance_after
        > 10 * with_canonicalization.metrics.distance_after
    )
    assert with_canonicalization.metrics.distance_after < 1e-4
    assert with_canonicalization.metrics.recovered_transform_error == 0.0
    assert with_canonicalization.metrics.transform_validity_error == 0.0
    assert with_canonicalization.metrics.residual_constraint_violations == 0
