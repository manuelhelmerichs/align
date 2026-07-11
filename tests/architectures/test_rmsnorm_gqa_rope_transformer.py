"""RMSNorm + RoPE + GQA transformer tests.

The symmetry-derivation tests are numerical proofs of the claims in the
wiki/Architecture-RMSNorm-GQA-RoPE-Transformer.md: each derived symmetry is
applied to a generic random network and must preserve the function exactly
(up to float32 noise), and each *removed* symmetry (qk permutations under
RoPE, cross-group query-head swaps under GQA) must demonstrably change the
function on generic weights.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures import RMSNormGQARoPETransformerRecipe
from align.canonicalization import ScaleCanonicalizer, ScaleState
from align.canonicalization.rmsnorm_gqa_rope import (
    apply_rms_gamma_scales,
    gqa_rope_circuit_constraints,
    gqa_vo_balancing_scales,
    qk_pair_scales,
    rmsnorm_scale_constraints,
)
from align.matching import TransformState, match_sample
from align.matching.objectives import UnsupportedGroupLinearization, get_objective
from align.symmetry.tensor_ops import _descend
from benchmarks import (
    make_rmsnorm_gqa_rope_transformer_orbit_case,
    make_rmsnorm_gqa_rope_transformer_orthogonal_orbit_case,
    make_rmsnorm_gqa_rope_transformer_params,
    make_rmsnorm_gqa_rope_transformer_rotated_orbit_case,
    make_rmsnorm_gqa_rope_transformer_scaled_orbit_case,
    make_synthetic_rmsnorm_gqa_rope_transformer_posterior_case,
    rmsnorm_gqa_rope_transformer_apply,
    run_alignment_benchmark,
)
from benchmarks.posterior import run_posterior_benchmark
from benchmarks.synthetic import default_schedule_grid

DRIFT_TOL = 1e-4
BREAK_TOL = 1e-3


def _case_params(seed: int = 0):
    params = make_rmsnorm_gqa_rope_transformer_params(key=jax.random.PRNGKey(seed))
    recipe = RMSNormGQARoPETransformerRecipe()
    return params, recipe.build_graph(params)


def _tokens(seed: int = 1) -> jax.Array:
    return jax.random.randint(jax.random.PRNGKey(seed), (4, 5), 0, 7, jnp.int32)


def _drift(params_a, params_b, tokens) -> float:
    out_a = np.asarray(rmsnorm_gqa_rope_transformer_apply(params_a, tokens))
    out_b = np.asarray(rmsnorm_gqa_rope_transformer_apply(params_b, tokens))
    return float(np.max(np.abs(out_a - out_b)))


def _perm_matrix(indices) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    mat = np.zeros((indices.shape[0], indices.shape[0]), dtype=np.float64)
    mat[np.arange(indices.shape[0]), indices] = 1.0
    return mat


def _random_state(graph, seed: int) -> TransformState:
    """Random permutations on permutation-capable groups (identity on
    rotation-pair groups, whose class contains no permutations)."""

    rng = np.random.default_rng(seed)
    perms = {
        group_id: _perm_matrix(rng.permutation(group.size))
        for group_id, group in graph.groups.items()
        if group.transform_family != "rotation_pairs"
    }
    return TransformState.from_transforms(graph, perms)


def _edit_leaf(params, path, value):
    import copy

    edited = copy.deepcopy(params)
    node = edited
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return edited


class TestSymmetryDerivations:
    """Function-invariance proofs for the derived symmetry model."""

    def test_wreath_permutations_preserve_function(self):
        params, graph = _case_params(seed=0)
        tokens = _tokens()
        for seed in (3, 7):
            permuted = graph.apply_transforms(params, _random_state(graph, seed))
            assert _drift(params, permuted, tokens) < DRIFT_TOL

    def test_rms_gamma_scales_preserve_function(self):
        params, graph = _case_params(seed=0)
        rng = np.random.default_rng(5)
        scales = {
            constraint.scale: np.exp(
                rng.uniform(
                    -1.5,
                    1.5,
                    size=graph.tensors[constraint.scale].shape[0],
                )
            ).astype(np.float32)
            for constraint in rmsnorm_scale_constraints(graph)
        }
        scaled = apply_rms_gamma_scales(graph, params, scales)
        assert _drift(params, scaled, _tokens()) < DRIFT_TOL

    def test_qk_pair_scales_preserve_function(self):
        params, graph = _case_params(seed=0)
        rng = np.random.default_rng(6)
        scales = {}
        for constraint in gqa_rope_circuit_constraints(graph):
            half = constraint.head_dim // 2
            for group_id in constraint.qk_groups:
                pair = np.exp(rng.uniform(-1.5, 1.5, size=half))
                scales[group_id] = np.concatenate([pair, pair]).astype(np.float32)
        scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))
        assert _drift(params, scaled, _tokens()) < DRIFT_TOL

    def test_qk_non_pair_scales_break_function(self):
        """The qk scale freedom is pair-tied: unequal within-pair scales are
        not a symmetry under RoPE (the scaled map no longer commutes with the
        position rotations)."""

        params, graph = _case_params(seed=0)
        constraint = gqa_rope_circuit_constraints(graph)[0]
        head_dim = constraint.head_dim
        group_id = constraint.qk_groups[0]
        factors = np.ones(head_dim, dtype=np.float32)
        factors[0] = 2.0  # scale one pair member without its partner
        scaled = graph.apply_scales(
            params, ScaleState.from_scales(graph, {group_id: factors})
        )
        assert _drift(params, scaled, _tokens()) > BREAK_TOL

    def test_vo_scales_preserve_function(self):
        params, graph = _case_params(seed=0)
        rng = np.random.default_rng(7)
        scales = {
            group_id: np.exp(
                rng.uniform(-1.5, 1.5, size=graph.groups[group_id].size)
            ).astype(np.float32)
            for constraint in gqa_rope_circuit_constraints(graph)
            for group_id in constraint.vo_groups
        }
        scaled = graph.apply_scales(params, ScaleState.from_scales(graph, scales))
        assert _drift(params, scaled, _tokens()) < DRIFT_TOL

    def test_orthogonal_stream_preserves_function_after_gamma_folding(self):
        """RMSNorm streams admit the full orthogonal group, not just perms.

        Fold every gamma into its consumers (``gamma' = 1``), then apply one
        random orthogonal matrix along every stream-bound axis except the
        folded gammas. This is the O(d) stream symmetry derived in
        the RMSNorm/GQA/RoPE wiki page; it has no LayerNorm analogue (mean subtraction breaks
        it).
        """

        params, graph = _case_params(seed=0)
        tokens = _tokens()
        gammas = {
            constraint.scale: np.asarray(
                _descend(params, graph.tensors[constraint.scale].path)
            )
            for constraint in rmsnorm_scale_constraints(graph)
        }
        folded = apply_rms_gamma_scales(graph, params, gammas)
        assert _drift(params, folded, tokens) < DRIFT_TOL

        d_model = graph.groups["stream"].size
        rng = np.random.default_rng(11)
        q_matrix, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
        gamma_ids = set(gammas)

        rotated = folded
        for binding in graph.bindings_for_group("stream"):
            if binding.tensor_id in gamma_ids:
                continue
            path = graph.tensors[binding.tensor_id].path
            tensor = np.asarray(_descend(rotated, path))
            axis = binding.axis % tensor.ndim
            moved = np.moveaxis(tensor, axis, 0)
            transformed = (q_matrix @ moved.reshape(d_model, -1)).reshape(moved.shape)
            rotated = _edit_leaf(
                rotated, path, np.moveaxis(transformed, 0, axis).astype(tensor.dtype)
            )
        assert _drift(folded, rotated, tokens) < DRIFT_TOL

    @staticmethod
    def _rotate_pairs(kernel: np.ndarray, cos: np.ndarray, sin: np.ndarray):
        half = kernel.shape[-1] // 2
        first, second = kernel[..., :half], kernel[..., half:]
        return np.concatenate(
            [first * cos - second * sin, second * cos + first * sin], axis=-1
        )

    def test_rope_pair_scaled_rotations_preserve_function(self):
        """The RoPE qk symmetry is per-frequency-pair scaled rotations.

        Rotating the query and key kernels' half-split pair columns by the
        *same* per-pair angle (and scaling reciprocally) commutes with the
        position-dependent rotary rotations, so attention is unchanged.
        """

        params, graph = _case_params(seed=0)
        tokens = _tokens()
        rng = np.random.default_rng(13)
        edited = params
        for constraint in gqa_rope_circuit_constraints(graph):
            half = constraint.head_dim // 2
            num_groups = constraint.num_kv_groups
            angles = rng.uniform(-np.pi, np.pi, size=(num_groups, half))
            alphas = np.exp(rng.uniform(-1.0, 1.0, size=(num_groups, half)))
            cos, sin = np.cos(angles), np.sin(angles)

            query_path = graph.tensors[constraint.query].path
            key_path = graph.tensors[constraint.key].path
            query = np.asarray(_descend(edited, query_path))
            key = np.asarray(_descend(edited, key_path))
            alpha_q = np.concatenate([alphas, alphas], axis=-1)[:, None, :]
            alpha_k = np.concatenate([1.0 / alphas, 1.0 / alphas], axis=-1)
            query = self._rotate_pairs(query, cos[:, None], sin[:, None]) * alpha_q
            key = self._rotate_pairs(key, cos, sin) * alpha_k
            edited = _edit_leaf(edited, query_path, query.astype(np.float32))
            edited = _edit_leaf(edited, key_path, key.astype(np.float32))
        assert _drift(params, edited, tokens) < DRIFT_TOL

    def _swap_qk_dims(self, params, graph, dim_a: int, dim_b: int):
        """Swap two qk head dimensions of the first attention module."""

        constraint = gqa_rope_circuit_constraints(graph)[0]
        edited = params
        for role in ("query", "key"):
            path = graph.tensors[getattr(constraint, role)].path
            tensor = np.asarray(_descend(params, path)).copy()
            tensor[..., [dim_a, dim_b]] = tensor[..., [dim_b, dim_a]]
            edited = _edit_leaf(edited, path, tensor)
        return edited

    def test_rope_breaks_within_pair_swap(self):
        """Within-pair swaps are reflections, not rotations: not a symmetry."""

        params, graph = _case_params(seed=0)
        half = gqa_rope_circuit_constraints(graph)[0].head_dim // 2
        swapped = self._swap_qk_dims(params, graph, 0, half)
        assert _drift(params, swapped, _tokens()) > BREAK_TOL

    def test_rope_breaks_cross_pair_permutation(self):
        """Distinct rotary frequencies pin each pair: pair swaps break scores."""

        params, graph = _case_params(seed=0)
        half = gqa_rope_circuit_constraints(graph)[0].head_dim // 2
        assert half >= 2
        swapped = self._swap_qk_dims(params, graph, 0, 1)
        swapped = self._swap_qk_dims(swapped, graph, half, half + 1)
        assert _drift(params, swapped, _tokens()) > BREAK_TOL

    def test_gqa_breaks_cross_group_query_head_swap(self):
        """Query heads only permute within their kv group (the GQA quotient)."""

        params, graph = _case_params(seed=0)
        constraint = gqa_rope_circuit_constraints(graph)[0]
        query_path = graph.tensors[constraint.query].path
        out_path = graph.tensors[constraint.out].path
        query = np.asarray(_descend(params, query_path)).copy()
        out = np.asarray(_descend(params, out_path)).copy()
        query[:, [0, 1], 0] = query[:, [1, 0], 0]
        out[[0, 1], 0] = out[[1, 0], 0]
        edited = _edit_leaf(params, query_path, query)
        edited = _edit_leaf(edited, out_path, out)
        assert _drift(params, edited, _tokens()) > BREAK_TOL


class TestRecipeStructure:
    def test_groups_blocks_and_constraints(self):
        params, graph = _case_params(seed=0)
        assert graph.groups["stream"].size == 8
        for block in (0, 1):
            assert graph.groups[f"Block_{block}/attention/kv"].size == 2
            for slot in (0, 1):
                assert graph.groups[f"Block_{block}/attention/qh{slot}"].size == 2
                assert graph.groups[f"Block_{block}/attention/vo{slot}"].size == 4
            assert graph.groups[f"Block_{block}/ffn/h0"].size == 12
        kinds = {component.kind for component in graph.components.values()}
        assert kinds == {"residual_stream", "gqa_rope", "dense_chain"}
        assert len(gqa_rope_circuit_constraints(graph)) == 2
        # Two per block (attention norm, FFN norm) plus the final norm.
        assert len(rmsnorm_scale_constraints(graph)) == 5

    def test_group_transform_families(self):
        params, graph = _case_params(seed=0)
        assert graph.groups["stream"].transform_family == "signed_permutation"
        for block in (0, 1):
            assert graph.groups[f"Block_{block}/attention/kv"].transform_family == (
                "permutation"
            )
            for slot in (0, 1):
                prefix = f"Block_{block}/attention"
                assert (
                    graph.groups[f"{prefix}/qh{slot}"].transform_family == "permutation"
                )
                # RoPE: the qk circuit admits only per-pair rotations.
                assert graph.groups[f"{prefix}/qk{slot}"].transform_family == (
                    "rotation_pairs"
                )
                assert graph.groups[f"{prefix}/vo{slot}"].transform_family == (
                    "signed_permutation"
                )
            assert (
                graph.groups[f"Block_{block}/ffn/h0"].transform_family == "permutation"
            )
        # RMSNorm scales permute with the stream but are sign/orthogonal-exempt.
        gamma_ids = {
            constraint.scale for constraint in rmsnorm_scale_constraints(graph)
        }
        for binding in graph.bindings_for_group("stream"):
            expected_scope = (
                "permute_only" if binding.tensor_id in gamma_ids else "linear"
            )
            assert binding.transform_scope == expected_scope

    def test_orthogonal_stream_option(self):
        params, _ = _case_params(seed=0)
        recipe = RMSNormGQARoPETransformerRecipe(stream_transform_family="orthogonal")
        graph = recipe.build_graph(params)
        assert graph.groups["stream"].transform_family == "orthogonal"
        with pytest.raises(ValueError, match="stream_transform_family"):
            RMSNormGQARoPETransformerRecipe(stream_transform_family="permutation")

    def test_layernorm_tree_directed_to_layernorm_mha_recipe(self):
        params, _ = _case_params(seed=0)
        params["Block_0"]["RMSNorm_0"]["bias"] = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="'layernorm_mha_transformer' recipe"):
            RMSNormGQARoPETransformerRecipe().build_graph(params)

    def test_attention_bias_rejected(self):
        params, _ = _case_params(seed=0)
        params["Block_0"]["GQAttention_0"]["query"]["bias"] = np.zeros(
            (2, 2, 4), dtype=np.float32
        )
        with pytest.raises(ValueError, match="bias-free"):
            RMSNormGQARoPETransformerRecipe().build_graph(params)

    def test_flat_head_layout_rejected(self):
        params, _ = _case_params(seed=0)
        module = params["Block_0"]["GQAttention_0"]
        kernel = np.asarray(module["query"]["kernel"])
        module["query"]["kernel"] = kernel.reshape(8, 4, 4)
        with pytest.raises(ValueError, match="kv_groups"):
            RMSNormGQARoPETransformerRecipe().build_graph(params)

    def test_kv_group_rejects_generic_lap_linearization(self):
        params, graph = _case_params(seed=0)
        objective = get_objective("euclidean")
        data = graph.materialize(params, backend="numpy")
        state = TransformState.identity(graph, backend="numpy")
        with pytest.raises(UnsupportedGroupLinearization, match="kv-group"):
            objective.linearize_group(graph, data, data, state, "Block_0/attention/kv")


class TestCanonicalization:
    def test_plan_dispatch_idempotence_and_function_preservation(self):
        params, graph = _case_params(seed=0)
        tokens = _tokens()
        canonicalizer = ScaleCanonicalizer()
        once, _, aux = canonicalizer.canonicalize(graph, params)
        assert aux["plan"] == "rmsnorm_gqa_rope"
        assert _drift(params, once, tokens) < DRIFT_TOL
        twice, _, _ = canonicalizer.canonicalize(graph, once)
        for path in (
            ("RMSNorm_f", "scale"),
            ("Block_0", "GQAttention_0", "query", "kernel"),
            ("Block_0", "GQAttention_0", "value", "kernel"),
        ):
            np.testing.assert_allclose(
                np.asarray(_descend(twice, path)),
                np.asarray(_descend(once, path)),
                atol=1e-5,
            )

    def test_canonical_energies(self):
        params, graph = _case_params(seed=0)
        canonicalized, _, _ = ScaleCanonicalizer().canonicalize(graph, params)
        for constraint in rmsnorm_scale_constraints(graph):
            gamma = np.asarray(
                _descend(canonicalized, graph.tensors[constraint.scale].path)
            )
            # The signed fold maps gamma to +1, removing the sign freedom.
            np.testing.assert_allclose(gamma, 1.0, atol=1e-4)
        pair = qk_pair_scales(graph, canonicalized)
        for factors in pair.values():
            np.testing.assert_allclose(np.asarray(factors), 1.0, atol=1e-4)
        vo = gqa_vo_balancing_scales(graph, canonicalized)
        for factors in vo.values():
            np.testing.assert_allclose(np.asarray(factors), 1.0, atol=1e-4)

    def test_canonicalization_is_equivariant_on_scale_orbits(self):
        """Scale-equivalent samples collapse to one representative."""

        case = make_rmsnorm_gqa_rope_transformer_scaled_orbit_case(seed=0)
        canonicalizer = ScaleCanonicalizer()
        ref_n, _, _ = canonicalizer.canonicalize(case.graph, case.reference)
        tgt_n, _, _ = canonicalizer.canonicalize(case.graph, case.target)
        state = TransformState.from_transforms(
            case.graph,
            {g: np.asarray(p) for g, p in case.expected_transforms.items()},
        )
        aligned = case.graph.apply_transforms(tgt_n, state)
        for leaf_ref, leaf_aligned in zip(
            jax.tree_util.tree_leaves(ref_n),
            jax.tree_util.tree_leaves(aligned),
            strict=True,
        ):
            np.testing.assert_allclose(
                np.asarray(leaf_aligned), np.asarray(leaf_ref), atol=1e-4
            )

    def test_strategy_option_rejected(self):
        params, graph = _case_params(seed=0)
        with pytest.raises(ValueError, match="strategy"):
            ScaleCanonicalizer().canonicalize(graph, params, strategy="unit_norm")


class TestOrbitRecovery:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_exact_orbit_recovers(self, seed):
        case = make_rmsnorm_gqa_rope_transformer_orbit_case(seed=seed)
        result = run_alignment_benchmark(case, canonicalize=False, rng_seed=seed)
        metrics = result.metrics
        assert metrics.function_drift_max < DRIFT_TOL
        assert metrics.recovered_transform_error < 1e-7
        assert metrics.optimality_gap is not None
        assert metrics.optimality_gap < 1e-4

    def test_scaled_orbit_recovers_with_canonicalization(self):
        # Seed 0 is a plain-LAP coordinate-descent stall (as for the FRN
        # residual case); the mixed schedule is the regression pin.
        schedule = default_schedule_grid()["lap_sinkhorn_lap"]
        case = make_rmsnorm_gqa_rope_transformer_scaled_orbit_case(seed=0)
        schedule = [
            {
                **step,
                **(
                    {
                        "groups": [
                            group_id
                            for group_id, group in case.graph.groups.items()
                            if group.transform_family == "permutation"
                        ]
                    }
                    if step["solver"] == "sinkhorn"
                    else {}
                ),
            }
            for step in schedule
        ]
        result = run_alignment_benchmark(
            case, schedule=schedule, canonicalize=True, rng_seed=0
        )
        metrics = result.metrics
        assert metrics.function_drift_max < DRIFT_TOL
        assert metrics.recovered_transform_error < 1e-7
        assert metrics.optimality_gap is not None
        assert metrics.optimality_gap < 1e-4

    def test_scaled_orbit_needs_canonicalization(self):
        case = make_rmsnorm_gqa_rope_transformer_scaled_orbit_case(seed=1)
        result = run_alignment_benchmark(case, canonicalize=False, rng_seed=1)
        assert result.metrics.distance_after > 1.0

    @pytest.mark.parametrize("seed", [0, 1])
    def test_rotated_orbit_recovers_rotations(self, seed):
        """Per-pair qk rotations are matched by the closed-form projection."""

        case = make_rmsnorm_gqa_rope_transformer_rotated_orbit_case(seed=seed)
        result = run_alignment_benchmark(case, canonicalize=True, rng_seed=seed)
        metrics = result.metrics
        assert metrics.function_drift_max < DRIFT_TOL
        assert metrics.recovered_transform_error < 1e-4
        assert metrics.optimality_gap is not None
        assert metrics.optimality_gap < 1e-4
        assert metrics.transform_validity_error < 1e-5

    @pytest.mark.parametrize("seed", [0, 1])
    def test_orthogonal_orbit_recovers_via_procrustes(self, seed):
        """A random orthogonal stream copy is recovered in closed form."""

        schedule = [{"solver": "procrustes", "max_sweeps": 5, "tolerance": 0.0}]
        case = make_rmsnorm_gqa_rope_transformer_orthogonal_orbit_case(seed=seed)
        result = run_alignment_benchmark(
            case, schedule=schedule, canonicalize=True, rng_seed=seed
        )
        metrics = result.metrics
        assert metrics.function_drift_max < DRIFT_TOL
        assert metrics.recovered_transform_error < 1e-4
        assert metrics.optimality_gap is not None
        assert metrics.optimality_gap < 1e-4

    def test_procrustes_requires_folded_gammas(self):
        """Orthogonal stream alignment on unfolded parameters fails loudly."""

        params, _ = _case_params(seed=0)
        recipe = RMSNormGQARoPETransformerRecipe(stream_transform_family="orthogonal")
        graph = recipe.build_graph(params)
        schedule = [{"solver": "procrustes", "max_sweeps": 2, "tolerance": 0.0}]
        with pytest.raises(ValueError, match="folded"):
            match_sample(
                graph,
                params,
                params,
                objective="euclidean",
                schedule=schedule,
                rng_key=jax.random.PRNGKey(0),
            )

    def test_procrustes_rejects_non_orthogonal_groups(self):
        case = make_rmsnorm_gqa_rope_transformer_orthogonal_orbit_case(seed=0)
        schedule = [
            {
                "solver": "procrustes",
                "max_sweeps": 2,
                "tolerance": 0.0,
                "groups": ["Block_0/ffn/h0"],
            }
        ]
        with pytest.raises(ValueError, match="orthogonal"):
            match_sample(
                case.graph,
                case.reference,
                case.target,
                objective="euclidean",
                schedule=schedule,
                rng_key=jax.random.PRNGKey(0),
            )

    def test_sinkhorn_rejects_non_permutation_groups(self):
        """The doubly stochastic relaxation is invalid for larger families."""

        from align.matching.solvers import SinkhornSolver, SolverStep

        _, graph = _case_params(seed=0)
        step = SolverStep(solver="sinkhorn", max_steps=1)
        solver = SinkhornSolver(get_objective("euclidean"), step)
        with pytest.raises(ValueError, match="only plain permutation"):
            solver._groups(graph)


class TestFlatTransformerSignSymmetries:
    """Circuit sign symmetries also exist without RoPE (LayerNorm family)."""

    def test_qk_vo_sign_flips_preserve_function_and_are_recovered(self):
        from align.architectures import LayerNormMHATransformerRecipe
        from benchmarks import (
            layernorm_mha_transformer_apply,
            make_layernorm_mha_transformer_params,
        )
        from benchmarks.synthetic import (
            signed_permutation_matrix,
            tree_l2_distance,
        )

        params = make_layernorm_mha_transformer_params(key=jax.random.PRNGKey(3))
        graph = LayerNormMHATransformerRecipe().build_graph(params)
        assert all(
            graph.groups[gid].transform_family == "signed_permutation"
            for gid in graph.groups
            if "/qk" in gid or "/vo" in gid
        )
        rng = np.random.default_rng(17)
        forward = {
            gid: signed_permutation_matrix(rng, group.size)
            if group.transform_family == "signed_permutation"
            else _perm_matrix(rng.permutation(group.size))
            for gid, group in graph.groups.items()
        }
        target = graph.apply_transforms(
            params, TransformState.from_transforms(graph, forward)
        )

        tokens = jax.random.randint(jax.random.PRNGKey(4), (4, 5), 0, 7, jnp.int32)
        drift = float(
            np.max(
                np.abs(
                    np.asarray(layernorm_mha_transformer_apply(target, tokens))
                    - np.asarray(layernorm_mha_transformer_apply(params, tokens))
                )
            )
        )
        assert drift < DRIFT_TOL

        aligned, _, _ = match_sample(
            graph,
            params,
            target,
            objective="euclidean",
            schedule=[{"solver": "lap", "max_sweeps": 25, "tolerance": 0.0}],
            rng_key=jax.random.PRNGKey(0),
        )
        assert tree_l2_distance(params, aligned) < 1e-4


def test_synthetic_rmsnorm_gqa_rope_transformer_posterior_collapses():
    case = make_synthetic_rmsnorm_gqa_rope_transformer_posterior_case(
        seed=0, n_chains=2, n_samples=4
    )
    result = run_posterior_benchmark(
        case, canonicalize=True, rng_seed=0, barycenter_passes=1
    )
    before, after = result.metrics_before, result.metrics_after
    assert before.split_rhat_mean > 5.0
    assert after.split_rhat_mean < 1.2
    assert before.weight_averaging_gap > 0.2
    assert after.weight_averaging_gap < 0.05
    assert result.function_drift_max is not None
    assert result.function_drift_max < 1e-3
    assert result.head_assignment_accuracy == 1.0
    assert result.head_assignment_exact_rate == 1.0
