"""LayerNorm + MHA transformer structure, invariance, and matching tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures import LayerNormMHATransformerRecipe, MLPRecipe
from align.canonicalization import ScaleCanonicalizer
from align.matching import (
    TransformState,
    match_component_across,
    match_sample,
)
from align.symmetry import MHACircuitConstraint
from benchmarks import (
    layernorm_mha_transformer_apply,
    make_layernorm_mha_transformer_orbit_case,
    make_layernorm_mha_transformer_params,
    run_alignment_benchmark,
)
from benchmarks.synthetic import layer_norm, mhdpa_apply


def _perm_matrix(indices) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    mat = np.zeros((indices.shape[0], indices.shape[0]), dtype=np.float64)
    mat[np.arange(indices.shape[0]), indices] = 1.0
    return mat


def _random_state(graph, seed: int) -> TransformState:
    rng = np.random.default_rng(seed)
    perms = {
        group_id: _perm_matrix(rng.permutation(group.size))
        for group_id, group in graph.groups.items()
    }
    return TransformState.from_transforms(graph, perms)


# --- ViT-style synthetic tree + faithful forward -----------------------------


def make_vit_style_params(
    *,
    key: jax.Array,
    image_size: int = 4,
    patch_size: int = 2,
    d_model: int = 8,
    num_heads: int = 2,
    num_blocks: int = 2,
    ffn_dim: int = 12,
    classifier_dim: int = 6,
    num_classes: int = 3,
) -> dict:
    head_dim = d_model // num_heads
    num_patches = (image_size // patch_size) ** 2

    def _normal(rng_key, shape, scale):
        return scale * jax.random.normal(rng_key, shape, dtype=jnp.float32)

    keys = iter(jax.random.split(key, 16 * num_blocks + 16))
    transformer = {}
    for block in range(num_blocks):
        transformer[f"TransformerBlock_{block}"] = {
            "LayerNorm_0": {
                "scale": 1.0 + _normal(next(keys), (d_model,), 0.1),
                "bias": _normal(next(keys), (d_model,), 0.1),
            },
            "LayerNorm_1": {
                "scale": 1.0 + _normal(next(keys), (d_model,), 0.1),
                "bias": _normal(next(keys), (d_model,), 0.1),
            },
            "SelfAttention_0": {
                "query": {
                    "kernel": _normal(next(keys), (d_model, num_heads, head_dim), 0.6),
                    "bias": _normal(next(keys), (num_heads, head_dim), 0.1),
                },
                "key": {
                    "kernel": _normal(next(keys), (d_model, num_heads, head_dim), 0.6),
                    "bias": _normal(next(keys), (num_heads, head_dim), 0.1),
                },
                "value": {
                    "kernel": _normal(next(keys), (d_model, num_heads, head_dim), 0.6),
                    "bias": _normal(next(keys), (num_heads, head_dim), 0.1),
                },
                "out": {
                    "kernel": _normal(next(keys), (num_heads, head_dim, d_model), 0.6),
                    "bias": _normal(next(keys), (d_model,), 0.1),
                },
            },
            "FullyConnected_0": {
                "layer0": {
                    "kernel": _normal(next(keys), (d_model, ffn_dim), 0.6),
                    "bias": _normal(next(keys), (ffn_dim,), 0.1),
                },
                "layer1": {
                    "kernel": _normal(next(keys), (ffn_dim, d_model), 0.6),
                    "bias": _normal(next(keys), (d_model,), 0.1),
                },
            },
        }
    core = {
        "Transformer_0": transformer,
        "patch_embedding": {
            "kernel": _normal(next(keys), (patch_size, patch_size, 3, d_model), 0.5),
            "bias": _normal(next(keys), (d_model,), 0.1),
        },
        "class_tokens": _normal(next(keys), (1, 1, d_model), 0.5),
        "pos_embedding": _normal(next(keys), (1, num_patches + 1, d_model), 0.5),
        "mlp_fc1": {
            "kernel": _normal(next(keys), (d_model, classifier_dim), 0.6),
            "bias": _normal(next(keys), (classifier_dim,), 0.1),
        },
        "mlp_output": {
            "kernel": _normal(next(keys), (classifier_dim, num_classes), 0.6),
            "bias": _normal(next(keys), (num_classes,), 0.1),
        },
    }
    return {"core": core}


def vit_transformer_apply(params, images: jax.Array) -> jax.Array:
    """Forward pass matching the bundled ViTCore semantics (true pre-LN)."""

    core = params["core"]
    kernel = jnp.asarray(core["patch_embedding"]["kernel"])
    patch_size = kernel.shape[0]
    patches = jax.lax.conv_general_dilated(
        images,
        kernel,
        window_strides=(patch_size, patch_size),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + jnp.asarray(core["patch_embedding"]["bias"])
    batch = images.shape[0]
    d_model = kernel.shape[-1]
    patches = patches.reshape(batch, -1, d_model)

    cls = jnp.tile(jnp.asarray(core["class_tokens"]), (batch, 1, 1))
    x = jnp.concatenate([cls, patches], axis=1)
    x = x + jnp.asarray(core["pos_embedding"])

    blocks = core["Transformer_0"]
    for name in sorted(blocks, key=lambda item: int(item.split("_")[-1])):
        block = blocks[name]
        x_norm = layer_norm(x, block["LayerNorm_0"])
        x = x + mhdpa_apply(block["SelfAttention_0"], x_norm, causal=False)
        x_norm = layer_norm(x, block["LayerNorm_1"])
        ffn = block["FullyConnected_0"]
        hidden = jax.nn.gelu(
            x_norm @ jnp.asarray(ffn["layer0"]["kernel"])
            + jnp.asarray(ffn["layer0"]["bias"])
        )
        x = x + (
            hidden @ jnp.asarray(ffn["layer1"]["kernel"])
            + jnp.asarray(ffn["layer1"]["bias"])
        )

    pooled = x[:, 0]
    hidden = jax.nn.gelu(
        pooled @ jnp.asarray(core["mlp_fc1"]["kernel"])
        + jnp.asarray(core["mlp_fc1"]["bias"])
    )
    return hidden @ jnp.asarray(core["mlp_output"]["kernel"]) + jnp.asarray(
        core["mlp_output"]["bias"]
    )


@pytest.fixture(scope="module")
def bundled_gpt_params():
    """Deterministic fixture matching the bundled GPT layout."""

    return make_layernorm_mha_transformer_params(
        key=jax.random.PRNGKey(20),
        vocab_size=23,
        context_len=8,
        d_model=16,
        num_heads=2,
        num_blocks=2,
        ffn_dim=64,
    )


@pytest.fixture(scope="module")
def bundled_vit_params():
    """Deterministic fixture matching the bundled ViT layout."""

    return make_vit_style_params(
        key=jax.random.PRNGKey(21),
        image_size=16,
        patch_size=4,
        d_model=16,
        num_heads=2,
        num_blocks=2,
        ffn_dim=32,
        classifier_dim=16,
        num_classes=5,
    )


# --- structure ---------------------------------------------------------------


class TestTransformerStructure:
    def test_gpt_style_blocks_and_groups(self):
        params = make_layernorm_mha_transformer_params(key=jax.random.PRNGKey(0))
        graph = LayerNormMHATransformerRecipe().build_graph(params)

        assert set(graph.components) == {
            "stream",
            "Block_0/attention",
            "Block_0/ffn",
            "Block_1/attention",
            "Block_1/ffn",
        }
        assert graph.components["Block_0/attention"].kind == "mha"
        assert graph.components["Block_0/ffn"].kind == "dense_chain"
        assert graph.groups["stream"].size == 8
        assert graph.groups["Block_0/attention/heads"].size == 2
        assert graph.groups["Block_0/attention/qk0"].size == 4
        assert graph.groups["Block_0/ffn/h0"].size == 12
        assert graph.metadata["d_model"] == 8
        constraints = [
            constraint
            for constraint in graph.constraints
            if isinstance(constraint, MHACircuitConstraint)
        ]
        assert len(constraints) == 2

    def test_vit_style_blocks_include_head_chain(self):
        params = make_vit_style_params(key=jax.random.PRNGKey(1))
        graph = LayerNormMHATransformerRecipe(parameter_root="core").build_graph(params)

        assert "classifier" in graph.components
        assert graph.groups["classifier/h0"].size == 6
        component_ids = set(graph.components)
        assert "Transformer_0/TransformerBlock_0/attention" in component_ids
        assert "Transformer_0/TransformerBlock_1/ffn" in component_ids
        # Embeddings bind the stream on their trailing axis.
        stream_tensors = {
            binding.tensor_id for binding in graph.bindings_for_group("stream")
        }
        assert "core/class_tokens" in stream_tensors
        assert "core/pos_embedding" in stream_tensors
        assert "core/patch_embedding/kernel" in stream_tensors

    def test_bundled_gpt_layout(self, bundled_gpt_params):
        graph = LayerNormMHATransformerRecipe().build_graph(bundled_gpt_params)
        assert graph.metadata["d_model"] == 16
        assert graph.groups["Block_0/attention/heads"].size == 2
        assert graph.groups["Block_0/ffn/h0"].size == 64
        assert "classifier" not in graph.components  # one layer has no junctions

    def test_bundled_vit_layout(self, bundled_vit_params):
        graph = LayerNormMHATransformerRecipe(parameter_root="core").build_graph(
            bundled_vit_params
        )
        assert graph.metadata["d_model"] == 16
        assert graph.groups["classifier/h0"].size == 16
        assert graph.groups["Transformer_0/TransformerBlock_0/ffn/h0"].size == 32

    def test_non_transformer_tree_rejected(self):
        params = {"fcn": {"dense0": {"kernel": np.zeros((3, 4))}}}
        with pytest.raises(ValueError, match="No attention modules"):
            LayerNormMHATransformerRecipe().build_graph(params)

    def test_canonicalize_uses_attention_balance_plan(self):
        params = make_layernorm_mha_transformer_params(key=jax.random.PRNGKey(0))
        graph = LayerNormMHATransformerRecipe().build_graph(params)
        _, factors, aux = ScaleCanonicalizer().canonicalize(
            graph, params, activation="gelu"
        )
        assert aux["plan"] == "attention_balance"
        assert len(factors) == len(aux["balanced_groups"]) > 0


# --- function invariance -----------------------------------------------------


class TestTransformerInvariance:
    def test_gpt_random_states_preserve_function(self):
        params = make_layernorm_mha_transformer_params(key=jax.random.PRNGKey(2))
        graph = LayerNormMHATransformerRecipe().build_graph(params)
        tokens = jax.random.randint(jax.random.PRNGKey(3), (4, 5), 0, 7)
        baseline = np.asarray(layernorm_mha_transformer_apply(params, tokens))
        for seed in range(3):
            transformed = graph.apply_transforms(params, _random_state(graph, seed))
            np.testing.assert_allclose(
                np.asarray(layernorm_mha_transformer_apply(transformed, tokens)),
                baseline,
                atol=1e-5,
            )

    def test_vit_random_states_preserve_function(self):
        params = make_vit_style_params(key=jax.random.PRNGKey(4))
        graph = LayerNormMHATransformerRecipe(parameter_root="core").build_graph(params)
        images = jax.random.normal(
            jax.random.PRNGKey(5), (2, 4, 4, 3), dtype=jnp.float32
        )
        baseline = np.asarray(vit_transformer_apply(params, images))
        for seed in range(3):
            transformed = graph.apply_transforms(params, _random_state(graph, seed))
            np.testing.assert_allclose(
                np.asarray(vit_transformer_apply(transformed, images)),
                baseline,
                rtol=1e-6,
                atol=1e-5,
            )

    def test_bundled_gpt_layout_preserves_function(self, bundled_gpt_params):
        graph = LayerNormMHATransformerRecipe().build_graph(bundled_gpt_params)
        tokens = jax.random.randint(jax.random.PRNGKey(6), (3, 8), 0, 23)
        baseline = np.asarray(
            layernorm_mha_transformer_apply(bundled_gpt_params, tokens)
        )
        transformed = graph.apply_transforms(
            bundled_gpt_params, _random_state(graph, seed=7)
        )
        np.testing.assert_allclose(
            np.asarray(layernorm_mha_transformer_apply(transformed, tokens)),
            baseline,
            atol=1e-4,
        )

    def test_bundled_vit_layout_preserves_function(self, bundled_vit_params):
        graph = LayerNormMHATransformerRecipe(parameter_root="core").build_graph(
            bundled_vit_params
        )
        images = jax.random.normal(
            jax.random.PRNGKey(8), (2, 16, 16, 3), dtype=jnp.float32
        )
        baseline = np.asarray(vit_transformer_apply(bundled_vit_params, images))
        transformed = graph.apply_transforms(
            bundled_vit_params, _random_state(graph, seed=9)
        )
        np.testing.assert_allclose(
            np.asarray(vit_transformer_apply(transformed, images)),
            baseline,
            atol=1e-4,
        )


# --- exact orbit recovery ----------------------------------------------------


class TestTransformerOrbitRecovery:
    @pytest.mark.parametrize(
        "schedule",
        [
            [{"solver": "lap", "max_sweeps": 10, "tolerance": 0.0}],
            [
                {
                    "solver": "sinkhorn",
                    "max_steps": 30,
                    "tolerance": 1e-5,
                    "init_scale": 0.0,
                    "learning_rate": 0.05,
                },
                {"solver": "lap", "max_sweeps": 10, "tolerance": 0.0},
            ],
        ],
    )
    def test_orbit_case_recovers_exactly(self, schedule):
        case = make_layernorm_mha_transformer_orbit_case(seed=1)
        result = run_alignment_benchmark(case, schedule=schedule)
        assert result.metrics.function_drift_max < 1e-4
        assert result.metrics.distance_after == pytest.approx(0.0, abs=1e-5)
        assert result.metrics.transform_validity_error == 0.0
        assert result.metrics.recovered_transform_error == 0.0

    def test_vit_style_orbit_recovery(self):
        params = make_vit_style_params(key=jax.random.PRNGKey(10))
        graph = LayerNormMHATransformerRecipe(parameter_root="core").build_graph(params)
        target = graph.apply_transforms(params, _random_state(graph, seed=11))
        aligned, _, aux = match_sample(
            graph,
            params,
            target,
            schedule=[{"solver": "lap", "max_sweeps": 10}],
        )
        assert aux["objective_final"] == pytest.approx(0.0, abs=1e-6)
        flat_ref = jax.tree_util.tree_leaves(params)
        flat_aligned = jax.tree_util.tree_leaves(aligned)
        for ref_leaf, aligned_leaf in zip(flat_ref, flat_aligned, strict=True):
            np.testing.assert_allclose(
                np.asarray(aligned_leaf), np.asarray(ref_leaf), atol=1e-5
            )


# --- cross-architecture component matching ------------------------------------


class TestCrossArchitectureFFN:
    def test_gpt_ffn_aligns_mlp_component(self):
        gpt_params = make_layernorm_mha_transformer_params(key=jax.random.PRNGKey(12))
        gpt_graph = LayerNormMHATransformerRecipe().build_graph(gpt_params)

        ffn = gpt_params["Block_0"]["FullyConnected_0"]
        perm = np.random.default_rng(13).permutation(12)
        mlp_params = {
            "params": {
                "fcn": {
                    "dense0": {
                        "kernel": np.asarray(ffn["FFN_layer0"]["kernel"])[:, perm],
                        "bias": np.asarray(ffn["FFN_layer0"]["bias"])[perm],
                    },
                    "dense1": {
                        "kernel": np.asarray(ffn["FFN_layer1"]["kernel"])[perm, :],
                        "bias": np.asarray(ffn["FFN_layer1"]["bias"]),
                    },
                }
            }
        }
        mlp_graph = MLPRecipe().build_graph(mlp_params)

        aligned, perms, aux = match_component_across(
            gpt_graph,
            gpt_params,
            "Block_0/ffn",
            mlp_graph,
            mlp_params,
            "mlp",
            schedule=[{"solver": "lap", "max_sweeps": 5}],
        )

        np.testing.assert_allclose(
            np.asarray(aligned["params"]["fcn"]["dense0"]["kernel"]),
            np.asarray(ffn["FFN_layer0"]["kernel"]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(aligned["params"]["fcn"]["dense0"]["bias"]),
            np.asarray(ffn["FFN_layer0"]["bias"]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(aligned["params"]["fcn"]["dense1"]["kernel"]),
            np.asarray(ffn["FFN_layer1"]["kernel"]),
            atol=1e-6,
        )
        assert aux["group_map"] == {"Block_0/ffn/h0": "mlp/h0"}
        assert aux["objective_final"] == pytest.approx(0.0, abs=1e-8)
