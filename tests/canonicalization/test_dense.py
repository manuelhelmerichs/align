"""Math tests for scale canonicalization via the graph-native ScaleCanonicalizer."""

import jax.numpy as jnp
import pytest

from align.architectures import MLPRecipe
from align.canonicalization import (
    DenseLayer,
    ScaleCanonicalizer,
    compute_incoming_norms,
    validate_activation,
)


def _descend(mapping, path):
    node = mapping
    for key in path:
        node = node[key]
    return node


def _canonicalize(layers, **kwargs):
    """Canonicalize an ordered list of dense layers through ``ScaleCanonicalizer``.

    Builds a dense-MLP ``SymmetryGraph`` from the layers, runs the graph-native
    canonicalizer, and unpacks the result back into a ``DenseLayer`` list aligned with
    the input order so the math assertions can stay layer-oriented. These tests
    pin the ``unit_norm`` canonical form; ``balanced`` has its own suite in
    ``test_balanced_scale.py``.
    """
    kwargs.setdefault("strategy", "unit_norm")
    params = {
        "fcn": {
            f"Dense_{idx}": {"kernel": layer.kernel, "bias": layer.bias}
            for idx, layer in enumerate(layers)
        }
    }
    graph = MLPRecipe(parameter_root="fcn").build_graph(params)
    normalized_params, scales, aux = ScaleCanonicalizer().canonicalize(
        graph, params, **kwargs
    )
    normalized_layers = [
        DenseLayer(
            kernel=jnp.asarray(_descend(normalized_params, (*path, "kernel"))),
            bias=jnp.asarray(_descend(normalized_params, (*path, "bias"))),
        )
        for path in graph.metadata["layer_paths"]
    ]
    return normalized_layers, scales, aux


class TestComputeIncomingNorms:
    def test_simple_layer(self):
        layer = DenseLayer(
            kernel=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            bias=jnp.array([0.0, 0.0]),
        )
        norms = compute_incoming_norms(layer)
        assert jnp.allclose(norms, jnp.array([1.0, 1.0]), atol=1e-6)

    def test_with_bias(self):
        layer = DenseLayer(
            kernel=jnp.array([[3.0, 0.0], [0.0, 1.0]]),
            bias=jnp.array([4.0, 0.0]),
        )
        norms = compute_incoming_norms(layer)
        assert jnp.allclose(norms, jnp.array([5.0, 1.0]), atol=1e-6)


class TestCanonicalize:
    @staticmethod
    def _forward(seq, vec):
        h = jnp.maximum(0, seq[0].kernel.T @ vec + seq[0].bias)
        return seq[1].kernel.T @ h + seq[1].bias

    def test_function_preservation_regression(self):
        layers = [
            DenseLayer(
                kernel=jnp.array([[2.0, 1.0], [1.0, 2.0]]), bias=jnp.array([0.5, -0.5])
            ),
            DenseLayer(kernel=jnp.array([[1.0], [1.0]]), bias=jnp.array([0.0])),
        ]

        canonicalized, _, _ = _canonicalize(layers)

        x = jnp.array([1.0, 1.0])

        y_orig = self._forward(layers, x)
        y_norm = self._forward(canonicalized, x)

        assert jnp.allclose(y_orig, y_norm, atol=1e-5)

    def test_unit_norm_constraint(self):
        layers = [
            DenseLayer(
                kernel=jnp.array([[3.0, 0.0], [0.0, 4.0]]), bias=jnp.array([4.0, 3.0])
            ),
            DenseLayer(kernel=jnp.array([[1.0], [1.0]]), bias=jnp.array([0.0])),
        ]

        canonicalized, _, _ = _canonicalize(layers)

        for k in range(canonicalized[0].kernel.shape[1]):
            w_k = canonicalized[0].kernel[:, k]
            b_k = canonicalized[0].bias[k]
            norm_k = jnp.sqrt(jnp.sum(w_k**2) + b_k**2)
            assert jnp.allclose(norm_k, 1.0, atol=1e-5)

    def test_exclude_bias_from_norm_still_scales_biases_to_preserve_function(self):
        layers = [
            DenseLayer(
                kernel=jnp.array([[3.0, 0.0], [0.0, 4.0]]),
                bias=jnp.array([4.0, 3.0]),
            ),
            DenseLayer(kernel=jnp.array([[1.5], [-0.75]]), bias=jnp.array([0.25])),
        ]

        x = jnp.array([1.25, -0.5])
        canonicalized, scales, aux = _canonicalize(
            layers,
            include_bias_in_norm=False,
        )

        assert aux["include_bias_in_norm"] is False
        assert jnp.allclose(scales["mlp/h0"], jnp.array([3.0, 4.0]), atol=1e-6)
        assert jnp.allclose(
            canonicalized[0].bias, jnp.array([4.0 / 3.0, 3.0 / 4.0]), atol=1e-6
        )
        assert jnp.allclose(
            jnp.sqrt(jnp.sum(canonicalized[0].kernel ** 2, axis=0)),
            jnp.ones(2),
            atol=1e-6,
        )
        assert jnp.allclose(
            self._forward(layers, x), self._forward(canonicalized, x), atol=1e-5
        )

    def test_scale_invariance_multiple_hidden_layers(self):
        layers = [
            DenseLayer(
                kernel=jnp.arange(12.0).reshape(3, 4) / 10.0,
                bias=jnp.array([0.1, -0.2, 0.3, -0.4]),
            ),
            DenseLayer(
                kernel=jnp.arange(20.0).reshape(4, 5) / 7.0,
                bias=jnp.array([0.2, 0.1, -0.1, 0.0, 0.3]),
            ),
            DenseLayer(
                kernel=jnp.arange(10.0).reshape(5, 2) / 5.0,
                bias=jnp.array([0.0, 0.1]),
            ),
        ]

        s1 = jnp.array([1.5, 0.5, 2.0, 0.8])
        s2 = jnp.array([0.7, 1.2, 0.9, 1.1, 1.3])
        scaled = [
            DenseLayer(kernel=layers[0].kernel / s1, bias=layers[0].bias / s1),
            DenseLayer(
                kernel=(s1[:, None] * layers[1].kernel) / s2,
                bias=layers[1].bias / s2,
            ),
            DenseLayer(kernel=s2[:, None] * layers[2].kernel, bias=layers[2].bias),
        ]

        canonicalized, _, _ = _canonicalize(layers)
        canonicalized_scaled, _, _ = _canonicalize(scaled)

        for left, right in zip(canonicalized, canonicalized_scaled, strict=True):
            assert jnp.allclose(left.kernel, right.kernel, atol=1e-6)
            assert jnp.allclose(left.bias, right.bias, atol=1e-6)

    def test_unit_norm_constraint_multiple_hidden_layers(self):
        layers = [
            DenseLayer(
                kernel=jnp.arange(12.0).reshape(3, 4) / 10.0,
                bias=jnp.array([0.1, -0.2, 0.3, -0.4]),
            ),
            DenseLayer(
                kernel=jnp.arange(20.0).reshape(4, 5) / 7.0,
                bias=jnp.array([0.2, 0.1, -0.1, 0.0, 0.3]),
            ),
            DenseLayer(
                kernel=jnp.arange(10.0).reshape(5, 2) / 5.0,
                bias=jnp.array([0.0, 0.1]),
            ),
        ]

        canonicalized, _, _ = _canonicalize(layers)
        for layer in canonicalized[:-1]:
            norms = jnp.sqrt(jnp.sum(layer.kernel**2, axis=0) + layer.bias**2)
            assert jnp.allclose(norms, jnp.ones_like(norms), atol=1e-5)

    def test_invalid_degenerate_handling_raises(self):
        layers = [
            DenseLayer(kernel=jnp.array([[1.0], [0.0]]), bias=jnp.array([0.0])),
            DenseLayer(kernel=jnp.array([[1.0]]), bias=jnp.array([0.0])),
        ]

        with pytest.raises(ValueError, match="Unknown degenerate_channels"):
            _canonicalize(layers, degenerate_channels="missing")


class TestDegenerateNeurons:
    def test_zero_neuron_zeroes_outgoing_when_requested(self):
        layers = [
            DenseLayer(
                kernel=jnp.array([[1.0, 0.0], [1.0, 0.0]]), bias=jnp.array([1.0, 0.0])
            ),
            DenseLayer(kernel=jnp.array([[1.0], [1.0]]), bias=jnp.array([0.0])),
        ]

        canonicalized, scales, _ = _canonicalize(
            layers,
            degenerate_channels="zero_outgoing",
        )

        assert jnp.allclose(
            canonicalized[1].kernel[1], jnp.zeros_like(canonicalized[1].kernel[1])
        )
        assert scales["mlp/h0"].shape[0] == 2

    def test_near_zero_neuron_counts_as_degenerate(self):
        layers = [
            DenseLayer(
                kernel=jnp.array([[1e-5, 1.0], [0.0, 1.0]], dtype=jnp.float32),
                bias=jnp.array([0.0, 0.0], dtype=jnp.float32),
            ),
            DenseLayer(
                kernel=jnp.array([[1.0], [1.0]], dtype=jnp.float32),
                bias=jnp.array([0.0], dtype=jnp.float32),
            ),
        ]

        canonicalized, _, _ = _canonicalize(
            layers,
            degenerate_channels="zero_outgoing",
        )

        assert jnp.allclose(
            canonicalized[1].kernel[0], jnp.zeros_like(canonicalized[1].kernel[0])
        )
        assert not jnp.allclose(
            canonicalized[1].kernel[1], jnp.zeros_like(canonicalized[1].kernel[1])
        )


class TestActivationValidation:
    def test_leaky_relu_uses_shared_norms(self):
        layer = DenseLayer(
            kernel=jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
            bias=jnp.array([0.5, -0.5], dtype=jnp.float32),
        )
        relu_norms = compute_incoming_norms(layer)
        _, scales, aux = _canonicalize(
            [
                layer,
                DenseLayer(
                    kernel=jnp.array([[1.0], [1.0]], dtype=jnp.float32),
                    bias=jnp.array([0.0], dtype=jnp.float32),
                ),
            ],
            activation="leaky_relu",
        )
        assert aux["activation"] == "leaky_relu"
        assert jnp.allclose(scales["mlp/h0"], relu_norms, atol=1e-6)

    def test_activation_kwargs_rejected(self):
        with pytest.raises(ValueError, match="activation_kwargs"):
            validate_activation("leaky_relu", {"negative_slope": 0.2})

    def test_unknown_activation_rejected(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            validate_activation("swish")


class TestRecipeCanonicalize:
    """Integration checks for recipe-driven canonicalization."""

    def test_recipe_canonicalize_basic(self):
        params = {
            "fcn": {
                "Dense_0": {
                    "kernel": jnp.array([[2.0, 1.0], [1.0, 2.0]]),
                    "bias": jnp.array([0.5, -0.5]),
                },
                "Dense_1": {
                    "kernel": jnp.array([[1.0], [1.0]]),
                    "bias": jnp.array([0.0]),
                },
            }
        }

        recipe = MLPRecipe(parameter_root="fcn")
        graph = recipe.build_graph(params)
        normalized_params, scales, aux = ScaleCanonicalizer().canonicalize(
            graph, params
        )

        assert "fcn" in normalized_params
        assert len(scales) == 1  # One hidden layer
        assert aux["plan"] == "dense_chain"

    def test_recipe_canonicalize_rejects_linear_model_without_hidden_group(self):
        params = {
            "fcn": {
                "Dense_0": {
                    "kernel": jnp.array([[2.0], [1.0]]),
                    "bias": jnp.array([0.5]),
                },
            }
        }

        recipe = MLPRecipe(parameter_root="fcn")
        graph = recipe.build_graph(params)
        with pytest.raises(ValueError, match="at least one hidden permutation group"):
            ScaleCanonicalizer().canonicalize(graph, params)

    def test_recipe_canonicalize_with_all_kwargs(self):
        params = {
            "fcn": {
                "Dense_0": {
                    "kernel": jnp.array([[2.0, 1.0], [1.0, 2.0]]),
                    "bias": jnp.array([0.5, -0.5]),
                },
                "Dense_1": {
                    "kernel": jnp.array([[1.0], [1.0]]),
                    "bias": jnp.array([0.0]),
                },
            }
        }

        recipe = MLPRecipe(parameter_root="fcn")
        graph = recipe.build_graph(params)
        normalized_params, scales, aux = ScaleCanonicalizer().canonicalize(
            graph,
            params,
            epsilon=1e-8,
            degenerate_channels="preserve",
            include_bias_in_norm=True,
            activation="relu",
        )

        assert aux["epsilon"] == 1e-8
        assert aux["degenerate_channels"] == "preserve"
        assert aux["include_bias_in_norm"] is True
        assert aux["activation"] == "relu"


class TestScaleCanonicalizerTypeHandling:
    """Tests to ensure the canonicalizer handles types correctly."""

    def test_preserves_already_canonical_network(self):
        layers = [
            DenseLayer(
                kernel=jnp.array([[1.0, 0.0], [0.0, 1.0]]), bias=jnp.array([0.0, 0.0])
            ),
            DenseLayer(kernel=jnp.array([[1.0], [1.0]]), bias=jnp.array([0.0])),
        ]
        canonicalized, scales, aux = _canonicalize(layers, epsilon=1e-8)

        assert len(canonicalized) == 2
        assert aux["epsilon"] == 1e-8
        assert jnp.allclose(scales["mlp/h0"], jnp.ones(2), atol=1e-6)
        assert jnp.allclose(canonicalized[0].kernel, layers[0].kernel, atol=1e-6)
        assert jnp.allclose(canonicalized[1].kernel, layers[1].kernel, atol=1e-6)

    def test_compute_incoming_norms_with_float_epsilon(self):
        """compute_incoming_norms should return the expected per-neuron norms."""
        layer = DenseLayer(
            kernel=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            bias=jnp.array([0.0, 0.0]),
        )
        norms = compute_incoming_norms(layer, epsilon=1e-8, include_bias_in_norm=True)

        assert norms.shape == (2,)
        assert jnp.allclose(norms, jnp.ones(2), atol=1e-6)

    def test_compute_incoming_norms_exclude_bias_from_norm(self):
        """compute_incoming_norms should handle include_bias_in_norm=False."""
        layer = DenseLayer(
            kernel=jnp.array([[3.0, 0.0], [0.0, 4.0]]),
            bias=jnp.array([4.0, 3.0]),  # Would contribute to norm if included
        )
        norms_with_bias = compute_incoming_norms(layer, include_bias_in_norm=True)
        norms_without_bias = compute_incoming_norms(layer, include_bias_in_norm=False)

        # With bias: sqrt(3^2 + 4^2) = 5, sqrt(4^2 + 3^2) = 5
        # Without bias: sqrt(3^2) = 3, sqrt(4^2) = 4
        assert jnp.allclose(norms_with_bias, jnp.array([5.0, 5.0]), atol=1e-6)
        assert jnp.allclose(norms_without_bias, jnp.array([3.0, 4.0]), atol=1e-6)
