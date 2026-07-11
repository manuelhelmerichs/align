"""Shared builders for matching-objective tests."""

import jax.numpy as jnp
import numpy as np

from align.architectures import MLPRecipe


def permutation_matrix(indices) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    matrix = np.zeros((idx.size, idx.size), dtype=np.float32)
    matrix[np.arange(idx.size), idx] = 1.0
    return matrix


def two_layer_graph(seed: int, hidden: int):
    """Build a single-hidden-group MLP with random reference and target trees."""

    rng = np.random.default_rng(seed)
    recipe = MLPRecipe(parameter_root="fcn")

    def tree():
        return {
            "fcn": {
                "Dense_0": {
                    "kernel": jnp.asarray(
                        0.6 * rng.standard_normal((3, hidden)), dtype=jnp.float32
                    ),
                    "bias": jnp.asarray(
                        0.6 * rng.standard_normal((hidden,)), dtype=jnp.float32
                    ),
                },
                "Dense_1": {
                    "kernel": jnp.asarray(
                        0.6 * rng.standard_normal((hidden, 2)), dtype=jnp.float32
                    ),
                    "bias": jnp.asarray(
                        0.6 * rng.standard_normal((2,)), dtype=jnp.float32
                    ),
                },
            }
        }

    reference, target = tree(), tree()
    graph = recipe.build_graph(reference)
    return graph, reference, target
