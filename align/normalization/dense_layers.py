"""Dense-layer data structures used by dense MLP utilities.

This module intentionally centralizes dense-layer representations so that
architecture adapters and normalization do not depend on `align.rebasin` for
dense-specific helpers.
"""

from dataclasses import dataclass

import jax


@dataclass
class DenseLayer:
    """Dense layer parameters.

    Conventions:
    - kernel has shape (in_features, out_features)
    - bias has shape (out_features,)
    """

    kernel: jax.Array
    bias: jax.Array


def _denselayer_flatten(layer: DenseLayer):
    return (layer.kernel, layer.bias), None


def _denselayer_unflatten(_, data):
    kernel, bias = data
    return DenseLayer(kernel=kernel, bias=bias)


jax.tree_util.register_pytree_node(
    DenseLayer, _denselayer_flatten, _denselayer_unflatten
)


__all__ = ["DenseLayer"]
