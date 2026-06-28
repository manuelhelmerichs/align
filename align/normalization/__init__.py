"""Scale symmetry removal: normalization of dense ReLU-style MLPs.

Bundles the dense-layer representation (:class:`DenseLayer`), the activation
normalizer registry, the scale-normalization math kernel, and the graph-driven
:class:`ScaleNormalizer`/:class:`ScaleState` that apply it over an
``align.alignment.AlignmentProblem``.
"""

from .activations import (
    ActivationNormalizer,
    LeakyReLUNormalizer,
    ReLUNormalizer,
    get_activation_normalizer,
    register_activation_normalizer,
)
from .dense_layers import DenseLayer
from .kernel import (
    batch_normalize_layers,
    compute_incoming_norms,
    normalize_hidden_layer,
    normalize_last_layer_classification,
    normalize_layers,
    scale_outgoing_weights,
)
from .scale_normalizer import ScaleNormalizer
from .scale_state import ScaleState

__all__ = [
    "ActivationNormalizer",
    "DenseLayer",
    "LeakyReLUNormalizer",
    "ReLUNormalizer",
    "ScaleNormalizer",
    "ScaleState",
    "batch_normalize_layers",
    "compute_incoming_norms",
    "get_activation_normalizer",
    "normalize_hidden_layer",
    "normalize_last_layer_classification",
    "normalize_layers",
    "register_activation_normalizer",
    "scale_outgoing_weights",
]
