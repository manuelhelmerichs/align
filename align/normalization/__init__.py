"""Scale symmetry removal: normalization of dense ReLU-style MLPs.

Bundles the dense-layer representation (:class:`DenseLayer`), activation
validation, the scale-normalization math kernel, and the graph-driven
:class:`ScaleNormalizer`/:class:`ScaleState` that apply it over an
``align.alignment.AlignmentProblem``.
"""

from .activations import SUPPORTED_ACTIVATIONS, validate_activation
from .dense_layers import DenseLayer
from .kernel import (
    compute_incoming_norms,
    normalize_layers,
)
from .scale_normalizer import ScaleNormalizer
from .scale_state import ScaleState

__all__ = [
    "DenseLayer",
    "ScaleNormalizer",
    "ScaleState",
    "SUPPORTED_ACTIVATIONS",
    "compute_incoming_norms",
    "normalize_layers",
    "validate_activation",
]
