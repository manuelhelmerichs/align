"""Scale symmetry removal: canonicalization of dense ReLU-style MLPs.

Bundles the dense-layer representation (:class:`DenseLayer`), activation
validation, the scale-canonicalization math kernel, and the graph-driven
:class:`ScaleCanonicalizer`/:class:`ScaleState` that apply it over an
``align.symmetry.SymmetryGraph``.
"""

from .activations import SUPPORTED_ACTIVATIONS, validate_activation
from .dense import DenseLayer, compute_incoming_norms
from .scale import ScaleCanonicalizer
from .state import ScaleState

__all__ = [
    "DenseLayer",
    "ScaleCanonicalizer",
    "ScaleState",
    "SUPPORTED_ACTIVATIONS",
    "compute_incoming_norms",
    "validate_activation",
]
