"""Scale symmetry removal: canonicalization over symmetry graphs.

Bundles the dense-layer representation (:class:`DenseLayer`), activation
validation, the per-plan scale math (dense chains, conv stacks, attention and
RMSNorm circuits), and the graph-driven
:class:`ScaleCanonicalizer`/:class:`ScaleState` that apply it over an
``align.symmetry.SymmetryGraph``. The softmax-head translation
(:func:`center_softmax_head`) is a separate probability-preserving
post-processing step, deliberately outside the exact-logit canonicalizer.
"""

from .activations import SUPPORTED_ACTIVATIONS, validate_activation
from .dense import DenseLayer, compute_incoming_norms
from .scale import ScaleCanonicalizer
from .softmax_head import center_softmax_head, detect_softmax_head
from .state import ScaleState

__all__ = [
    "DenseLayer",
    "ScaleCanonicalizer",
    "ScaleState",
    "SUPPORTED_ACTIVATIONS",
    "center_softmax_head",
    "compute_incoming_norms",
    "detect_softmax_head",
    "validate_activation",
]
