"""Shared graph-native alignment core.

Defines the architecture-derived alignment graph (:class:`AlignmentProblem` and
its tensor/axis/group/constraint types) plus the low-level array and pytree
primitives both symmetry-removal stages build on. The permutation stage lives in
``align.rebasin`` and the scale stage in ``align.normalization``; both consume the
``AlignmentProblem`` and its symmetry actions (``apply`` / ``apply_scales``).
"""

from .problem import (
    AlignmentProblem,
    AxisBinding,
    GraphConstraint,
    PermutationGroup,
    TensorSpec,
    binding_axis_interval,
    materialize_many,
)
from .tensor_ops import apply_perm_to_axis

__all__ = [
    "AlignmentProblem",
    "AxisBinding",
    "GraphConstraint",
    "PermutationGroup",
    "TensorSpec",
    "apply_perm_to_axis",
    "binding_axis_interval",
    "materialize_many",
]
