"""Shared graph-native alignment core.

Defines the architecture-derived alignment graph (:class:`AlignmentProblem` and
its tensor/axis/group/constraint types) plus the low-level array and pytree
primitives both symmetry-removal stages build on. The permutation stage lives in
``align.rebasin`` and the scale stage in ``align.normalization``; both consume the
``AlignmentProblem`` and its symmetry actions (``apply`` / ``apply_scales``).
"""

from .blocks import (
    block_signature,
    describe_problem,
    extract_block_problem,
    format_problem_listing,
    groups_for_blocks,
    match_block_tensors,
    resolve_block_patterns,
    tensors_for_block,
)
from .problem import (
    GROUP_TRANSFORM_CLASSES,
    AlignmentProblem,
    AxisBinding,
    BlockSpec,
    GraphConstraint,
    PermutationGroup,
    TensorSpec,
    materialize_many,
)
from .tensor_ops import (
    apply_matrix_to_axis,
    apply_perm_to_axis,
    axis_slice,
    binding_axis_interval,
    binding_transform_matrix,
)

__all__ = [
    "AlignmentProblem",
    "AxisBinding",
    "BlockSpec",
    "GraphConstraint",
    "GROUP_TRANSFORM_CLASSES",
    "PermutationGroup",
    "TensorSpec",
    "apply_matrix_to_axis",
    "apply_perm_to_axis",
    "binding_transform_matrix",
    "axis_slice",
    "binding_axis_interval",
    "block_signature",
    "describe_problem",
    "extract_block_problem",
    "format_problem_listing",
    "groups_for_blocks",
    "match_block_tensors",
    "materialize_many",
    "resolve_block_patterns",
    "tensors_for_block",
]
