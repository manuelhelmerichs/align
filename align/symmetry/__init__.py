"""Shared graph-native alignment core.

Defines the architecture-derived alignment graph (:class:`SymmetryGraph` and
its tensor/axis/group/constraint types) plus the low-level array and pytree
primitives both symmetry-removal stages build on. The permutation stage lives in
``align.matching`` and the scale stage in ``align.canonicalization``; both consume the
``SymmetryGraph`` and its symmetry actions (``apply`` / ``apply_scales``).
"""

from .components import (
    component_signature,
    describe_symmetry,
    extract_component_graph,
    format_symmetry_description,
    groups_for_components,
    match_component_tensors,
    resolve_component_patterns,
    tensors_for_component,
)
from .graph import (
    TRANSFORM_FAMILIES,
    AxisBinding,
    ComponentSpec,
    GraphConstraint,
    SymmetryGraph,
    SymmetryGroup,
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
    "SymmetryGraph",
    "AxisBinding",
    "ComponentSpec",
    "GraphConstraint",
    "TRANSFORM_FAMILIES",
    "SymmetryGroup",
    "TensorSpec",
    "apply_matrix_to_axis",
    "apply_perm_to_axis",
    "binding_transform_matrix",
    "axis_slice",
    "binding_axis_interval",
    "component_signature",
    "describe_symmetry",
    "extract_component_graph",
    "format_symmetry_description",
    "groups_for_components",
    "match_component_tensors",
    "materialize_many",
    "resolve_component_patterns",
    "tensors_for_component",
]
