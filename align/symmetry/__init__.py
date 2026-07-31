"""Shared graph-native alignment core.

Defines the architecture-derived alignment graph (:class:`SymmetryGraph` and
its tensor/axis/group/constraint types) plus the low-level array and pytree
primitives both alignment stages build on. Matching lives in ``align.matching``
and scale canonicalization in ``align.canonicalization``; both consume the
``SymmetryGraph`` actions (``apply_transforms`` / ``apply_scales``).
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
from .constraints import (
    ConstraintRecord,
    GQARoPECircuitConstraint,
    MHACircuitConstraint,
    ResidualChannelTie,
    RMSNormScaleConstraint,
)
from .coverage import symmetry_parameter_coverage
from .graph import (
    TRANSFORM_FAMILIES,
    AxisBinding,
    ComponentSpec,
    SymmetryGraph,
    SymmetryGroup,
    TensorSpec,
    materialize_many,
)
from .tensor_ops import (
    apply_matrix_to_axis,
    apply_perm_to_axis,
    binding_axis_interval,
    binding_axis_intervals,
    binding_transform_matrix,
)

__all__ = [
    "SymmetryGraph",
    "AxisBinding",
    "ComponentSpec",
    "ConstraintRecord",
    "GQARoPECircuitConstraint",
    "MHACircuitConstraint",
    "RMSNormScaleConstraint",
    "ResidualChannelTie",
    "TRANSFORM_FAMILIES",
    "SymmetryGroup",
    "TensorSpec",
    "apply_matrix_to_axis",
    "apply_perm_to_axis",
    "binding_transform_matrix",
    "binding_axis_interval",
    "binding_axis_intervals",
    "component_signature",
    "describe_symmetry",
    "extract_component_graph",
    "format_symmetry_description",
    "groups_for_components",
    "match_component_tensors",
    "materialize_many",
    "resolve_component_patterns",
    "symmetry_parameter_coverage",
    "tensors_for_component",
]
