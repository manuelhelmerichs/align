"""Matrix-valued symmetry matching over graph-native alignment data.

Builds on the shared ``align.symmetry`` graph core: objectives score the graph
tensor distance, solvers optimize supported transform families, a fixed
:class:`SolverSequence` runs the configured steps, and :class:`TransformState`
carries the solution. The
:mod:`align.matching.api` surface ties these together for callers.
"""

from .api import (
    build_solver_sequence,
    default_lap_schedule,
    match_batch,
    match_component_across,
    match_sample,
)
from .fisher import (
    estimate_diag_fisher_tree,
    estimate_diag_fisher_weights,
    load_tensor_weights_npz,
    resolve_calibration_kwargs,
    save_tensor_weights_npz,
)
from .objectives import (
    DiagonalFisherObjective,
    EuclideanObjective,
    Objective,
    RelativeFisherObjective,
    UnsupportedGroupLinearization,
    available_objectives,
    get_objective,
    register_objective,
)
from .relative_fisher import (
    estimate_activation_gram_metrics,
    load_activation_gram_metrics_npz,
    resolve_relative_fisher_calibration_kwargs,
    save_activation_gram_metrics_npz,
)
from .sequence import SolverSequence
from .solvers import (
    LAPGroupSolver,
    ProcrustesGroupSolver,
    SinkhornSolver,
    SolverStep,
    available_solvers,
    solve_orthogonal_maximize,
    solve_rotation_pairs_maximize,
    solve_signed_lap_maximize,
    update_group_transform,
)
from .state import (
    TransformState,
    as_permutation_matrix,
    sinkhorn_operator,
    solve_lap_maximize,
)

__all__ = [
    "DiagonalFisherObjective",
    "EuclideanObjective",
    "LAPGroupSolver",
    "Objective",
    "RelativeFisherObjective",
    "TransformState",
    "ProcrustesGroupSolver",
    "SinkhornSolver",
    "SolverStep",
    "SolverSequence",
    "UnsupportedGroupLinearization",
    "as_permutation_matrix",
    "available_objectives",
    "available_solvers",
    "build_solver_sequence",
    "default_lap_schedule",
    "estimate_diag_fisher_tree",
    "estimate_diag_fisher_weights",
    "estimate_activation_gram_metrics",
    "get_objective",
    "load_tensor_weights_npz",
    "load_activation_gram_metrics_npz",
    "match_batch",
    "match_component_across",
    "match_sample",
    "register_objective",
    "resolve_calibration_kwargs",
    "resolve_relative_fisher_calibration_kwargs",
    "save_tensor_weights_npz",
    "save_activation_gram_metrics_npz",
    "sinkhorn_operator",
    "solve_lap_maximize",
    "solve_orthogonal_maximize",
    "solve_rotation_pairs_maximize",
    "solve_signed_lap_maximize",
    "update_group_transform",
]
