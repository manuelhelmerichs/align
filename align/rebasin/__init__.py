"""Permutation symmetry removal (rebasin) for graph-native alignment problems.

Builds on the shared ``align.alignment`` graph core: objectives score the graph
tensor distance, solvers (LAP/Sinkhorn) optimize permutations, the scheduler
runs a solver schedule, and :class:`PermutationState` carries the solution. The
:mod:`align.rebasin.api` surface ties these together for callers.
"""

from .api import (
    build_scheduler,
    default_lap_schedule,
    rebasin_batch,
    rebasin_block_across,
    rebasin_single_sample,
)
from .fisher import (
    estimate_diag_fisher_tree,
    estimate_diag_fisher_weights,
    load_tensor_weights_npz,
    resolve_calibration_kwargs,
    save_tensor_weights_npz,
)
from .objectives import (
    FisherL2Objective,
    L2WeightObjective,
    Objective,
    UnsupportedGroupLinearization,
    available_objectives,
    get_objective,
    register_objective,
)
from .permutation_state import (
    PermutationState,
    as_permutation_matrix,
    sinkhorn_operator,
    solve_lap_maximize,
)
from .scheduler import SolverScheduler
from .solvers import (
    LAPGroupSolver,
    ProcrustesGroupSolver,
    SinkhornBlockSolver,
    SolverScheduleStep,
    available_solvers,
    solve_orthogonal_maximize,
    solve_rotation_pairs_maximize,
    solve_signed_lap_maximize,
    update_group_transform,
)

__all__ = [
    "FisherL2Objective",
    "L2WeightObjective",
    "LAPGroupSolver",
    "Objective",
    "PermutationState",
    "ProcrustesGroupSolver",
    "SinkhornBlockSolver",
    "SolverScheduleStep",
    "SolverScheduler",
    "UnsupportedGroupLinearization",
    "as_permutation_matrix",
    "available_objectives",
    "available_solvers",
    "build_scheduler",
    "default_lap_schedule",
    "estimate_diag_fisher_tree",
    "estimate_diag_fisher_weights",
    "get_objective",
    "load_tensor_weights_npz",
    "rebasin_batch",
    "rebasin_block_across",
    "rebasin_single_sample",
    "register_objective",
    "resolve_calibration_kwargs",
    "save_tensor_weights_npz",
    "sinkhorn_operator",
    "solve_lap_maximize",
    "solve_orthogonal_maximize",
    "solve_rotation_pairs_maximize",
    "solve_signed_lap_maximize",
    "update_group_transform",
]
