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
    rebasin_single_sample,
)
from .objectives import (
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
    SinkhornBlockSolver,
    SolverScheduleStep,
    available_solvers,
)

__all__ = [
    "L2WeightObjective",
    "LAPGroupSolver",
    "Objective",
    "PermutationState",
    "SinkhornBlockSolver",
    "SolverScheduleStep",
    "SolverScheduler",
    "UnsupportedGroupLinearization",
    "as_permutation_matrix",
    "available_objectives",
    "available_solvers",
    "build_scheduler",
    "default_lap_schedule",
    "get_objective",
    "rebasin_batch",
    "rebasin_single_sample",
    "register_objective",
    "sinkhorn_operator",
    "solve_lap_maximize",
]
