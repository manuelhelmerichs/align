"""Graph-native alignment core."""

from .objectives import (
    L2WeightObjective,
    Objective,
    available_objectives,
    get_objective,
    register_objective,
)
from .problem import (
    AlignmentProblem,
    AxisBinding,
    GraphConstraint,
    TensorSpec,
    binding_axis_interval,
    materialize_many,
)
from .scale_state import ScaleState
from .scheduler import SolverScheduler
from .solvers import (
    LAPGroupSolver,
    SinkhornBlockSolver,
    SolverScheduleStep,
    UnsupportedGroupLinearization,
    available_solvers,
)
from .state import (
    PermutationState,
    as_permutation_matrix,
    sinkhorn_operator,
    solve_lap_maximize,
)

__all__ = [
    "AlignmentProblem",
    "AxisBinding",
    "GraphConstraint",
    "TensorSpec",
    "binding_axis_interval",
    "materialize_many",
    "PermutationState",
    "ScaleState",
    "as_permutation_matrix",
    "sinkhorn_operator",
    "solve_lap_maximize",
    "Objective",
    "L2WeightObjective",
    "register_objective",
    "get_objective",
    "available_objectives",
    "SolverScheduleStep",
    "SolverScheduler",
    "LAPGroupSolver",
    "SinkhornBlockSolver",
    "UnsupportedGroupLinearization",
    "available_solvers",
]
