r"""Sequence-regularized discrete gauges for temporally ordered samples.

This module is deliberately separate from ordinary population matching.  A
temporal gauge solves a different problem: every sample is still matched to a
fixed population reference, but adjacent hard transforms pay a transition
penalty.  The resulting path is suitable for studying time-series diagnostics;
it is not used by :func:`align.matching.match_sample` or
:func:`align.matching.match_batch`.

Transition metric
-----------------

The transition penalty uses the half squared chordal distance between the
orthogonal matrix realizations of two hard transforms,

.. math:: d(T, T') = \tfrac12 \lVert M(T) - M(T') \rVert_F^2 .

On permutations this is exactly the unit-level Hamming distance (each
relabeled unit costs 1).  On signed permutations it counts 1 per relabeled
unit and 2 per in-place sign flip — a sign flip moves that unit's aligned
weights to their antipode, which is a strictly larger path jump than
relabeling to an independent unit.  For isotropic unit rows ``w`` with
``E[w wᵀ] = σ² I`` the expected squared aligned-coordinate jump induced by
changing the gauge is ``2 σ² d(T, T')``, so ``λ·d`` penalizes exactly the
expected gauge-induced path-roughness increment, in the same squared-weight
units as the alignment cost.

The metric is Hamming-like (integer-valued, additive per unit, bi-invariant)
precisely for the discrete transform families.  Connected continuous families
(``rotation_pairs``, the full orthogonal update) admit no non-trivial
integer-valued invariant transition metric — any continuous invariant metric
on a connected group takes a full interval of values — so those groups are
rejected rather than silently discretized; ``orthogonal`` groups are matched
within their signed-permutation subgroup, mirroring per-draw ``lap``
schedules.  Attention-coupled circuits have a well-defined composite metric (a
moved head costs its head dimension, a stationary head pays its intra-head
signed distance) but no implemented exact coupled window solve, and are
rejected as well.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import jax
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..samples import ParamTree
from ..symmetry import SymmetryGraph
from ..symmetry.constraints import GQARoPECircuitConstraint, MHACircuitConstraint
from .objectives import Objective, get_objective
from .solvers import assignment_objective_margin
from .state import (
    PermutationTransform,
    SignedPermutationTransform,
    TransformState,
    solve_lap_maximize,
    transform_matrix,
)

TemporalStrategy = Literal["greedy", "lookahead"]

_DISCRETE_FAMILIES = ("permutation", "signed_permutation", "orthogonal")


@dataclass(frozen=True)
class TemporalGaugeConfig:
    r"""Configuration for a transition-penalized discrete gauge path.

    ``regularization_strength`` is :math:`\lambda` in squared alignment-cost
    units per unit of transition distance (half squared chordal distance;
    unit Hamming on permutations).  ``lookahead_window`` is used only by the
    ``lookahead`` strategy and includes the sample currently being committed.
    """

    regularization_strength: float
    strategy: TemporalStrategy = "greedy"
    lookahead_window: int = 5
    max_sweeps: int = 25
    tolerance: float = 0.0
    record_ambiguity: bool = True

    def __post_init__(self) -> None:
        strength = float(self.regularization_strength)
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("regularization_strength must be finite and non-negative.")
        if self.strategy not in ("greedy", "lookahead"):
            raise ValueError(
                f"Unknown temporal strategy {self.strategy!r}; expected 'greedy' "
                "or 'lookahead'."
            )
        if self.lookahead_window < 1:
            raise ValueError("lookahead_window must be positive.")
        if self.strategy == "lookahead" and self.lookahead_window < 2:
            raise ValueError("lookahead_window must be at least 2 for lookahead.")
        if self.max_sweeps < 1:
            raise ValueError("max_sweeps must be positive.")
        if not np.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")


def transform_transition_distance(
    graph: SymmetryGraph, left: TransformState, right: TransformState
) -> float:
    """Half squared chordal distance ``½‖M(T)−M(T′)‖²_F`` summed over groups.

    Equals unit Hamming distance on permutation groups; counts 1 per
    relabeled unit and 2 per in-place sign flip on signed-permutation
    groups; is continuous-valued on matrix states (diagnostic use only — the
    exact temporal solver accepts only discrete assignments, including the
    signed subgroup selected by LAP for an ``orthogonal`` group).
    """

    if any(
        isinstance(constraint, (MHACircuitConstraint, GQARoPECircuitConstraint))
        for constraint in graph.constraints
    ):
        raise ValueError(
            "Attention-coupled transition distance requires the composite "
            "head/intra-head metric; refusing to silently sum independent "
            "unit distances."
        )

    total = 0.0
    for group_id in graph.group_order:
        left_transform = left.transforms[group_id]
        right_transform = right.transforms[group_id]
        if isinstance(left_transform, PermutationTransform) and isinstance(
            right_transform, PermutationTransform
        ):
            total += float(
                np.count_nonzero(
                    np.asarray(left_transform.indices)
                    != np.asarray(right_transform.indices)
                )
            )
        elif isinstance(left_transform, SignedPermutationTransform) and isinstance(
            right_transform, SignedPermutationTransform
        ):
            left_indices = np.asarray(left_transform.indices)
            right_indices = np.asarray(right_transform.indices)
            moved = left_indices != right_indices
            flipped = ~moved & (
                np.asarray(left_transform.signs, dtype=np.float64)
                != np.asarray(right_transform.signs, dtype=np.float64)
            )
            total += float(np.count_nonzero(moved) + 2 * np.count_nonzero(flipped))
        else:
            delta = np.asarray(
                transform_matrix(left_transform, dtype=np.float64)
            ) - np.asarray(transform_matrix(right_transform, dtype=np.float64))
            total += 0.5 * float(np.vdot(delta, delta).real)
    return total


def _group_assignment(transform: PermutationTransform | SignedPermutationTransform):
    """Return ``(indices, sign_index)`` with sign index 0 → +1, 1 → −1."""

    indices = np.asarray(transform.indices, dtype=np.int64)
    if isinstance(transform, SignedPermutationTransform):
        sign_index = (np.asarray(transform.signs, dtype=np.float64) < 0).astype(
            np.int64
        )
    else:
        sign_index = None
    return indices, sign_index


def _with_assignment(
    state: TransformState,
    group_id: str,
    indices: np.ndarray,
    sign_index: np.ndarray | None,
) -> TransformState:
    if sign_index is None:
        transform: Any = PermutationTransform(np.asarray(indices, dtype=np.int32))
    else:
        transform = SignedPermutationTransform(
            np.asarray(indices, dtype=np.int32),
            np.where(np.asarray(sign_index) == 0, 1, -1).astype(np.int8),
        )
    return state.with_transforms({group_id: transform})


def _assignment_changed(
    transform: PermutationTransform | SignedPermutationTransform,
    indices: np.ndarray,
    sign_index: np.ndarray | None,
) -> int:
    current_indices, current_signs = _group_assignment(transform)
    changed = np.asarray(indices) != current_indices
    if sign_index is not None:
        changed |= np.asarray(sign_index) != current_signs
    return int(np.count_nonzero(changed))


def _transition_bonus(indices: np.ndarray, strength: float) -> np.ndarray:
    """LAP payoff whose maximization subtracts ``strength * Hamming``.

    LAP linearizations use half the squared-cost scale: an L2 coordinate update
    minimizes ``constant - 2 * payoff``.  A Hamming penalty of ``lambda``
    therefore appears as a ``lambda / 2`` bonus for retaining an assignment.
    """

    size = int(indices.size)
    bonus = np.zeros((size, size), dtype=np.float64)
    bonus[np.arange(size), np.asarray(indices, dtype=np.int64)] = 0.5 * strength
    return bonus


def _signed_transition_bonus(
    indices: np.ndarray, sign_index: np.ndarray, strength: float
) -> np.ndarray:
    """Signed payoff adjustment realizing the chordal transition penalty.

    Up to a per-row constant, ``-(λ/2)·d`` becomes ``+λ/2`` for keeping the
    signed destination, ``-λ/2`` for keeping the destination with a flipped
    sign, and 0 for relabeling.
    """

    size = int(indices.size)
    bonus = np.zeros((size, size, 2), dtype=np.float64)
    rows = np.arange(size)
    columns = np.asarray(indices, dtype=np.int64)
    signs = np.asarray(sign_index, dtype=np.int64)
    bonus[rows, columns, signs] = 0.5 * strength
    bonus[rows, columns, 1 - signs] = -0.5 * strength
    return bonus


def _signed_payoff(signed: np.ndarray, invariant: np.ndarray) -> np.ndarray:
    """Stack the ± sign payoffs as ``(size, size, 2)`` (index 0 → +1)."""

    signed = np.asarray(signed, dtype=np.float64)
    invariant = np.asarray(invariant, dtype=np.float64)
    return np.stack([invariant + signed, invariant - signed], axis=-1)


def _solve_signed_assignment(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact signed assignment maximizing a ``(size, size, 2)`` payoff."""

    reduced = score.max(axis=-1)
    indices = solve_lap_maximize(reduced)
    rows = np.arange(score.shape[0])
    sign_index = np.argmax(score[rows, indices, :], axis=-1)
    return indices, sign_index


def _signed_assignment_margin(score: np.ndarray) -> float | None:
    """Best-vs-second-best payoff gap over all signed assignments.

    The second-best signed assignment either changes the destination
    assignment (bounded by the LAP margin of the sign-maximized scores) or
    flips a single assigned unit's sign (the cheapest per-unit sign gap).
    """

    reduced = score.max(axis=-1)
    indices = solve_lap_maximize(reduced)
    rows = np.arange(score.shape[0])
    flip = float(np.min(np.abs(score[rows, indices, 0] - score[rows, indices, 1])))
    destination = assignment_objective_margin(reduced)
    return flip if destination is None else min(destination, flip)


def _solve_window_assignment(
    scores: Sequence[np.ndarray],
    *,
    strength: float,
    previous_indices: np.ndarray | None,
) -> list[np.ndarray]:
    """Solve one permutation group's finite-window assignment path exactly.

    Binary ``x[t, i, j]`` variables encode each time-local permutation.  A
    continuous stay variable receives the positive transition bonus exactly
    when the same assignment ``i -> j`` occurs on both sides of an edge.  The
    assignment variables are integral, so this is an exact linearization of
    unit-level Hamming distance for the finite window.
    """

    if not scores:
        return []
    score_array = np.asarray(scores, dtype=np.float64)
    steps, size, width = score_array.shape
    if size != width:
        raise ValueError(f"Window LAP scores must be square, got {score_array.shape}.")

    x_count = steps * size * size
    edge_count = max(steps - 1, 0)
    y_count = edge_count * size * size
    variable_count = x_count + y_count
    objective = np.empty(variable_count, dtype=np.float64)
    objective[:x_count] = -score_array.reshape(-1)
    objective[x_count:] = -0.5 * strength

    def x_index(step: int, row: int, column: int) -> int:
        return (step * size + row) * size + column

    if previous_indices is not None:
        previous = np.asarray(previous_indices, dtype=np.int64)
        if previous.shape != (size,):
            raise ValueError(
                f"Previous permutation has shape {previous.shape}, expected {(size,)}."
            )
        for row, column in enumerate(previous):
            objective[x_index(0, row, int(column))] -= 0.5 * strength

    constraints, integrality = _window_milp_structure(steps, size)
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            "Look-ahead temporal assignment failed: "
            f"status={result.status}, message={result.message}"
        )

    assignments = result.x[:x_count].reshape(steps, size, size)
    paths = [solve_lap_maximize(assignment) for assignment in assignments]
    return paths


@lru_cache(maxsize=32)
def _window_milp_structure(
    steps: int, size: int
) -> tuple[LinearConstraint, np.ndarray]:
    """Cache the shape-only sparse constraints used by rolling windows."""

    x_count = steps * size * size
    edge_count = max(steps - 1, 0)
    variable_count = x_count + edge_count * size * size

    def x_index(step: int, row: int, column: int) -> int:
        return (step * size + row) * size + column

    def y_index(edge: int, row: int, column: int) -> int:
        return x_count + (edge * size + row) * size + column

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(entries: Sequence[tuple[int, float]], lb: float, ub: float):
        constraint = len(lower)
        for variable, coefficient in entries:
            matrix_rows.append(constraint)
            matrix_columns.append(variable)
            matrix_values.append(coefficient)
        lower.append(lb)
        upper.append(ub)

    for step in range(steps):
        for row in range(size):
            add_constraint(
                [(x_index(step, row, column), 1.0) for column in range(size)],
                1.0,
                1.0,
            )
        for column in range(size):
            add_constraint(
                [(x_index(step, row, column), 1.0) for row in range(size)],
                1.0,
                1.0,
            )

    # With a negative objective coefficient on y, these upper bounds force
    # y=min(x_left, x_right) at the optimum.  No lower envelope is needed.
    for edge in range(edge_count):
        for row in range(size):
            for column in range(size):
                stay = y_index(edge, row, column)
                add_constraint(
                    [
                        (stay, 1.0),
                        (x_index(edge, row, column), -1.0),
                    ],
                    -np.inf,
                    0.0,
                )
                add_constraint(
                    [
                        (stay, 1.0),
                        (x_index(edge + 1, row, column), -1.0),
                    ],
                    -np.inf,
                    0.0,
                )

    constraint_matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(lower), variable_count),
    ).tocsr()
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[:x_count] = 1
    return LinearConstraint(constraint_matrix, lower, upper), integrality


def _solve_signed_window_assignment(
    scores: Sequence[np.ndarray],
    *,
    strength: float,
    previous: tuple[np.ndarray, np.ndarray] | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Solve one signed group's finite-window assignment path exactly.

    Binary ``x[t, i, j, s]`` variables encode each time-local signed
    permutation.  A continuous full-stay variable earns payoff ``λ`` when the
    identical signed assignment occurs on both sides of an edge, and a
    continuous destination-stay variable pays ``λ/2`` when the destination
    (with either sign) is retained; together they realize the chordal
    transition cost 0 / 1 / 2 for stay / relabel / in-place sign flip, up to
    a constant.  The assignment variables are integral, so the linearization
    is exact for the finite window.
    """

    if not scores:
        return []
    score_array = np.asarray(scores, dtype=np.float64)
    steps, size = score_array.shape[0], score_array.shape[1]
    if score_array.shape[1:] != (size, size, 2):
        raise ValueError(
            f"Signed window scores must be (steps, n, n, 2), got {score_array.shape}."
        )

    x_count = steps * size * size * 2
    edge_count = max(steps - 1, 0)
    y_count = edge_count * size * size * 2
    a_count = edge_count * size * size
    variable_count = x_count + y_count + a_count
    objective = np.empty(variable_count, dtype=np.float64)
    objective[:x_count] = -score_array.reshape(-1)
    objective[x_count : x_count + y_count] = -strength
    objective[x_count + y_count :] = 0.5 * strength

    def x_index(step: int, row: int, column: int, sign: int) -> int:
        return ((step * size + row) * size + column) * 2 + sign

    if previous is not None:
        previous_indices, previous_signs = previous
        for row in range(size):
            column = int(previous_indices[row])
            sign = int(previous_signs[row])
            objective[x_index(0, row, column, sign)] -= 0.5 * strength
            objective[x_index(0, row, column, 1 - sign)] += 0.5 * strength

    constraints, integrality = _signed_window_milp_structure(steps, size)
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            "Signed look-ahead temporal assignment failed: "
            f"status={result.status}, message={result.message}"
        )

    assignments = result.x[:x_count].reshape(steps, size, size, 2)
    paths = []
    for assignment in assignments:
        indices = solve_lap_maximize(assignment.sum(axis=-1))
        rows = np.arange(size)
        sign_index = np.argmax(assignment[rows, indices, :], axis=-1)
        paths.append((indices, sign_index))
    return paths


@lru_cache(maxsize=32)
def _signed_window_milp_structure(
    steps: int, size: int
) -> tuple[LinearConstraint, np.ndarray]:
    """Cache the shape-only sparse constraints for signed rolling windows."""

    x_count = steps * size * size * 2
    edge_count = max(steps - 1, 0)
    y_count = edge_count * size * size * 2
    variable_count = x_count + y_count + edge_count * size * size

    def x_index(step: int, row: int, column: int, sign: int) -> int:
        return ((step * size + row) * size + column) * 2 + sign

    def y_index(edge: int, row: int, column: int, sign: int) -> int:
        return x_count + ((edge * size + row) * size + column) * 2 + sign

    def a_index(edge: int, row: int, column: int) -> int:
        return x_count + y_count + (edge * size + row) * size + column

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(entries: Sequence[tuple[int, float]], lb: float, ub: float):
        constraint = len(lower)
        for variable, coefficient in entries:
            matrix_rows.append(constraint)
            matrix_columns.append(variable)
            matrix_values.append(coefficient)
        lower.append(lb)
        upper.append(ub)

    for step in range(steps):
        for row in range(size):
            add_constraint(
                [
                    (x_index(step, row, column, sign), 1.0)
                    for column in range(size)
                    for sign in (0, 1)
                ],
                1.0,
                1.0,
            )
        for column in range(size):
            add_constraint(
                [
                    (x_index(step, row, column, sign), 1.0)
                    for row in range(size)
                    for sign in (0, 1)
                ],
                1.0,
                1.0,
            )

    # y = min(x_left, x_right) at the optimum (negative objective pushes up);
    # a = max(0, dest_left + dest_right - 1) (positive objective pushes down).
    for edge in range(edge_count):
        for row in range(size):
            for column in range(size):
                for sign in (0, 1):
                    stay = y_index(edge, row, column, sign)
                    add_constraint(
                        [(stay, 1.0), (x_index(edge, row, column, sign), -1.0)],
                        -np.inf,
                        0.0,
                    )
                    add_constraint(
                        [(stay, 1.0), (x_index(edge + 1, row, column, sign), -1.0)],
                        -np.inf,
                        0.0,
                    )
                add_constraint(
                    [
                        (a_index(edge, row, column), 1.0),
                        (x_index(edge, row, column, 0), -1.0),
                        (x_index(edge, row, column, 1), -1.0),
                        (x_index(edge + 1, row, column, 0), -1.0),
                        (x_index(edge + 1, row, column, 1), -1.0),
                    ],
                    -1.0,
                    np.inf,
                )

    constraint_matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(lower), variable_count),
    ).tocsr()
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[:x_count] = 1
    return LinearConstraint(constraint_matrix, lower, upper), integrality


def _flatten_tree(params: ParamTree) -> np.ndarray:
    return np.concatenate(
        [
            np.ravel(np.asarray(leaf, dtype=np.float64))
            for leaf in jax.tree_util.tree_leaves(params)
        ]
    )


def _path_roughness(params_sequence: Sequence[ParamTree]) -> dict[str, float]:
    if len(params_sequence) < 2:
        return {
            "step_l2_mean": 0.0,
            "step_l2_median": 0.0,
            "step_l2_rms": 0.0,
            "step_l2_max": 0.0,
            "second_difference_l2_mean": 0.0,
        }
    flattened = np.stack([_flatten_tree(params) for params in params_sequence])
    steps = np.linalg.norm(np.diff(flattened, axis=0), axis=1)
    second = (
        np.linalg.norm(np.diff(flattened, n=2, axis=0), axis=1)
        if len(params_sequence) >= 3
        else np.zeros(1, dtype=np.float64)
    )
    return {
        "step_l2_mean": float(np.mean(steps)),
        "step_l2_median": float(np.median(steps)),
        "step_l2_rms": float(np.sqrt(np.mean(np.square(steps)))),
        "step_l2_max": float(np.max(steps)),
        "second_difference_l2_mean": float(np.mean(second)),
    }


class TemporalGaugeSolver:
    """Solve an opt-in sequence-regularized discrete gauge."""

    def __init__(self, objective: Objective, config: TemporalGaugeConfig) -> None:
        self.objective = objective
        self.config = config

    @property
    def backend(self) -> str:
        return "numpy"

    def validate_graph(self, graph: SymmetryGraph) -> None:
        coupled = [
            constraint
            for constraint in graph.constraints
            if isinstance(constraint, (MHACircuitConstraint, GQARoPECircuitConstraint))
        ]
        if coupled:
            raise ValueError(
                "Temporal gauges do not support attention-coupled circuits: "
                "the composite chordal transition metric is defined (a moved "
                "head costs its head dimension; a stationary head pays its "
                "intra-head signed distance), but the exact coupled "
                "head/intra-head window solve is not implemented. Refusing to "
                "decouple the circuit silently."
            )
        unsupported = [
            group_id
            for group_id, group in graph.groups.items()
            if group.transform_family not in _DISCRETE_FAMILIES
        ]
        if unsupported:
            detail = ", ".join(
                f"{group_id} ({graph.groups[group_id].transform_family})"
                for group_id in unsupported
            )
            raise ValueError(
                "Temporal gauges support permutation, signed-permutation, and "
                "the signed subgroup of orthogonal groups; unsupported "
                f"groups: {detail}. Connected continuous transform families "
                "admit no Hamming-like "
                "(integer-valued, invariant) transition metric — their half "
                "squared chordal transition distance is continuous-valued and "
                "requires a smoothing treatment outside the exact discrete "
                "transition model."
            )
        repeated = graph.repeated_group_terms()
        if repeated:
            joined = ", ".join(sorted(repeated))
            raise ValueError(
                "Temporal gauges require LAP-linearizable groups; group(s) "
                f"{joined} appear multiple times in one tensor term (QAP)."
            )
        self.objective.validate_graph(graph)

    def _group_signed(self, graph: SymmetryGraph, group_id: str) -> bool:
        return graph.groups[group_id].transform_family in (
            "signed_permutation",
            "orthogonal",
        )

    def _group_score(
        self,
        graph: SymmetryGraph,
        reference_data: Mapping[str, Any],
        target_data: Mapping[str, Any],
        state: TransformState,
        group_id: str,
    ) -> np.ndarray:
        """Return the payoff — ``(n, n)`` plain or ``(n, n, 2)`` signed."""

        self.objective.begin_lap_execution(graph, target_data)
        try:
            signed, invariant = self.objective.linearize_group_split(
                graph, reference_data, target_data, state, group_id
            )
        finally:
            self.objective.end_lap_execution()
        if self._group_signed(graph, group_id):
            return _signed_payoff(signed, invariant)
        return np.asarray(signed, dtype=np.float64) + np.asarray(
            invariant, dtype=np.float64
        )

    def _previous_assignment(
        self, previous_state: TransformState | None, group_id: str
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        if previous_state is None:
            return None
        transform = previous_state.transforms[group_id]
        assert isinstance(transform, (PermutationTransform, SignedPermutationTransform))
        return _group_assignment(transform)

    def _solve_one(
        self,
        graph: SymmetryGraph,
        reference_data: Mapping[str, Any],
        target_data: Mapping[str, Any],
        *,
        previous_state: TransformState | None,
        initial_state: TransformState,
    ) -> tuple[TransformState, int]:
        state = initial_state
        sweeps = 0
        for _ in range(self.config.max_sweeps):
            sweeps += 1
            changed = 0
            for group_id in graph.group_order:
                score = self._group_score(
                    graph, reference_data, target_data, state, group_id
                )
                previous = self._previous_assignment(previous_state, group_id)
                if self._group_signed(graph, group_id):
                    if previous is not None:
                        score = score + _signed_transition_bonus(
                            previous[0],
                            previous[1],
                            self.config.regularization_strength,
                        )
                    indices, sign_index = _solve_signed_assignment(score)
                else:
                    if previous is not None:
                        score = score + _transition_bonus(
                            previous[0], self.config.regularization_strength
                        )
                    indices = solve_lap_maximize(score)
                    sign_index = None
                current = state.transforms[group_id]
                assert isinstance(
                    current, (PermutationTransform, SignedPermutationTransform)
                )
                changed = max(
                    changed, _assignment_changed(current, indices, sign_index)
                )
                state = _with_assignment(state, group_id, indices, sign_index)
            if changed <= self.config.tolerance:
                break
        return state, sweeps

    def _independent_state(
        self,
        graph: SymmetryGraph,
        reference_data: Mapping[str, Any],
        target_data: Mapping[str, Any],
    ) -> TransformState:
        state, _ = self._solve_one(
            graph,
            reference_data,
            target_data,
            previous_state=None,
            initial_state=TransformState.identity(graph, backend="numpy"),
        )
        return state

    def _solve_greedy(
        self,
        graph: SymmetryGraph,
        reference_data: Mapping[str, Any],
        target_data_sequence: Sequence[Mapping[str, Any]],
        *,
        initial_state: TransformState | None,
    ) -> tuple[list[TransformState], list[int]]:
        states: list[TransformState] = []
        sweeps: list[int] = []
        previous = initial_state
        for target_data in target_data_sequence:
            start = previous or TransformState.identity(graph, backend="numpy")
            state, count = self._solve_one(
                graph,
                reference_data,
                target_data,
                previous_state=previous,
                initial_state=start,
            )
            states.append(state)
            sweeps.append(count)
            previous = state
        return states, sweeps

    def _refine_window(
        self,
        graph: SymmetryGraph,
        reference_data: Mapping[str, Any],
        target_window: Sequence[Mapping[str, Any]],
        states: list[TransformState],
        *,
        previous_state: TransformState | None,
    ) -> tuple[list[TransformState], int]:
        sweeps = 0
        for _ in range(self.config.max_sweeps):
            sweeps += 1
            changed = 0
            for group_id in graph.group_order:
                scores = [
                    self._group_score(
                        graph, reference_data, target_data, state, group_id
                    )
                    for target_data, state in zip(target_window, states, strict=True)
                ]
                previous = self._previous_assignment(previous_state, group_id)
                if self._group_signed(graph, group_id):
                    assignments = _solve_signed_window_assignment(
                        scores,
                        strength=self.config.regularization_strength,
                        previous=previous,
                    )
                else:
                    assignments = [
                        (indices, None)
                        for indices in _solve_window_assignment(
                            scores,
                            strength=self.config.regularization_strength,
                            previous_indices=(
                                None if previous is None else previous[0]
                            ),
                        )
                    ]
                for index, (indices, sign_index) in enumerate(assignments):
                    current = states[index].transforms[group_id]
                    assert isinstance(
                        current, (PermutationTransform, SignedPermutationTransform)
                    )
                    changed = max(
                        changed, _assignment_changed(current, indices, sign_index)
                    )
                    states[index] = _with_assignment(
                        states[index], group_id, indices, sign_index
                    )
            if changed <= self.config.tolerance:
                break
        return states, sweeps

    def _solve_lookahead(
        self,
        graph: SymmetryGraph,
        reference_data: Mapping[str, Any],
        target_data_sequence: Sequence[Mapping[str, Any]],
        *,
        initial_state: TransformState | None,
    ) -> tuple[list[TransformState], list[int]]:
        committed: list[TransformState] = []
        sweeps: list[int] = []
        planned: list[TransformState] = []
        window = self.config.lookahead_window
        for start in range(len(target_data_sequence)):
            stop = min(len(target_data_sequence), start + window)
            needed = stop - start
            planned = planned[:needed]
            while len(planned) < needed:
                offset = len(planned)
                planned.append(
                    self._independent_state(
                        graph,
                        reference_data,
                        target_data_sequence[start + offset],
                    )
                )
            previous = committed[-1] if committed else initial_state
            planned, count = self._refine_window(
                graph,
                reference_data,
                target_data_sequence[start:stop],
                planned,
                previous_state=previous,
            )
            committed.append(planned.pop(0))
            sweeps.append(count)
        return committed, sweeps

    def _regularized_score(
        self,
        score: np.ndarray,
        group_id: str,
        signed: bool,
        neighbor_states: Sequence[TransformState],
    ) -> np.ndarray:
        regularized = np.array(score, copy=True)
        for neighbor in neighbor_states:
            transform = neighbor.transforms[group_id]
            assert isinstance(
                transform, (PermutationTransform, SignedPermutationTransform)
            )
            indices, sign_index = _group_assignment(transform)
            if signed:
                if sign_index is None:
                    sign_index = np.zeros_like(indices)
                regularized += _signed_transition_bonus(
                    indices, sign_index, self.config.regularization_strength
                )
            else:
                regularized += _transition_bonus(
                    indices, self.config.regularization_strength
                )
        return regularized

    def _ambiguity(
        self,
        graph: SymmetryGraph,
        reference_data: Mapping[str, Any],
        target_data_sequence: Sequence[Mapping[str, Any]],
        states: Sequence[TransformState],
        *,
        initial_state: TransformState | None,
    ) -> tuple[list[dict[str, dict[str, float | None]]], dict[str, float | None]]:
        records: list[dict[str, dict[str, float | None]]] = []
        base_margins: list[float] = []
        regularized_margins: list[float] = []
        for index, (target_data, state) in enumerate(
            zip(target_data_sequence, states, strict=True)
        ):
            previous = states[index - 1] if index else initial_state
            following = states[index + 1] if index + 1 < len(states) else None
            step_record: dict[str, dict[str, float | None]] = {}
            for group_id in graph.group_order:
                signed = self._group_signed(graph, group_id)
                score = self._group_score(
                    graph, reference_data, target_data, state, group_id
                )
                margin_of = (
                    _signed_assignment_margin if signed else assignment_objective_margin
                )
                base_margin = margin_of(score)
                neighbors = [
                    neighbor
                    for neighbor in (
                        previous,
                        following if self.config.strategy == "lookahead" else None,
                    )
                    if neighbor is not None
                ]
                regularized_margin = margin_of(
                    self._regularized_score(score, group_id, signed, neighbors)
                )
                if base_margin is not None:
                    base_margins.append(2.0 * base_margin)
                if regularized_margin is not None:
                    regularized_margins.append(2.0 * regularized_margin)
                step_record[group_id] = {
                    "alignment_cost_margin": (
                        None if base_margin is None else 2.0 * float(base_margin)
                    ),
                    "regularized_cost_margin": (
                        None
                        if regularized_margin is None
                        else 2.0 * float(regularized_margin)
                    ),
                }
            records.append(step_record)

        def summary(values: Sequence[float], prefix: str) -> dict[str, float | None]:
            if not values:
                return {
                    f"{prefix}_mean": None,
                    f"{prefix}_median": None,
                    f"{prefix}_min": None,
                }
            return {
                f"{prefix}_mean": float(np.mean(values)),
                f"{prefix}_median": float(np.median(values)),
                f"{prefix}_min": float(np.min(values)),
            }

        ambiguity_summary = {
            **summary(base_margins, "alignment_cost_margin"),
            **summary(regularized_margins, "regularized_cost_margin"),
        }
        return records, ambiguity_summary

    def solve(
        self,
        graph: SymmetryGraph,
        reference_data: Mapping[str, Any],
        target_data_sequence: Sequence[Mapping[str, Any]],
        *,
        initial_state: TransformState | None = None,
    ) -> tuple[list[TransformState], dict[str, Any]]:
        """Solve a temporally ordered sequence against one fixed reference."""

        self.validate_graph(graph)
        if not target_data_sequence:
            raise ValueError("Temporal matching requires at least one target sample.")
        if initial_state is not None:
            initial_state.validate(graph, hard=True)
            continuous_orthogonal = [
                group_id
                for group_id, group in graph.groups.items()
                if group.transform_family == "orthogonal"
                and not isinstance(
                    initial_state.transforms[group_id], SignedPermutationTransform
                )
            ]
            if continuous_orthogonal:
                joined = ", ".join(continuous_orthogonal)
                raise ValueError(
                    "Temporal LAP restricts orthogonal groups to their signed-"
                    "permutation subgroup; the initial state contains a "
                    f"continuous orthogonal transform for: {joined}."
                )
        self.objective.set_reference_data(reference_data)

        if self.config.strategy == "greedy":
            states, sweeps = self._solve_greedy(
                graph,
                reference_data,
                target_data_sequence,
                initial_state=initial_state,
            )
        else:
            states, sweeps = self._solve_lookahead(
                graph,
                reference_data,
                target_data_sequence,
                initial_state=initial_state,
            )

        for state in states:
            state.validate(graph, hard=True)
        alignment_costs = [
            self.objective.value_hard_numpy(graph, reference_data, target_data, state)
            for target_data, state in zip(target_data_sequence, states, strict=True)
        ]
        boundary = [initial_state, *states[:-1]]
        distances = [
            0.0 if left is None else transform_transition_distance(graph, left, right)
            for left, right in zip(boundary, states, strict=True)
        ]
        # The t=0 edge is a real transition only when an initial state was
        # explicitly supplied.  Ordinary chain solves start their switch-rate
        # denominator at t=1.
        reported_distances = distances if initial_state is not None else distances[1:]
        unit_count = sum(group.size for group in graph.groups.values())
        edge_count = len(reported_distances)
        changed_groups = 0
        group_edges = 0
        transition_pairs = (
            list(zip([initial_state, *states[:-1]], states, strict=True))
            if initial_state is not None
            else list(zip(states[:-1], states[1:], strict=True))
        )
        for left, right in transition_pairs:
            assert left is not None
            for group_id in graph.group_order:
                left_transform = left.transforms[group_id]
                right_transform = right.transforms[group_id]
                left_indices, left_signs = _group_assignment(left_transform)
                right_indices, right_signs = _group_assignment(right_transform)
                same = np.array_equal(left_indices, right_indices) and (
                    left_signs is None or np.array_equal(left_signs, right_signs)
                )
                changed_groups += int(not same)
                group_edges += 1

        ambiguity_records = None
        ambiguity_summary = None
        if self.config.record_ambiguity:
            ambiguity_records, ambiguity_summary = self._ambiguity(
                graph,
                reference_data,
                target_data_sequence,
                states,
                initial_state=initial_state,
            )

        diagnostics: dict[str, Any] = {
            "protocol": "temporal_gauge",
            "strategy": self.config.strategy,
            "regularization_strength": self.config.regularization_strength,
            "transition_metric": (
                "half_squared_chordal (unit Hamming on permutations)"
            ),
            "lookahead_window": (
                self.config.lookahead_window
                if self.config.strategy == "lookahead"
                else 1
            ),
            "max_sweeps": self.config.max_sweeps,
            "sweeps": sweeps,
            "transform_sequence": [state.to_artifacts() for state in states],
            "alignment_cost": {
                "per_sample": alignment_costs,
                "sum": float(np.sum(alignment_costs)),
                "mean": float(np.mean(alignment_costs)),
            },
            "transition_distance": {
                "per_edge": [float(value) for value in reported_distances],
                "sum": float(np.sum(reported_distances)),
                "mean": (float(np.mean(reported_distances)) if edge_count else 0.0),
                "per_unit_mean": (
                    float(np.sum(reported_distances)) / (edge_count * unit_count)
                    if edge_count
                    else 0.0
                ),
            },
            "switch_rate": {
                "step": (
                    float(np.count_nonzero(reported_distances)) / edge_count
                    if edge_count
                    else 0.0
                ),
                "group": changed_groups / group_edges if group_edges else 0.0,
            },
            "regularized_objective": {
                "sum": float(
                    np.sum(alignment_costs)
                    + self.config.regularization_strength * np.sum(reported_distances)
                )
            },
        }
        if ambiguity_records is not None:
            diagnostics["assignment_ambiguity"] = {
                "per_sample": ambiguity_records,
                "summary": ambiguity_summary,
            }
        return states, diagnostics


def match_temporal_sequence(
    graph: SymmetryGraph,
    reference_params: ParamTree,
    params_sequence: Sequence[ParamTree],
    *,
    regularization_strength: float,
    strategy: TemporalStrategy = "greedy",
    lookahead_window: int = 5,
    max_sweeps: int = 25,
    tolerance: float = 0.0,
    record_ambiguity: bool = True,
    objective: str = "euclidean",
    objective_kwargs: Mapping[str, Any] | None = None,
    solver: TemporalGaugeSolver | None = None,
    reference_data: Mapping[str, Any] | None = None,
    initial_state: TransformState | None = None,
) -> tuple[list[ParamTree], list[Mapping[str, np.ndarray]], dict[str, Any]]:
    """Match one ordered chain with a sequence-regularized temporal gauge.

    Scale canonicalization intentionally remains outside this function.  For
    positively homogeneous networks, canonicalize every draw independently
    before calling this API, then use one fixed population reference here.
    """

    if not params_sequence:
        raise ValueError("Temporal matching requires at least one target sample.")
    solver = solver or TemporalGaugeSolver(
        get_objective(objective, **dict(objective_kwargs or {})),
        TemporalGaugeConfig(
            regularization_strength=regularization_strength,
            strategy=strategy,
            lookahead_window=lookahead_window,
            max_sweeps=max_sweeps,
            tolerance=tolerance,
            record_ambiguity=record_ambiguity,
        ),
    )
    if reference_data is None:
        reference_data = graph.materialize(
            reference_params, backend="numpy", cache=True
        )
    target_data = [
        graph.materialize(params, backend="numpy", cache=True)
        for params in params_sequence
    ]
    states, diagnostics = solver.solve(
        graph,
        reference_data,
        target_data,
        initial_state=initial_state,
    )
    aligned = [
        graph.apply_transforms(params, state)
        for params, state in zip(params_sequence, states, strict=True)
    ]
    diagnostics["path_roughness"] = _path_roughness(aligned)
    diagnostics["input_path_roughness"] = _path_roughness(params_sequence)
    input_mean = diagnostics["input_path_roughness"]["step_l2_mean"]
    diagnostics["path_roughness"]["step_l2_mean_ratio_to_input"] = diagnostics[
        "path_roughness"
    ]["step_l2_mean"] / max(input_mean, 1e-30)
    artifacts = [state.to_artifacts() for state in states]
    return aligned, artifacts, diagnostics


__all__ = [
    "TemporalGaugeConfig",
    "TemporalGaugeSolver",
    "TemporalStrategy",
    "match_temporal_sequence",
    "transform_transition_distance",
]
