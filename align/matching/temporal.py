"""Sequence-regularized permutation gauges for temporally ordered samples.

This module is deliberately separate from ordinary population matching.  A
temporal gauge solves a different problem: every sample is still matched to a
fixed population reference, but adjacent hard permutations pay a unit-level
Hamming penalty.  The resulting path is suitable for studying time-series
diagnostics; it is not used by :func:`align.matching.match_sample` or
:func:`align.matching.match_batch`.
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
from .objectives import Objective, get_objective
from .solvers import assignment_objective_margin
from .state import PermutationTransform, TransformState, solve_lap_maximize

TemporalStrategy = Literal["greedy", "lookahead"]


@dataclass(frozen=True)
class TemporalGaugeConfig:
    r"""Configuration for a transition-penalized permutation path.

    ``regularization_strength`` is :math:`\lambda` in squared alignment-cost
    units per changed unit.  ``lookahead_window`` is used only by the
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


def transform_hamming_distance(
    graph: SymmetryGraph, left: TransformState, right: TransformState
) -> int:
    """Return unit-level Hamming distance between two permutation states."""

    distance = 0
    for group_id in graph.group_order:
        left_transform = left.transforms[group_id]
        right_transform = right.transforms[group_id]
        if not isinstance(left_transform, PermutationTransform) or not isinstance(
            right_transform, PermutationTransform
        ):
            raise ValueError("Temporal Hamming distance supports permutations only.")
        distance += int(
            np.count_nonzero(
                np.asarray(left_transform.indices)
                != np.asarray(right_transform.indices)
            )
        )
    return distance


def _with_permutation(
    state: TransformState, group_id: str, indices: np.ndarray
) -> TransformState:
    return state.with_transforms(
        {group_id: PermutationTransform(np.asarray(indices, dtype=np.int32))}
    )


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


def _solve_window_assignment(
    scores: Sequence[np.ndarray],
    *,
    strength: float,
    previous_indices: np.ndarray | None,
) -> list[np.ndarray]:
    """Solve one group's finite-window assignment path exactly.

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
    """Solve an opt-in sequence-regularized permutation gauge."""

    def __init__(self, objective: Objective, config: TemporalGaugeConfig) -> None:
        self.objective = objective
        self.config = config

    @property
    def backend(self) -> str:
        return "numpy"

    def validate_graph(self, graph: SymmetryGraph) -> None:
        unsupported = [
            group_id
            for group_id, group in graph.groups.items()
            if group.transform_family != "permutation"
        ]
        if unsupported:
            detail = ", ".join(
                f"{group_id} ({graph.groups[group_id].transform_family})"
                for group_id in unsupported
            )
            raise ValueError(
                "Temporal gauges currently support plain permutation groups only; "
                f"unsupported groups: {detail}."
            )
        self.objective.validate_graph(graph)

    def _group_score(
        self,
        graph: SymmetryGraph,
        reference_data: Mapping[str, Any],
        target_data: Mapping[str, Any],
        state: TransformState,
        group_id: str,
    ) -> np.ndarray:
        self.objective.begin_lap_execution(graph, target_data)
        try:
            score = self.objective.linearize_group(
                graph, reference_data, target_data, state, group_id
            )
        finally:
            self.objective.end_lap_execution()
        return np.asarray(score, dtype=np.float64)

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
                if previous_state is not None:
                    previous = previous_state.transforms[group_id]
                    assert isinstance(previous, PermutationTransform)
                    score += _transition_bonus(
                        np.asarray(previous.indices),
                        self.config.regularization_strength,
                    )
                updated = solve_lap_maximize(score)
                current = state.transforms[group_id]
                assert isinstance(current, PermutationTransform)
                changed = max(
                    changed,
                    int(
                        np.count_nonzero(
                            updated != np.asarray(current.indices, dtype=np.int32)
                        )
                    ),
                )
                state = _with_permutation(state, group_id, updated)
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
                previous_indices = None
                if previous_state is not None:
                    previous_transform = previous_state.transforms[group_id]
                    assert isinstance(previous_transform, PermutationTransform)
                    previous_indices = np.asarray(previous_transform.indices)
                assignments = _solve_window_assignment(
                    scores,
                    strength=self.config.regularization_strength,
                    previous_indices=previous_indices,
                )
                for index, assignment in enumerate(assignments):
                    current = states[index].transforms[group_id]
                    assert isinstance(current, PermutationTransform)
                    changed = max(
                        changed,
                        int(
                            np.count_nonzero(assignment != np.asarray(current.indices))
                        ),
                    )
                    states[index] = _with_permutation(
                        states[index], group_id, assignment
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
                score = self._group_score(
                    graph, reference_data, target_data, state, group_id
                )
                base_margin = assignment_objective_margin(score)
                regularized_score = np.array(score, copy=True)
                if previous is not None:
                    transform = previous.transforms[group_id]
                    assert isinstance(transform, PermutationTransform)
                    regularized_score += _transition_bonus(
                        np.asarray(transform.indices),
                        self.config.regularization_strength,
                    )
                if self.config.strategy == "lookahead" and following is not None:
                    transform = following.transforms[group_id]
                    assert isinstance(transform, PermutationTransform)
                    regularized_score += _transition_bonus(
                        np.asarray(transform.indices),
                        self.config.regularization_strength,
                    )
                regularized_margin = assignment_objective_margin(regularized_score)
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
        hamming = [
            0 if left is None else transform_hamming_distance(graph, left, right)
            for left, right in zip(boundary, states, strict=True)
        ]
        # The t=0 edge is a real transition only when an initial state was
        # explicitly supplied.  Ordinary chain solves start their switch-rate
        # denominator at t=1.
        reported_hamming = hamming if initial_state is not None else hamming[1:]
        unit_count = sum(group.size for group in graph.groups.values())
        edge_count = len(reported_hamming)
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
                assert isinstance(left_transform, PermutationTransform)
                assert isinstance(right_transform, PermutationTransform)
                changed_groups += int(
                    not np.array_equal(
                        np.asarray(left_transform.indices),
                        np.asarray(right_transform.indices),
                    )
                )
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
            "transition_distance": "unit_hamming",
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
            "transition_hamming": {
                "per_edge": reported_hamming,
                "sum": int(np.sum(reported_hamming)),
                "mean": float(np.mean(reported_hamming)) if edge_count else 0.0,
                "unit_rate": (
                    float(np.sum(reported_hamming)) / (edge_count * unit_count)
                    if edge_count
                    else 0.0
                ),
            },
            "switch_rate": {
                "step": (
                    float(np.count_nonzero(reported_hamming)) / edge_count
                    if edge_count
                    else 0.0
                ),
                "group": changed_groups / group_edges if group_edges else 0.0,
            },
            "regularized_objective": {
                "sum": float(
                    np.sum(alignment_costs)
                    + self.config.regularization_strength * np.sum(reported_hamming)
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
    "transform_hamming_distance",
]
