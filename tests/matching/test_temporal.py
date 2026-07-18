"""Tests for the opt-in sequence-regularized permutation gauge."""

import itertools

import numpy as np
import pytest

import align
from align.matching import (
    TemporalGaugeConfig,
    TemporalGaugeSolver,
    TransformState,
    get_objective,
    match_sample,
    match_temporal_sequence,
)
from align.symmetry import AxisBinding, SymmetryGraph, SymmetryGroup, TensorSpec


def _crossing_case(draws: int = 9):
    graph = SymmetryGraph(
        groups={"units": SymmetryGroup(id="units", size=2)},
        tensors={
            "features": TensorSpec(id="features", path=("features",), shape=(2, 1))
        },
        axis_bindings=(AxisBinding(tensor_id="features", axis=0, group="units"),),
    )
    reference = {"features": np.array([[-1.0], [1.0]], dtype=np.float32)}
    sequence = [
        {
            "features": np.array(
                [[-1.0 + 2.0 * alpha], [1.0 - 2.0 * alpha]],
                dtype=np.float32,
            )
        }
        for alpha in np.linspace(0.0, 1.0, draws)
    ]
    return graph, reference, sequence


def _permutations(transform_sequence):
    return [tuple(np.asarray(item["units"]).tolist()) for item in transform_sequence]


def test_zero_regularization_matches_independent_lap():
    graph, reference, sequence = _crossing_case()
    _, transforms, diagnostics = match_temporal_sequence(
        graph,
        reference,
        sequence,
        regularization_strength=0.0,
        strategy="greedy",
    )
    independent = [
        match_sample(graph, reference, sample)[1]["units"] for sample in sequence
    ]

    for observed, expected in zip(transforms, independent, strict=True):
        np.testing.assert_array_equal(observed["units"], expected)
    assert diagnostics["protocol"] == "temporal_gauge"
    assert diagnostics["regularization_strength"] == 0.0
    assert len(diagnostics["transform_sequence"]) == len(sequence)
    assert "assignment_ambiguity" in diagnostics


def test_temporal_api_is_available_from_package_root():
    assert align.match_temporal_sequence is match_temporal_sequence
    assert align.TemporalGaugeConfig is TemporalGaugeConfig


def test_lookahead_recovers_planted_crossing_without_greedy_hysteresis():
    graph, reference, sequence = _crossing_case()
    _, greedy, greedy_diagnostics = match_temporal_sequence(
        graph,
        reference,
        sequence,
        regularization_strength=2.0,
        strategy="greedy",
    )
    _, lookahead, lookahead_diagnostics = match_temporal_sequence(
        graph,
        reference,
        sequence,
        regularization_strength=2.0,
        strategy="lookahead",
        lookahead_window=4,
    )

    planted_switch = len(sequence) // 2
    greedy_switch = _permutations(greedy).index((1, 0))
    lookahead_switch = _permutations(lookahead).index((1, 0))

    assert greedy_switch > planted_switch
    assert lookahead_switch == planted_switch
    assert (
        lookahead_diagnostics["path_roughness"]["step_l2_mean"]
        < greedy_diagnostics["path_roughness"]["step_l2_mean"]
    )
    assert lookahead_diagnostics["transition_hamming"]["sum"] == 2


def test_lookahead_matches_brute_force_window_objective():
    graph, reference, sequence = _crossing_case(draws=5)
    strength = 1.25
    solver = TemporalGaugeSolver(
        get_objective("euclidean"),
        TemporalGaugeConfig(
            regularization_strength=strength,
            strategy="lookahead",
            lookahead_window=len(sequence),
            max_sweeps=3,
            record_ambiguity=False,
        ),
    )
    reference_data = graph.materialize(reference, backend="numpy")
    target_data = [graph.materialize(sample, backend="numpy") for sample in sequence]
    states, diagnostics = solver.solve(graph, reference_data, target_data)

    identity = TransformState.from_transforms(graph, {"units": np.array([0, 1])})
    swap = TransformState.from_transforms(graph, {"units": np.array([1, 0])})
    candidates = (identity, swap)
    objective = get_objective("euclidean")
    brute_force = np.inf
    for path in itertools.product(candidates, repeat=len(sequence)):
        alignment = sum(
            objective.value_hard_numpy(graph, reference_data, target, state)
            for target, state in zip(target_data, path, strict=True)
        )
        switches = sum(
            np.count_nonzero(
                np.asarray(left.transforms["units"].indices)
                != np.asarray(right.transforms["units"].indices)
            )
            for left, right in zip(path[:-1], path[1:], strict=True)
        )
        brute_force = min(brute_force, alignment + strength * switches)

    assert diagnostics["regularized_objective"]["sum"] == pytest.approx(brute_force)
    assert len(states) == len(sequence)


def test_temporal_gauge_rejects_non_permutation_families():
    graph = SymmetryGraph(
        groups={
            "signed": SymmetryGroup(
                id="signed", size=2, transform_family="signed_permutation"
            )
        },
        tensors={"weights": TensorSpec("weights", ("weights",), (2,))},
        axis_bindings=(AxisBinding("weights", 0, "signed"),),
    )
    params = {"weights": np.ones(2, dtype=np.float32)}

    with pytest.raises(ValueError, match="plain permutation groups only"):
        match_temporal_sequence(
            graph,
            params,
            [params, params],
            regularization_strength=1.0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"regularization_strength": -1.0},
        {"regularization_strength": np.inf},
        {"regularization_strength": 1.0, "strategy": "unknown"},
        {
            "regularization_strength": 1.0,
            "strategy": "lookahead",
            "lookahead_window": 1,
        },
        {"regularization_strength": 1.0, "max_sweeps": 0},
    ],
)
def test_temporal_config_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        TemporalGaugeConfig(**kwargs)
