"""Tests for the opt-in sequence-regularized discrete gauge."""

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
    transform_transition_distance,
)
from align.matching.state import (
    MatrixTransform,
    PermutationTransform,
    SignedPermutationTransform,
    transform_matrix,
)
from align.symmetry import AxisBinding, SymmetryGraph, SymmetryGroup, TensorSpec
from align.symmetry.constraints import MHACircuitConstraint


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


def _sign_crossing_case(draws: int = 16):
    """One signed unit whose signature drifts through zero and flips."""

    graph = SymmetryGraph(
        groups={
            "stream": SymmetryGroup(
                id="stream", size=1, transform_family="signed_permutation"
            )
        },
        tensors={
            "features": TensorSpec(id="features", path=("features",), shape=(1, 2))
        },
        axis_bindings=(AxisBinding(tensor_id="features", axis=0, group="stream"),),
    )
    reference = {"features": np.array([[1.0, 1.0]], dtype=np.float32)}
    sequence = [
        {"features": np.array([[value, value]], dtype=np.float32)}
        for value in np.linspace(1.0, -1.0, draws)
    ]
    return graph, reference, sequence


def _permutations(transform_sequence):
    return [tuple(np.asarray(item["units"]).tolist()) for item in transform_sequence]


def _signed_states(graph, indices, signs):
    return TransformState.from_transforms(
        graph,
        {
            "stream": SignedPermutationTransform(
                np.asarray(indices, dtype=np.int32),
                np.asarray(signs, dtype=np.int8),
            )
        },
    )


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
    assert lookahead_diagnostics["transition_distance"]["sum"] == 2


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


def test_lookahead_recovers_planted_sign_flip_without_greedy_hysteresis():
    graph, reference, sequence = _sign_crossing_case()
    _, greedy, _ = match_temporal_sequence(
        graph,
        reference,
        sequence,
        regularization_strength=1.0,
        strategy="greedy",
    )
    _, lookahead, lookahead_diagnostics = match_temporal_sequence(
        graph,
        reference,
        sequence,
        regularization_strength=1.0,
        strategy="lookahead",
        lookahead_window=4,
    )

    def flip_time(transforms):
        # Signed artifacts stack (indices, signs); row 1 is the sign vector.
        signs = [float(np.asarray(item["stream"])[1, 0]) for item in transforms]
        return next((index for index, value in enumerate(signs) if value < 0.0), None)

    planted_flip = len(sequence) // 2
    greedy_flip = flip_time(greedy)
    lookahead_flip = flip_time(lookahead)

    assert greedy_flip is not None and greedy_flip > planted_flip
    assert lookahead_flip == planted_flip
    # One in-place sign flip has chordal transition distance 2.
    assert lookahead_diagnostics["transition_distance"]["sum"] == pytest.approx(2.0)
    assert lookahead_diagnostics["transition_distance"][
        "per_unit_mean"
    ] == pytest.approx(2.0 / (len(sequence) - 1))
    assert lookahead_diagnostics["transition_metric"].startswith("half_squared_chordal")
    assert lookahead_diagnostics["switch_rate"]["step"] == pytest.approx(
        1.0 / (len(sequence) - 1)
    )


def test_signed_lookahead_matches_brute_force_window_objective():
    graph = SymmetryGraph(
        groups={
            "stream": SymmetryGroup(
                id="stream", size=2, transform_family="signed_permutation"
            )
        },
        tensors={
            "features": TensorSpec(id="features", path=("features",), shape=(2, 3))
        },
        axis_bindings=(AxisBinding(tensor_id="features", axis=0, group="stream"),),
    )
    rng = np.random.default_rng(7)
    reference = {"features": rng.normal(size=(2, 3)).astype(np.float32)}
    sequence = [
        {"features": rng.normal(size=(2, 3)).astype(np.float32)} for _ in range(4)
    ]
    strength = 0.8
    solver = TemporalGaugeSolver(
        get_objective("euclidean"),
        TemporalGaugeConfig(
            regularization_strength=strength,
            strategy="lookahead",
            lookahead_window=len(sequence),
            max_sweeps=5,
            record_ambiguity=False,
        ),
    )
    reference_data = graph.materialize(reference, backend="numpy")
    target_data = [graph.materialize(sample, backend="numpy") for sample in sequence]
    _, diagnostics = solver.solve(graph, reference_data, target_data)

    candidates = [
        _signed_states(graph, indices, signs)
        for indices in ((0, 1), (1, 0))
        for signs in itertools.product((1, -1), repeat=2)
    ]
    objective = get_objective("euclidean")
    brute_force = np.inf
    for path in itertools.product(candidates, repeat=len(sequence)):
        alignment = sum(
            objective.value_hard_numpy(graph, reference_data, target, state)
            for target, state in zip(target_data, path, strict=True)
        )
        transition = sum(
            transform_transition_distance(graph, left, right)
            for left, right in zip(path[:-1], path[1:], strict=True)
        )
        brute_force = min(brute_force, alignment + strength * transition)

    assert diagnostics["regularized_objective"]["sum"] == pytest.approx(brute_force)


def test_transition_distance_closed_forms_match_chordal_matrices():
    graph = SymmetryGraph(
        groups={
            "perm": SymmetryGroup(id="perm", size=3),
            "stream": SymmetryGroup(
                id="stream", size=3, transform_family="signed_permutation"
            ),
        },
        tensors={
            "a": TensorSpec(id="a", path=("a",), shape=(3, 1)),
            "b": TensorSpec(id="b", path=("b",), shape=(3, 1)),
        },
        axis_bindings=(
            AxisBinding(tensor_id="a", axis=0, group="perm"),
            AxisBinding(tensor_id="b", axis=0, group="stream"),
        ),
    )
    left = TransformState.from_transforms(
        graph,
        {
            "perm": PermutationTransform(np.array([0, 1, 2], dtype=np.int32)),
            "stream": SignedPermutationTransform(
                np.array([0, 1, 2], dtype=np.int32),
                np.array([1, 1, 1], dtype=np.int8),
            ),
        },
    )
    right = TransformState.from_transforms(
        graph,
        {
            # Two relabeled units: Hamming 2.
            "perm": PermutationTransform(np.array([1, 0, 2], dtype=np.int32)),
            # One in-place sign flip (2) plus a two-unit relabel (1+1).
            "stream": SignedPermutationTransform(
                np.array([0, 2, 1], dtype=np.int32),
                np.array([-1, 1, 1], dtype=np.int8),
            ),
        },
    )
    observed = transform_transition_distance(graph, left, right)
    assert observed == pytest.approx(2.0 + 2.0 + 2.0)

    expected = 0.0
    for group_id in ("perm", "stream"):
        left_matrix = np.asarray(
            transform_matrix(left.transforms[group_id], dtype=np.float64)
        )
        right_matrix = np.asarray(
            transform_matrix(right.transforms[group_id], dtype=np.float64)
        )
        expected += 0.5 * float(np.sum(np.square(left_matrix - right_matrix)))
    assert observed == pytest.approx(expected)


def test_transition_distance_rotation_matches_chordal_formula():
    graph = SymmetryGraph(
        groups={
            "rope": SymmetryGroup(id="rope", size=2, transform_family="rotation_pairs")
        },
        tensors={"q": TensorSpec(id="q", path=("q",), shape=(2, 1))},
        axis_bindings=(AxisBinding(tensor_id="q", axis=0, group="rope"),),
    )

    def rotation(angle):
        cos, sin = np.cos(angle), np.sin(angle)
        return TransformState.from_transforms(
            graph,
            {
                "rope": MatrixTransform(
                    np.array([[cos, -sin], [sin, cos]]), family="rotation_pairs"
                )
            },
        )

    angle = 0.7
    observed = transform_transition_distance(graph, rotation(0.0), rotation(angle))
    assert observed == pytest.approx(2.0 * (1.0 - np.cos(angle)))


def test_temporal_gauge_accepts_signed_and_rejects_continuous_families():
    graph = SymmetryGraph(
        groups={
            "rope": SymmetryGroup(id="rope", size=2, transform_family="rotation_pairs")
        },
        tensors={"q": TensorSpec(id="q", path=("q",), shape=(2, 1))},
        axis_bindings=(AxisBinding(tensor_id="q", axis=0, group="rope"),),
    )
    params = {"q": np.ones((2, 1), dtype=np.float32)}

    with pytest.raises(ValueError, match="no Hamming-like"):
        match_temporal_sequence(
            graph,
            params,
            [params, params],
            regularization_strength=1.0,
        )


def test_temporal_gauge_uses_signed_subgroup_for_orthogonal_lap_groups():
    graph = SymmetryGraph(
        groups={
            "stream": SymmetryGroup(id="stream", size=1, transform_family="orthogonal")
        },
        tensors={
            "features": TensorSpec(id="features", path=("features",), shape=(1, 2))
        },
        axis_bindings=(AxisBinding(tensor_id="features", axis=0, group="stream"),),
    )
    reference = {"features": np.array([[1.0, 1.0]], dtype=np.float32)}
    sequence = [
        {"features": np.array([[value, value]], dtype=np.float32)}
        for value in (1.0, 0.5, -0.5, -1.0)
    ]

    _, artifacts, diagnostics = match_temporal_sequence(
        graph,
        reference,
        sequence,
        regularization_strength=0.5,
        strategy="lookahead",
        lookahead_window=4,
    )

    assert np.asarray(artifacts[0]["stream"])[1, 0] == 1
    assert np.asarray(artifacts[-1]["stream"])[1, 0] == -1
    assert diagnostics["transition_distance"]["sum"] == pytest.approx(2.0)

    continuous = TransformState.from_transforms(
        graph,
        {"stream": MatrixTransform(np.eye(1), family="orthogonal")},
    )
    with pytest.raises(ValueError, match="signed-permutation subgroup"):
        match_temporal_sequence(
            graph,
            reference,
            sequence,
            regularization_strength=0.5,
            initial_state=continuous,
        )


def test_temporal_gauge_rejects_attention_coupled_circuits():
    heads = 2
    graph = SymmetryGraph(
        groups={"heads": SymmetryGroup(id="heads", size=heads)},
        tensors={
            "query": TensorSpec(id="query", path=("q",), shape=(heads, heads)),
            "key": TensorSpec(id="key", path=("k",), shape=(heads, heads)),
            "value": TensorSpec(id="value", path=("v",), shape=(heads, heads)),
            "out": TensorSpec(id="out", path=("o",), shape=(heads, heads)),
        },
        axis_bindings=(AxisBinding(tensor_id="query", axis=0, group="heads"),),
        constraints=(
            MHACircuitConstraint(
                head_group="heads",
                qk_groups=(),
                vo_groups=(),
                query="query",
                key="key",
                value="value",
                out="out",
            ),
        ),
    )
    params = {
        "q": np.ones((heads, heads), dtype=np.float32),
        "k": np.ones((heads, heads), dtype=np.float32),
        "v": np.ones((heads, heads), dtype=np.float32),
        "o": np.ones((heads, heads), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="attention-coupled"):
        match_temporal_sequence(
            graph,
            params,
            [params, params],
            regularization_strength=1.0,
        )

    identity = TransformState.identity(graph, backend="numpy")
    with pytest.raises(ValueError, match="composite head/intra-head"):
        transform_transition_distance(graph, identity, identity)


def test_signed_margin_reports_sign_flip_ambiguity():
    graph, reference, sequence = _sign_crossing_case(draws=5)
    _, _, diagnostics = match_temporal_sequence(
        graph,
        reference,
        sequence,
        regularization_strength=0.0,
        strategy="greedy",
    )
    margins = [
        record["stream"]["alignment_cost_margin"]
        for record in diagnostics["assignment_ambiguity"]["per_sample"]
    ]
    # The best-vs-second-best signed assignment gap is the cost of flipping
    # the sign: |(ref - v)^2 - (ref + v)^2| = 8|v| for the two-feature unit.
    values = np.linspace(1.0, -1.0, 5)
    expected = [8.0 * abs(value) for value in values]
    assert margins == pytest.approx(expected, abs=1e-5)


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
