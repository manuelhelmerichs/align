"""Tests for head-structured (selector) bindings and attention updates."""

import copy

import numpy as np
import pytest

from align.matching import (
    DiagonalFisherObjective,
    EuclideanObjective,
    TransformState,
    UnsupportedGroupLinearization,
    build_solver_sequence,
    match_sample,
)
from align.matching.attention import (
    _quotient_circuit_precision,
    _weighted_circuit_cost,
    attention_module_specs,
    head_cost_matrix,
    update_attention_module,
)
from align.symmetry import (
    AxisBinding,
    ComponentSpec,
    MHACircuitConstraint,
    SymmetryGraph,
    SymmetryGroup,
    TensorSpec,
    materialize_many,
)

D_MODEL = 6
NUM_HEADS = 3
HEAD_DIM = 4


def _perm_matrix(indices) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    mat = np.zeros((indices.shape[0], indices.shape[0]), dtype=np.float64)
    mat[np.arange(indices.shape[0]), indices] = 1.0
    return mat


def _attention_graph() -> tuple[SymmetryGraph, dict]:
    rng = np.random.default_rng(0)
    params = {
        "attn": {
            "query": {
                "kernel": rng.normal(size=(D_MODEL, NUM_HEADS, HEAD_DIM)).astype(
                    np.float32
                ),
                "bias": rng.normal(size=(NUM_HEADS, HEAD_DIM)).astype(np.float32),
            },
            "key": {
                "kernel": rng.normal(size=(D_MODEL, NUM_HEADS, HEAD_DIM)).astype(
                    np.float32
                ),
                "bias": rng.normal(size=(NUM_HEADS, HEAD_DIM)).astype(np.float32),
            },
            "value": {
                "kernel": rng.normal(size=(D_MODEL, NUM_HEADS, HEAD_DIM)).astype(
                    np.float32
                ),
                "bias": rng.normal(size=(NUM_HEADS, HEAD_DIM)).astype(np.float32),
            },
            "out": {
                "kernel": rng.normal(size=(NUM_HEADS, HEAD_DIM, D_MODEL)).astype(
                    np.float32
                ),
                "bias": rng.normal(size=(D_MODEL,)).astype(np.float32),
            },
        }
    }

    groups = {
        "stream": SymmetryGroup(id="stream", size=D_MODEL),
        "heads": SymmetryGroup(id="heads", size=NUM_HEADS),
    }
    for slot in range(NUM_HEADS):
        groups[f"qk{slot}"] = SymmetryGroup(id=f"qk{slot}", size=HEAD_DIM)
        groups[f"vo{slot}"] = SymmetryGroup(id=f"vo{slot}", size=HEAD_DIM)

    def _tensor(*path: str) -> TensorSpec:
        tensor_id = "/".join(path)
        value = params
        for key in path:
            value = value[key]
        return TensorSpec(id=tensor_id, path=tuple(path), shape=tuple(np.shape(value)))

    tensors = {
        spec.id: spec
        for spec in (
            _tensor("attn", "query", "kernel"),
            _tensor("attn", "query", "bias"),
            _tensor("attn", "key", "kernel"),
            _tensor("attn", "key", "bias"),
            _tensor("attn", "value", "kernel"),
            _tensor("attn", "value", "bias"),
            _tensor("attn", "out", "kernel"),
            _tensor("attn", "out", "bias"),
        )
    }

    bindings: list[AxisBinding] = []
    for name, intra in (("query", "qk"), ("key", "qk"), ("value", "vo")):
        kernel_id = f"attn/{name}/kernel"
        bias_id = f"attn/{name}/bias"
        bindings.append(
            AxisBinding(tensor_id=kernel_id, axis=0, group="stream", role="in")
        )
        bindings.append(AxisBinding(tensor_id=kernel_id, axis=1, group="heads"))
        bindings.append(AxisBinding(tensor_id=bias_id, axis=0, group="heads"))
        for slot in range(NUM_HEADS):
            bindings.append(
                AxisBinding(
                    tensor_id=kernel_id,
                    axis=2,
                    group=f"{intra}{slot}",
                    selector=((1, slot),),
                )
            )
            bindings.append(
                AxisBinding(
                    tensor_id=bias_id,
                    axis=1,
                    group=f"{intra}{slot}",
                    selector=((0, slot),),
                )
            )
    bindings.append(AxisBinding(tensor_id="attn/out/kernel", axis=0, group="heads"))
    for slot in range(NUM_HEADS):
        bindings.append(
            AxisBinding(
                tensor_id="attn/out/kernel",
                axis=1,
                group=f"vo{slot}",
                selector=((0, slot),),
            )
        )
    bindings.append(
        AxisBinding(tensor_id="attn/out/kernel", axis=2, group="stream", role="out")
    )
    bindings.append(
        AxisBinding(tensor_id="attn/out/bias", axis=0, group="stream", role="out")
    )

    constraint = MHACircuitConstraint(
        head_group="heads",
        qk_groups=tuple(f"qk{slot}" for slot in range(NUM_HEADS)),
        vo_groups=tuple(f"vo{slot}" for slot in range(NUM_HEADS)),
        query="attn/query/kernel",
        key="attn/key/kernel",
        value="attn/value/kernel",
        out="attn/out/kernel",
    )

    group_order = ["stream", "heads"]
    for slot in range(NUM_HEADS):
        group_order.extend([f"qk{slot}", f"vo{slot}"])

    graph = SymmetryGraph(
        groups=groups,
        tensors=tensors,
        axis_bindings=tuple(bindings),
        constraints=(constraint,),
        components={
            "stream": ComponentSpec(
                id="stream", kind="residual_stream", groups=("stream",)
            ),
            "attn": ComponentSpec(
                id="attn",
                kind="mha",
                groups=tuple(gid for gid in group_order if gid != "stream"),
            ),
        },
        metadata={"architecture": "toy_attention", "group_order": group_order},
    )
    graph.validate(params)
    return graph, params


def _random_state(graph, seed: int, *, identity_stream: bool = False):
    rng = np.random.default_rng(seed)
    perms = {}
    for group_id, group in graph.groups.items():
        if identity_stream and group_id == "stream":
            continue
        perms[group_id] = _perm_matrix(rng.permutation(group.size))
    return TransformState.from_transforms(graph, perms)


def _mha_apply(params, x: np.ndarray) -> np.ndarray:
    """Multi-head attention forward with flax-style head-structured kernels."""

    attn = params["attn"]
    q = np.einsum("td,dhk->thk", x, attn["query"]["kernel"]) + attn["query"]["bias"]
    k = np.einsum("td,dhk->thk", x, attn["key"]["kernel"]) + attn["key"]["bias"]
    v = np.einsum("td,dhk->thk", x, attn["value"]["kernel"]) + attn["value"]["bias"]
    scores = np.einsum("thk,shk->hts", q, k) / np.sqrt(HEAD_DIM)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    context = np.einsum("hts,shk->thk", weights, v)
    return (
        np.einsum("thk,hkd->td", context, attn["out"]["kernel"]) + attn["out"]["bias"]
    )


class TestSelectorActions:
    def test_hard_apply_matches_manual_composition(self):
        graph, params = _attention_graph()
        head_perm = np.array([2, 0, 1])
        intra_perms = {
            f"qk{slot}": np.roll(np.arange(HEAD_DIM), slot + 1)
            for slot in range(NUM_HEADS)
        }
        state = TransformState.from_transforms(
            graph,
            {
                "heads": _perm_matrix(head_perm),
                **{gid: _perm_matrix(idx) for gid, idx in intra_perms.items()},
            },
        )
        transformed = graph.apply_transforms(params, state)

        kernel = np.asarray(params["attn"]["query"]["kernel"])
        expected = kernel[:, head_perm, :]
        for slot in range(NUM_HEADS):
            expected[:, slot, :] = expected[:, slot, intra_perms[f"qk{slot}"]]
        np.testing.assert_allclose(
            np.asarray(transformed["attn"]["query"]["kernel"]), expected
        )

        bias = np.asarray(params["attn"]["query"]["bias"])
        expected_bias = bias[head_perm, :]
        for slot in range(NUM_HEADS):
            expected_bias[slot, :] = expected_bias[slot, intra_perms[f"qk{slot}"]]
        np.testing.assert_allclose(
            np.asarray(transformed["attn"]["query"]["bias"]), expected_bias
        )

    def test_attention_action_preserves_mha_function(self):
        graph, params = _attention_graph()
        x = np.random.default_rng(5).normal(size=(7, D_MODEL)).astype(np.float32)
        baseline = _mha_apply(params, x)
        for seed in range(4):
            state = _random_state(graph, seed, identity_stream=True)
            transformed = graph.apply_transforms(params, state)
            np.testing.assert_allclose(_mha_apply(transformed, x), baseline, atol=1e-5)

    def test_value_zero_on_exact_orbit(self):
        graph, params = _attention_graph()
        state = _random_state(graph, seed=3)
        permuted = graph.apply_transforms(params, state)
        objective = EuclideanObjective()
        # value applies ``state`` to the target tree, so aligning ``params``
        # onto the permuted reference with the same state must give zero.
        reference_data = graph.materialize(permuted, backend="jax")
        target_data = graph.materialize(params, backend="jax")
        value = float(objective.value(graph, reference_data, target_data, state))
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_batched_value_matches_single(self):
        graph, params = _attention_graph()
        state = _random_state(graph, seed=4)
        rng = np.random.default_rng(11)
        other = {
            "attn": {
                module: {
                    name: np.asarray(arr)
                    + rng.normal(size=np.shape(arr)).astype(np.float32) * 0.1
                    for name, arr in tensors.items()
                }
                for module, tensors in params["attn"].items()
            }
        }
        objective = EuclideanObjective()
        reference_data = graph.materialize(params, backend="jax")
        batched = materialize_many(graph, [params, other], backend="jax")
        values = np.asarray(
            objective.value(graph, reference_data, batched, state)
        ).ravel()
        singles = [
            float(
                objective.value(
                    graph,
                    reference_data,
                    graph.materialize(tree, backend="jax"),
                    state,
                )
            )
            for tree in (params, other)
        ]
        np.testing.assert_allclose(values, singles, rtol=1e-5)


class TestSelectorValidation:
    def test_selector_on_bound_axis_rejected(self):
        graph, params = _attention_graph()
        bad = AxisBinding(
            tensor_id="attn/query/kernel",
            axis=2,
            group="qk0",
            selector=((2, 0),),
        )
        clone = SymmetryGraph(
            groups=graph.groups,
            tensors=graph.tensors,
            axis_bindings=(*graph.axis_bindings, bad),
            metadata={"group_order": list(graph.group_order)},
        )
        with pytest.raises(ValueError, match="selector axis"):
            clone.validate()

    def test_conflicting_selector_binding_rejected(self):
        graph, params = _attention_graph()
        bad = AxisBinding(
            tensor_id="attn/query/kernel",
            axis=2,
            group="qk1",
            selector=((1, 0),),  # slot 0 already bound to qk0
        )
        clone = SymmetryGraph(
            groups=graph.groups,
            tensors=graph.tensors,
            axis_bindings=(*graph.axis_bindings, bad),
            metadata={"group_order": list(graph.group_order)},
        )
        with pytest.raises(ValueError, match="overlaps"):
            clone.validate()

    def test_selector_index_out_of_range_rejected(self):
        graph, _ = _attention_graph()
        bad = AxisBinding(
            tensor_id="attn/query/kernel",
            axis=2,
            group="qk0",
            selector=((1, NUM_HEADS),),
        )
        clone = SymmetryGraph(
            groups=graph.groups,
            tensors=graph.tensors,
            axis_bindings=(*graph.axis_bindings, bad),
            metadata={"group_order": list(graph.group_order)},
        )
        with pytest.raises(ValueError, match="out of range"):
            clone.validate()


class TestAttentionUpdates:
    def test_diagonal_quotient_circuit_precision_matches_formula(self):
        rng = np.random.default_rng(17)
        left = rng.normal(size=(3, 4))
        right = rng.normal(size=(2, 4))
        left_fisher = rng.uniform(0.2, 3.0, size=left.shape)
        right_fisher = rng.uniform(0.2, 3.0, size=right.shape)
        precision = _quotient_circuit_precision(left, right, left_fisher, right_fisher)
        expected = np.empty((3, 2))
        for a in range(3):
            for b in range(2):
                expected[a, b] = 1.0 / np.sum(
                    right[b] ** 2 / left_fisher[a] + left[a] ** 2 / right_fisher[b]
                )
        np.testing.assert_allclose(precision, expected)

    def test_weighted_circuit_cost_keeps_assignment_dependent_self_term(self):
        reference = np.zeros((2, 1, 1))
        target = np.array([[[1.0]], [[3.0]]])
        precision = np.array([[[1.0]], [[10.0]]])
        cost = _weighted_circuit_cost(reference, target, precision)
        np.testing.assert_array_equal(cost, np.array([[1.0, 9.0], [10.0, 90.0]]))

    def test_head_group_generic_linearization_refused(self):
        graph, params = _attention_graph()
        objective = EuclideanObjective()
        reference_data = graph.materialize(params, backend="numpy")
        state = TransformState.identity(graph, backend="numpy")
        with pytest.raises(UnsupportedGroupLinearization, match="head group"):
            objective.linearize_group(
                graph, reference_data, reference_data, state, "heads"
            )

    def test_module_spec_from_constraint(self):
        graph, _ = _attention_graph()
        (spec,) = attention_module_specs(graph)
        assert spec.head_group == "heads"
        assert spec.num_heads == NUM_HEADS
        assert spec.head_dim == HEAD_DIM
        assert spec.query == "attn/query/kernel"

    def test_head_cost_matrix_invariant_to_intra_perms(self):
        graph, params = _attention_graph()
        (spec,) = attention_module_specs(graph)
        reference_data = graph.materialize(params, backend="numpy")
        state = TransformState.identity(graph, backend="numpy")
        base = head_cost_matrix(
            graph, EuclideanObjective(), reference_data, reference_data, state, spec
        )

        intra_only = TransformState.from_transforms(
            graph,
            {
                f"qk{slot}": _perm_matrix(
                    np.random.default_rng(slot).permutation(HEAD_DIM)
                )
                for slot in range(NUM_HEADS)
            },
        )
        target = graph.apply_transforms(params, intra_only)
        target_data = graph.materialize(target, backend="numpy")
        shuffled = head_cost_matrix(
            graph, EuclideanObjective(), reference_data, target_data, state, spec
        )
        np.testing.assert_allclose(shuffled, base, atol=1e-4)

    def test_affine_circuits_include_query_and_value_but_not_key_bias(self):
        graph, params = _attention_graph()
        (spec,) = attention_module_specs(graph)
        reference_data = graph.materialize(params, backend="numpy")
        state = TransformState.identity(graph, backend="numpy")
        base = head_cost_matrix(
            graph, EuclideanObjective(), reference_data, reference_data, state, spec
        )

        for projection in ("query", "value"):
            changed = copy.deepcopy(params)
            changed["attn"][projection]["bias"][0, 0] += 2.0
            cost = head_cost_matrix(
                graph,
                EuclideanObjective(),
                reference_data,
                graph.materialize(changed, backend="numpy"),
                state,
                spec,
            )
            assert cost[0, 0] > base[0, 0] + 1.0

        changed = copy.deepcopy(params)
        changed["attn"]["key"]["bias"] += 2.0
        key_cost = head_cost_matrix(
            graph,
            EuclideanObjective(),
            reference_data,
            graph.materialize(changed, backend="numpy"),
            state,
            spec,
        )
        np.testing.assert_allclose(key_cost, base)

    def test_fisher_weighted_head_cost_is_intra_invariant(self):
        graph, params = _attention_graph()
        (spec,) = attention_module_specs(graph)
        rng = np.random.default_rng(23)
        fisher = {
            tensor_id: rng.uniform(0.1, 4.0, size=tensor.shape)
            for tensor_id, tensor in graph.tensors.items()
        }
        objective = DiagonalFisherObjective(tensor_weights=fisher)
        reference_data = graph.materialize(params, backend="numpy")
        state = TransformState.identity(graph, backend="numpy")
        base = head_cost_matrix(
            graph, objective, reference_data, reference_data, state, spec
        )

        intra_only = TransformState.from_transforms(
            graph,
            {
                **{
                    f"qk{slot}": _perm_matrix(rng.permutation(HEAD_DIM))
                    for slot in range(NUM_HEADS)
                },
                **{
                    f"vo{slot}": _perm_matrix(rng.permutation(HEAD_DIM))
                    for slot in range(NUM_HEADS)
                },
            },
        )
        target = graph.apply_transforms(params, intra_only)
        shuffled = head_cost_matrix(
            graph,
            objective,
            reference_data,
            graph.materialize(target, backend="numpy"),
            state,
            spec,
        )
        np.testing.assert_allclose(shuffled, base, atol=1e-10)

    def test_structured_update_recovers_known_perms(self):
        graph, params = _attention_graph()
        (spec,) = attention_module_specs(graph)
        forward = _random_state(graph, seed=9, identity_stream=True)
        target = graph.apply_transforms(params, forward)

        reference_data = graph.materialize(params, backend="numpy")
        target_data = graph.materialize(target, backend="numpy")
        state = TransformState.identity(graph, backend="numpy")
        objective = EuclideanObjective()
        state, aux = update_attention_module(
            graph, objective, reference_data, target_data, state, spec
        )
        aligned = graph.apply_transforms(target, state)
        for module in ("query", "key", "value", "out"):
            for name in ("kernel", "bias"):
                np.testing.assert_allclose(
                    np.asarray(aligned["attn"][module][name]),
                    np.asarray(params["attn"][module][name]),
                    atol=1e-5,
                )

    @pytest.mark.parametrize(
        "schedule",
        [
            [{"solver": "lap", "max_sweeps": 10, "tolerance": 0.0}],
            [
                {
                    "solver": "sinkhorn",
                    "max_steps": 30,
                    "tolerance": 1e-5,
                    "init_scale": 0.0,
                    "learning_rate": 0.05,
                },
                {"solver": "lap", "max_sweeps": 10, "tolerance": 0.0},
            ],
        ],
    )
    def test_full_schedule_recovers_orbit_with_stream(self, schedule):
        graph, params = _attention_graph()
        forward = _random_state(graph, seed=13)
        target = graph.apply_transforms(params, forward)

        aligned, perms, aux = match_sample(
            graph,
            params,
            target,
            schedule=schedule,
        )
        for module in ("query", "key", "value", "out"):
            for name in ("kernel", "bias"):
                np.testing.assert_allclose(
                    np.asarray(aligned["attn"][module][name]),
                    np.asarray(params["attn"][module][name]),
                    atol=1e-4,
                )
        assert aux["objective_final"] == pytest.approx(0.0, abs=1e-6)

    def test_components_restricted_lap_touches_only_attention(self):
        graph, params = _attention_graph()
        forward = _random_state(graph, seed=17, identity_stream=True)
        target = graph.apply_transforms(params, forward)

        solver_sequence = build_solver_sequence(
            schedule=[{"solver": "lap", "max_sweeps": 5, "components": ["attn"]}]
        )
        reference_data = graph.materialize(params, backend="numpy")
        target_data = graph.materialize(target, backend="numpy")
        state, _ = solver_sequence.solve(graph, reference_data, target_data)
        np.testing.assert_array_equal(
            np.asarray(state.matrices["stream"]), np.eye(D_MODEL, dtype=np.float32)
        )
        aligned = graph.apply_transforms(target, state)
        np.testing.assert_allclose(
            np.asarray(aligned["attn"]["query"]["kernel"]),
            np.asarray(params["attn"]["query"]["kernel"]),
            atol=1e-5,
        )
