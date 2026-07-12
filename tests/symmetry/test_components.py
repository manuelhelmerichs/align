"""Tests for component-structured symmetry graphs."""

import dataclasses

import numpy as np
import pytest

from align.architectures.mlp import MLPRecipe
from align.matching import match_component_across
from align.matching.solvers import SolverStep, _scheduled_groups
from align.symmetry import (
    AxisBinding,
    ComponentSpec,
    SymmetryGraph,
    SymmetryGroup,
    TensorSpec,
    component_signature,
    describe_symmetry,
    extract_component_graph,
    format_symmetry_description,
    groups_for_components,
    match_component_tensors,
    resolve_component_patterns,
)


def _mlp_graph():
    rng = np.random.default_rng(0)
    fcn = {}
    for idx, (din, dout) in enumerate(((3, 5), (5, 4), (4, 2))):
        fcn[f"dense{idx}"] = {
            "kernel": rng.normal(size=(din, dout)).astype(np.float32),
            "bias": rng.normal(size=(dout,)).astype(np.float32),
        }
    params = {"params": {"fcn": fcn}}
    return MLPRecipe().build_graph(params), params


def _ffn_net(*, root: str, hidden_group: str, seed: int, with_other: bool):
    """Hand-built net with one FFN-style component plus optional unrelated component."""

    rng = np.random.default_rng(seed)
    params = {
        root: {
            "l0": {
                "kernel": rng.normal(size=(4, 6)).astype(np.float32),
                "bias": rng.normal(size=(6,)).astype(np.float32),
            },
            "l1": {"kernel": rng.normal(size=(6, 3)).astype(np.float32)},
        }
    }
    groups = {hidden_group: SymmetryGroup(id=hidden_group, size=6)}
    tensors = {
        f"{root}/l0/kernel": TensorSpec(
            id=f"{root}/l0/kernel", path=(root, "l0", "kernel"), shape=(4, 6)
        ),
        f"{root}/l0/bias": TensorSpec(
            id=f"{root}/l0/bias", path=(root, "l0", "bias"), shape=(6,)
        ),
        f"{root}/l1/kernel": TensorSpec(
            id=f"{root}/l1/kernel", path=(root, "l1", "kernel"), shape=(6, 3)
        ),
    }
    bindings = [
        AxisBinding(
            tensor_id=f"{root}/l0/kernel", axis=1, group=hidden_group, role="out"
        ),
        AxisBinding(
            tensor_id=f"{root}/l0/bias", axis=0, group=hidden_group, role="out"
        ),
        AxisBinding(
            tensor_id=f"{root}/l1/kernel", axis=0, group=hidden_group, role="in"
        ),
    ]
    components = {
        "ffn": ComponentSpec(id="ffn", kind="dense_chain", groups=(hidden_group,))
    }
    group_order = [hidden_group]
    if with_other:
        params[root]["extra"] = {"kernel": rng.normal(size=(5, 3)).astype(np.float32)}
        other_group = f"{hidden_group}_other"
        groups[other_group] = SymmetryGroup(id=other_group, size=3)
        tensors[f"{root}/extra/kernel"] = TensorSpec(
            id=f"{root}/extra/kernel", path=(root, "extra", "kernel"), shape=(5, 3)
        )
        bindings.append(
            AxisBinding(
                tensor_id=f"{root}/extra/kernel",
                axis=1,
                group=other_group,
                role="out",
            )
        )
        components["other"] = ComponentSpec(
            id="other", kind="dense_chain", groups=(other_group,)
        )
        group_order.append(other_group)

    graph = SymmetryGraph(
        groups=groups,
        tensors=tensors,
        axis_bindings=tuple(bindings),
        components=components,
        metadata={"architecture": "hand_built", "group_order": group_order},
    )
    graph.validate(params)
    return graph, params


class TestComponentValidation:
    def _base(self):
        return {
            "groups": {"g0": SymmetryGroup(id="g0", size=2)},
            "tensors": {
                "t": TensorSpec(id="t", path=("t",), shape=(2,)),
            },
            "axis_bindings": (AxisBinding(tensor_id="t", axis=0, group="g0"),),
        }

    def test_unknown_group_rejected(self):
        graph = SymmetryGraph(
            **self._base(),
            components={"b": ComponentSpec(id="b", kind="x", groups=("missing",))},
        )
        with pytest.raises(ValueError, match="unknown group"):
            graph.validate()

    def test_duplicate_ownership_rejected(self):
        graph = SymmetryGraph(
            **self._base(),
            components={
                "a": ComponentSpec(id="a", kind="x", groups=("g0",)),
                "b": ComponentSpec(id="b", kind="x", groups=("g0",)),
            },
        )
        with pytest.raises(ValueError, match="partition"):
            graph.validate()

    def test_uncovered_group_rejected(self):
        base = self._base()
        base["groups"]["g1"] = SymmetryGroup(id="g1", size=2)
        graph = SymmetryGraph(
            **base,
            components={"a": ComponentSpec(id="a", kind="x", groups=("g0",))},
        )
        with pytest.raises(ValueError, match="cover every group"):
            graph.validate()

    def test_empty_component_rejected(self):
        graph = SymmetryGraph(
            **self._base(),
            components={
                "a": ComponentSpec(id="a", kind="x", groups=("g0",)),
                "b": ComponentSpec(id="b", kind="x", groups=()),
            },
        )
        with pytest.raises(ValueError, match="declares no groups"):
            graph.validate()

    def test_id_mismatch_rejected(self):
        graph = SymmetryGraph(
            **self._base(),
            components={"a": ComponentSpec(id="b", kind="x", groups=("g0",))},
        )
        with pytest.raises(ValueError, match="does not match id"):
            graph.validate()


class TestComponentListing:
    def test_graph_public_mappings_and_nested_metadata_are_immutable(self):
        graph, _ = _mlp_graph()
        with pytest.raises(TypeError):
            graph.groups["new"] = SymmetryGroup(id="new", size=1)
        with pytest.raises(TypeError):
            graph.metadata["group_order"] = ()
        with pytest.raises(TypeError):
            graph.metadata["layer_paths"][0] = ("changed",)

    def test_mlp_emits_semantic_component(self):
        graph, _ = _mlp_graph()
        assert set(graph.components) == {"mlp"}
        assert graph.components["mlp"].kind == "dense_chain"
        assert graph.components["mlp"].groups == ("mlp/h0", "mlp/h1")
        assert graph.component_for_group("mlp/h0") == "mlp"

    def test_describe_symmetry_reports_components(self):
        graph, _ = _mlp_graph()
        description = describe_symmetry(graph)
        assert description["architecture"] == "mlp"
        (component,) = description["components"]
        assert component["id"] == "mlp"
        assert [group["size"] for group in component["groups"]] == [5, 4]
        assert component["num_tensors"] == 5
        assert component["notes"] == []

    def test_describe_symmetry_without_components_uses_implicit_all(self):
        graph, _ = _mlp_graph()
        graph = dataclasses.replace(graph, components={})
        (component,) = describe_symmetry(graph)["components"]
        assert component["id"] == "all"
        assert len(component["groups"]) == 2

    def test_format_symmetry_description_mentions_components(self):
        graph, _ = _mlp_graph()
        text = format_symmetry_description(graph)
        assert "mlp [dense_chain]" in text
        assert "mlp/h0(5)" in text


class TestComponentSelection:
    def test_resolve_patterns_fnmatch(self):
        graph, _ = _ffn_net(root="net", hidden_group="h", seed=0, with_other=True)
        assert resolve_component_patterns(graph, ["ffn"]) == ("ffn",)
        assert resolve_component_patterns(graph, ["*"]) == ("ffn", "other")

    def test_resolve_patterns_unmatched_raises(self):
        graph, _ = _ffn_net(root="net", hidden_group="h", seed=0, with_other=False)
        with pytest.raises(ValueError, match="matches no components"):
            resolve_component_patterns(graph, ["missing*"])

    def test_resolve_patterns_requires_components(self):
        graph, _ = _mlp_graph()
        graph = dataclasses.replace(graph, components={})
        with pytest.raises(ValueError, match="declares no components"):
            resolve_component_patterns(graph, ["fcn"])

    def test_groups_for_components_in_group_order(self):
        graph, _ = _ffn_net(root="net", hidden_group="h", seed=0, with_other=True)
        assert groups_for_components(graph, ["other", "ffn"]) == ("h", "h_other")

    def test_scheduled_groups_resolves_components(self):
        graph, _ = _ffn_net(root="net", hidden_group="h", seed=0, with_other=True)
        step = SolverStep(solver="lap", components=("ffn",))
        assert _scheduled_groups(graph, step) == ("h",)

    def test_step_rejects_groups_and_components(self):
        with pytest.raises(ValueError, match="not both"):
            SolverStep(solver="lap", groups=("g",), components=("b",))


class TestComponentExtraction:
    def test_extract_keeps_only_selected_component(self):
        graph, params = _ffn_net(root="net", hidden_group="h", seed=0, with_other=True)
        sub = extract_component_graph(graph, ["ffn"])
        assert set(sub.groups) == {"h"}
        assert set(sub.components) == {"ffn"}
        assert set(sub.tensors) == {
            "net/l0/kernel",
            "net/l0/bias",
            "net/l1/kernel",
        }
        sub.validate(params)

    def test_extract_unknown_component_raises(self):
        graph, _ = _ffn_net(root="net", hidden_group="h", seed=0, with_other=False)
        with pytest.raises(ValueError, match="Unknown component"):
            extract_component_graph(graph, ["nope"])


class TestCrossNetworkComponents:
    def test_signatures_match_across_different_networks(self):
        graph_a, _ = _ffn_net(
            root="net_a", hidden_group="A_h", seed=1, with_other=False
        )
        graph_b, _ = _ffn_net(root="net_b", hidden_group="B_h", seed=2, with_other=True)
        assert component_signature(graph_a, "ffn") == component_signature(
            graph_b, "ffn"
        )

    def test_match_component_tensors_maps_by_structure(self):
        graph_a, _ = _ffn_net(
            root="net_a", hidden_group="A_h", seed=1, with_other=False
        )
        graph_b, _ = _ffn_net(root="net_b", hidden_group="B_h", seed=2, with_other=True)
        sub_a = extract_component_graph(graph_a, ["ffn"])
        sub_b = extract_component_graph(graph_b, ["ffn"])
        tensor_map, group_map = match_component_tensors(sub_a, "ffn", sub_b, "ffn")
        assert tensor_map == {
            "net_a/l0/kernel": "net_b/l0/kernel",
            "net_a/l0/bias": "net_b/l0/bias",
            "net_a/l1/kernel": "net_b/l1/kernel",
        }
        assert group_map == {"A_h": "B_h"}

    def test_incompatible_signature_raises(self):
        graph_a, _ = _ffn_net(
            root="net_a", hidden_group="A_h", seed=1, with_other=False
        )
        graph_b, _ = _ffn_net(root="net_b", hidden_group="B_h", seed=2, with_other=True)
        with pytest.raises(ValueError, match="incompatible"):
            match_component_tensors(graph_a, "ffn", graph_b, "other")

    def test_match_component_across_recovers_permutation(self):
        graph_a, params_a = _ffn_net(
            root="net_a", hidden_group="A_h", seed=1, with_other=False
        )
        graph_b, params_b = _ffn_net(
            root="net_b", hidden_group="B_h", seed=2, with_other=True
        )
        # Overwrite net B's ffn with a hidden-permuted copy of net A's ffn.
        perm = np.array([3, 0, 4, 1, 5, 2])
        params_b["net_b"]["l0"]["kernel"] = params_a["net_a"]["l0"]["kernel"][:, perm]
        params_b["net_b"]["l0"]["bias"] = params_a["net_a"]["l0"]["bias"][perm]
        params_b["net_b"]["l1"]["kernel"] = params_a["net_a"]["l1"]["kernel"][perm, :]
        extra_before = np.array(params_b["net_b"]["extra"]["kernel"])

        aligned, perms, aux = match_component_across(
            graph_a,
            params_a,
            "ffn",
            graph_b,
            params_b,
            "ffn",
            schedule=[{"solver": "lap", "max_sweeps": 5}],
        )

        np.testing.assert_allclose(
            aligned["net_b"]["l0"]["kernel"],
            params_a["net_a"]["l0"]["kernel"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            aligned["net_b"]["l0"]["bias"],
            params_a["net_a"]["l0"]["bias"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            aligned["net_b"]["l1"]["kernel"],
            params_a["net_a"]["l1"]["kernel"],
            atol=1e-6,
        )
        # Tensors outside the component are untouched.
        np.testing.assert_array_equal(aligned["net_b"]["extra"]["kernel"], extra_before)
        assert set(perms) == {"B_h"}
        assert aux["group_map"] == {"A_h": "B_h"}
        assert aux["objective_final"] == pytest.approx(0.0, abs=1e-8)
