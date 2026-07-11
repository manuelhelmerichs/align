"""Tests for architecture recipe discovery, graph construction, and actions."""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from flax.core import frozen_dict

from align.architectures import MLPRecipe, ResidualConvNetRecipe
from align.matching import TransformState
from align.symmetry import ResidualChannelTie, apply_perm_to_axis


def test_builtin_recipe_registry_is_targeted_and_side_effect_free():
    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import sys
        from align.architectures import available_recipes, get_recipe
        from align.architectures.recipe import recipe_options

        assert "align.architectures.mlp" not in sys.modules
        assert "align.architectures.residual_convnet" not in sys.modules
        assert available_recipes() == [
            "convnet",
            "layernorm_mha_transformer",
            "mlp",
            "residual_convnet",
            "rmsnorm_gqa_rope_transformer",
        ]
        assert "align.architectures.convnet" not in sys.modules
        assert "align.architectures.mlp" not in sys.modules
        assert "align.architectures.residual_convnet" not in sys.modules
        assert "align.architectures.layernorm_mha_transformer" not in sys.modules
        assert "jax" not in sys.modules

        assert recipe_options("rmsnorm_gqa_rope_transformer") == {
            "parameter_root",
            "stream_transform_family",
        }
        assert "align.architectures.rmsnorm_gqa_rope_transformer" not in sys.modules
        assert "jax" not in sys.modules

        recipe = get_recipe("mlp")
        assert type(recipe).__name__ == "MLPRecipe"
        assert "align.architectures.mlp" in sys.modules
        assert "align.architectures.residual_convnet" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_dense_recipe_permutation_round_trip():
    params = {
        "params": {
            "fcn": {
                "Dense_0": {
                    "kernel": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    "bias": np.array([0.5, -0.5], dtype=np.float32),
                },
                "Dense_1": {
                    "kernel": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
                    "bias": np.array([0.1, -0.2], dtype=np.float32),
                },
            }
        }
    }
    recipe = MLPRecipe(parameter_root="params.fcn")
    graph = recipe.build_graph(params)
    assert set(graph.groups) == {"mlp/h0"}
    roles = {
        (binding.tensor_id, binding.axis, binding.role)
        for binding in graph.bindings_for_group("mlp/h0")
    }
    # Producer kernel column + bias divide; consumer kernel row multiplies.
    assert ("params/fcn/Dense_0/kernel", 1, "out") in roles
    assert ("params/fcn/Dense_0/bias", 0, "out") in roles
    assert ("params/fcn/Dense_1/kernel", 0, "in") in roles

    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    permuted = frozen_dict.unfreeze(
        graph.apply_transforms(
            params, TransformState.from_transforms(graph, {"mlp/h0": swap})
        )
    )

    kernel0 = permuted["params"]["fcn"]["Dense_0"]["kernel"]
    bias0 = permuted["params"]["fcn"]["Dense_0"]["bias"]
    # Columns swapped
    np.testing.assert_allclose(
        kernel0[:, 0], params["params"]["fcn"]["Dense_0"]["kernel"][:, 1]
    )
    np.testing.assert_allclose(bias0, params["params"]["fcn"]["Dense_0"]["bias"][::-1])
    # Next layer input rows swapped
    kernel1 = permuted["params"]["fcn"]["Dense_1"]["kernel"]
    np.testing.assert_allclose(
        kernel1[0], params["params"]["fcn"]["Dense_1"]["kernel"][1]
    )


def test_residual_convnet_recipe_residual_grouping_permutes_all_sites():
    params = {
        "core": {
            "Conv_0": {
                "kernel": np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),
                "bias": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            },
            "FRN_0": {
                "gamma": np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),
                "beta": np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),
                "tau": np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),
                "eps": np.ones((1,), dtype=np.float32),
            },
            "Conv_1": {
                "kernel": np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4),
                "bias": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            },
            "FRN_1": {
                "gamma": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(
                    1, 1, 1, 4
                ),
                "beta": np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape(
                    1, 1, 1, 4
                ),
                "tau": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32).reshape(
                    1, 1, 1, 4
                ),
                "eps": np.ones((1,), dtype=np.float32),
            },
            "Dense_0": {
                "kernel": np.eye(4, dtype=np.float32),
                "bias": np.zeros((1,), dtype=np.float32),
            },
        }
    }
    residual_topology = {
        "nodes": [
            {
                "name": "add0",
                "type": "add",
                "inputs": ["core/Conv_0", "core/Conv_1"],
            }
        ]
    }
    recipe = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=residual_topology
    )
    graph = recipe.build_graph(params)
    assert len(graph.groups) == 1
    group_id = next(iter(graph.groups))
    bindings = {
        (binding.tensor_id, binding.axis, binding.role)
        for binding in graph.bindings_for_group(group_id)
    }
    # Conv_0 produces the residual stream; Conv_1 consumes it on its input axis.
    assert ("core/Conv_0/kernel", -1, "out") in bindings
    assert ("core/Conv_1/kernel", -2, "in") in bindings
    # The residual tie collapses both convs onto a single shared group.
    ties = [c for c in graph.constraints if isinstance(c, ResidualChannelTie)]
    assert len(ties) == 1
    assert ties[0].groups == (group_id,)
    assert set(ties[0].members) == {"core/Conv_0", "core/Conv_1"}

    swap = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    permuted = frozen_dict.unfreeze(
        graph.apply_transforms(
            params, TransformState.from_transforms(graph, {group_id: swap})
        )
    )
    conv0 = permuted["core"]["Conv_0"]["kernel"]
    # Output channels permuted
    np.testing.assert_allclose(
        conv0[..., 0], params["core"]["Conv_0"]["kernel"][..., 3]
    )
    frn0_gamma = permuted["core"]["FRN_0"]["gamma"]
    np.testing.assert_allclose(
        frn0_gamma[0, 0, 0, 0], params["core"]["FRN_0"]["gamma"][0, 0, 0, 3]
    )
    # Conv1 input axis permuted via same group
    conv1 = permuted["core"]["Conv_1"]["kernel"]
    expected_conv1 = apply_perm_to_axis(
        apply_perm_to_axis(params["core"]["Conv_1"]["kernel"], swap, axis=-2),
        swap,
        axis=-1,
    )
    np.testing.assert_allclose(conv1, expected_conv1)
    # Dense head rows permuted
    dense = permuted["core"]["Dense_0"]["kernel"]
    expected_dense = apply_perm_to_axis(
        params["core"]["Dense_0"]["kernel"], swap, axis=0
    )
    np.testing.assert_allclose(dense, expected_dense)


def test_residual_convnet_recipe_validates_join_sizes():
    params = {
        "core": {
            "Conv_0": {
                "kernel": np.zeros((1, 1, 1, 2), dtype=np.float32),
                "bias": np.zeros((2,), dtype=np.float32),
            },
            "Conv_1": {
                "kernel": np.zeros((1, 1, 2, 3), dtype=np.float32),
                "bias": np.zeros((3,), dtype=np.float32),
            },
        }
    }
    residual_topology = {
        "nodes": [
            {"type": "add", "inputs": ["core/Conv_0", "core/Conv_1"]},
        ]
    }
    recipe = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=residual_topology
    )
    with pytest.raises(ValueError):
        recipe.build_graph(params)


def test_residual_convnet_recipe_transitive_residual_union():
    params = {
        "core": {
            "Conv_0": {
                "kernel": np.zeros((1, 1, 1, 2), dtype=np.float32),
                "bias": np.zeros((2,), dtype=np.float32),
            },
            "Conv_1": {
                "kernel": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "bias": np.zeros((2,), dtype=np.float32),
            },
            "Conv_2": {
                "kernel": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "bias": np.zeros((2,), dtype=np.float32),
            },
        }
    }
    residual_topology = {
        "nodes": [
            {"type": "add", "inputs": ["core/Conv_0", "core/Conv_1"]},
            {"type": "add", "inputs": ["core/Conv_1", "core/Conv_2"]},
        ]
    }
    recipe = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=residual_topology
    )
    graph = recipe.build_graph(params)
    assert set(graph.groups) == {"core/Conv_0"}
    assert graph.metadata["group_order"] == ["core/Conv_0"]


def test_residual_convnet_recipe_stream_token_requires_stream_id():
    params = {
        "core": {
            "Conv_0": {
                "kernel": np.zeros((1, 1, 1, 2), dtype=np.float32),
                "bias": np.zeros((2,), dtype=np.float32),
            },
            "Conv_1": {
                "kernel": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "bias": np.zeros((2,), dtype=np.float32),
            },
        }
    }
    residual_topology = {
        "nodes": [
            {"type": "add", "inputs": ["core/Conv_1", "stream"]},
        ]
    }
    recipe = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=residual_topology
    )
    with pytest.raises(ValueError):
        recipe.build_graph(params)


def test_residual_convnet_recipe_stream_token_resolves_stream_id():
    params = {
        "core": {
            "Conv_0": {
                "kernel": np.zeros((1, 1, 1, 2), dtype=np.float32),
                "bias": np.zeros((2,), dtype=np.float32),
            },
            "Conv_1": {
                "kernel": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "bias": np.zeros((2,), dtype=np.float32),
            },
        }
    }
    residual_topology = {
        "nodes": [
            {
                "type": "add",
                "inputs": ["core/Conv_1", "stream"],
                "stream_id": "core/Conv_0",
            }
        ]
    }
    recipe = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=residual_topology
    )
    graph = recipe.build_graph(params)
    assert set(graph.groups) == {"core/Conv_0"}


def test_residual_convnet_recipe_recursive_conv_discovery():
    params = {
        "core": {
            "Block_0": {
                "Conv_0": {
                    "kernel": np.zeros((1, 1, 1, 2), dtype=np.float32),
                    "bias": np.zeros((2,), dtype=np.float32),
                },
                "Conv_1": {
                    "kernel": np.zeros((1, 1, 2, 2), dtype=np.float32),
                    "bias": np.zeros((2,), dtype=np.float32),
                },
            },
            "Conv_2": {
                "kernel": np.zeros((1, 1, 2, 2), dtype=np.float32),
                "bias": np.zeros((2,), dtype=np.float32),
            },
        }
    }
    residual_topology = {
        "nodes": [
            {
                "type": "add",
                "inputs": ["core/Block_0/Conv_1", "core/Conv_2"],
            }
        ]
    }
    recipe = ResidualConvNetRecipe(
        parameter_root="core", residual_topology=residual_topology
    )
    graph = recipe.build_graph(params)
    assert "core/Block_0/Conv_0" in graph.metadata["conv_names"]


def test_residual_convnet_recipe_batchnorm_permutation():
    params = {
        "params": {
            "core": {
                "Conv_0": {
                    "kernel": np.arange(3, dtype=np.float32).reshape(1, 1, 1, 3),
                    "bias": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                },
                "BatchNorm_0": {
                    "scale": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    "bias": np.array([-1.0, -2.0, -3.0], dtype=np.float32),
                },
                "Dense_0": {
                    "kernel": np.eye(3, dtype=np.float32),
                    "bias": np.zeros((1,), dtype=np.float32),
                },
            }
        },
        "batch_stats": {
            "core": {
                "BatchNorm_0": {
                    "mean": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "var": np.array([0.4, 0.5, 0.6], dtype=np.float32),
                }
            }
        },
    }
    recipe = ResidualConvNetRecipe(
        parameter_root="params.core", batch_stats_root="batch_stats.core"
    )
    graph = recipe.build_graph(params)
    group_bindings = {
        (binding.tensor_id, binding.role)
        for binding in graph.bindings_for_group("core/Conv_0")
    }
    # BN scale/bias and running stats permute with the conv output channels.
    assert ("params/core/BatchNorm_0/scale", "out") in group_bindings
    assert ("batch_stats/core/BatchNorm_0/mean", "out") in group_bindings
    assert ("batch_stats/core/BatchNorm_0/var", "out") in group_bindings
    swap = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    permuted = frozen_dict.unfreeze(
        graph.apply_transforms(
            params, TransformState.from_transforms(graph, {"core/Conv_0": swap})
        )
    )

    bn_params = permuted["params"]["core"]["BatchNorm_0"]
    np.testing.assert_allclose(
        bn_params["scale"],
        apply_perm_to_axis(
            params["params"]["core"]["BatchNorm_0"]["scale"],
            swap,
            axis=-1,
        ),
    )
    np.testing.assert_allclose(
        bn_params["bias"],
        apply_perm_to_axis(
            params["params"]["core"]["BatchNorm_0"]["bias"],
            swap,
            axis=-1,
        ),
    )
    bn_stats = permuted["batch_stats"]["core"]["BatchNorm_0"]
    np.testing.assert_allclose(
        bn_stats["mean"],
        apply_perm_to_axis(
            params["batch_stats"]["core"]["BatchNorm_0"]["mean"],
            swap,
            axis=-1,
        ),
    )
    np.testing.assert_allclose(
        bn_stats["var"],
        apply_perm_to_axis(
            params["batch_stats"]["core"]["BatchNorm_0"]["var"],
            swap,
            axis=-1,
        ),
    )
