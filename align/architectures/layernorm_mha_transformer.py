"""Transformer architecture recipe (pre-LN blocks, flax attention layout).

Targets the bayesmates GPT/ViT families: flax parameter trees whose attention
modules store head-structured kernels (``query/key/value`` as ``(d, H, dk)``,
``out`` as ``(H, dk, d)``). Discovery locates the blocks; construction is
delegated to symmetry rules:

- :class:`~align.architectures.rules.ResidualStreamRule`: one global
  ``stream`` group of size ``d_model`` binding LayerNorms and embeddings
  (LayerNorm commutes with feature permutations, so this is exact for
  LayerNorm-based stacks; orthogonal stream symmetry is out of scope),
- :class:`~align.architectures.rules.MHAAttentionRule` per block: the
  inter-head group plus per-slot qk/vo intra-head groups and the
  typed ``MHACircuitConstraint``,
- :class:`~align.architectures.rules.DenseChainRule` for FFN chains
  (stream-tied ends) and trailing classifier chains.

Discovery is structural and fail-loud: attention modules are found by their
``query/key/value/out`` children, transformer blocks are the nearest ancestor
scope holding a LayerNorm, and every remaining parameter must be accounted for
(LayerNorm, dense chain member, or embedding-style tensor with a trailing
``d_model`` axis).
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..symmetry import SymmetryGraph
from ..symmetry.tensor_ops import _descend, _maybe_descend
from .graph_builder import SymmetryGraphBuilder
from .recipe import ArchitectureRecipe, register_recipe
from .rules import (
    DenseChainRule,
    MHAAttentionRule,
    ResidualStreamRule,
    is_attention_module,
    is_dense_module,
    is_layernorm_module,
)


def _natural_key(value: str) -> list[int | str]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts if part]


def _path_key(path: tuple[str, ...]) -> list[list[int | str]]:
    return [_natural_key(part) for part in path]


def _shape(node: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.shape(node))


@dataclass(frozen=True)
class _AttentionModule:
    path: tuple[str, ...]
    num_heads: int
    head_dim: int


@dataclass(frozen=True)
class _DenseLayer:
    path: tuple[str, ...]
    din: int
    dout: int


@register_recipe
@dataclass
class LayerNormMHATransformerRecipe(ArchitectureRecipe):
    """Recipe for pre-LN transformer trees (bayesmates GPT/ViT layout)."""

    name: str = "layernorm_mha_transformer"
    parameter_root: str = ""

    # -- discovery -------------------------------------------------------

    def _root_path(self) -> tuple[str, ...]:
        return tuple(part for part in self.parameter_root.split(".") if part)

    def _find_attention_modules(
        self, subtree: Mapping[str, Any]
    ) -> list[tuple[str, ...]]:
        found: list[tuple[str, ...]] = []

        def _walk(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
            if is_attention_module(node):
                found.append(prefix)
                return
            for name, value in node.items():
                if isinstance(value, Mapping):
                    _walk(value, prefix + (str(name),))

        _walk(subtree, ())
        return sorted(found, key=_path_key)

    def _block_scope(
        self, subtree: Mapping[str, Any], attention_path: tuple[str, ...]
    ) -> tuple[str, ...]:
        for depth in range(len(attention_path) - 1, 0, -1):
            scope_path = attention_path[:depth]
            scope = _descend(subtree, scope_path)
            if any(is_layernorm_module(child) for child in scope.values()):
                return scope_path
        raise ValueError(
            "Could not locate a transformer block scope (an ancestor with a "
            f"LayerNorm) for attention module at {'/'.join(attention_path)}."
        )

    def _attention_spec(
        self, subtree: Mapping[str, Any], path: tuple[str, ...], d_model: int | None
    ) -> tuple[_AttentionModule, int]:
        module = _descend(subtree, path)
        q_shape = _shape(module["query"]["kernel"])
        if len(q_shape) != 3:
            raise ValueError(
                f"Attention module {'/'.join(path)} query kernel must be "
                f"(d_model, heads, head_dim), got {q_shape}."
            )
        d, num_heads, head_dim = q_shape
        if d_model is not None and d != d_model:
            raise ValueError(
                f"Attention module {'/'.join(path)} uses d_model {d}, expected "
                f"{d_model}."
            )
        return _AttentionModule(path=path, num_heads=num_heads, head_dim=head_dim), d

    def _collect_block_layers(
        self,
        scope: Mapping[str, Any],
        scope_path: tuple[str, ...],
        attention_path: tuple[str, ...],
    ) -> tuple[list[tuple[str, ...]], list[_DenseLayer]]:
        """Return (layernorm paths, dense FFN layers) inside one block scope."""

        layernorms: list[tuple[str, ...]] = []
        dense: list[_DenseLayer] = []
        attention_rel = attention_path[len(scope_path) :]

        def _walk(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
            for name, value in node.items():
                if not isinstance(value, Mapping):
                    continue
                path = prefix + (str(name),)
                if path == attention_rel:
                    continue
                if attention_rel[: len(path)] == path:
                    # Wrapper scope on the attention branch (e.g. the masked
                    # self-attention module around flax's MHDPA); walk through.
                    _walk(value, path)
                    continue
                if is_layernorm_module(value):
                    layernorms.append(scope_path + path)
                elif is_dense_module(value):
                    kernel_shape = _shape(value["kernel"])
                    dense.append(
                        _DenseLayer(
                            path=scope_path + path,
                            din=kernel_shape[0],
                            dout=kernel_shape[1],
                        )
                    )
                else:
                    _walk(value, path)

        _walk(scope, ())
        layernorms.sort(key=_path_key)
        dense.sort(key=lambda layer: _path_key(layer.path))
        return layernorms, dense

    def _collect_outside(
        self,
        subtree: Mapping[str, Any],
        block_scopes: set[tuple[str, ...]],
        d_model: int,
    ) -> tuple[list[tuple[str, ...]], list[_DenseLayer], list[tuple[str, ...]]]:
        """Return (layernorms, dense layers, embedding leaves) outside blocks."""

        layernorms: list[tuple[str, ...]] = []
        dense: list[_DenseLayer] = []
        leaves: list[tuple[str, ...]] = []

        def _walk(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
            for name, value in node.items():
                path = prefix + (str(name),)
                if path in block_scopes:
                    continue
                if isinstance(value, Mapping):
                    if is_layernorm_module(value):
                        layernorms.append(path)
                    elif is_dense_module(value):
                        kernel_shape = _shape(value["kernel"])
                        dense.append(
                            _DenseLayer(
                                path=path,
                                din=kernel_shape[0],
                                dout=kernel_shape[1],
                            )
                        )
                    elif "kernel" in value and not isinstance(value["kernel"], Mapping):
                        # Conv-style embedding (e.g. ViT patch embedding).
                        kernel_shape = _shape(value["kernel"])
                        if kernel_shape[-1] != d_model:
                            raise ValueError(
                                f"Module {'/'.join(path)} has kernel shape "
                                f"{kernel_shape}; expected a trailing d_model "
                                f"axis of {d_model}."
                            )
                        leaves.append(path + ("kernel",))
                        if "bias" in value:
                            bias_shape = _shape(value["bias"])
                            if bias_shape != (d_model,):
                                raise ValueError(
                                    f"Module {'/'.join(path)} bias must be "
                                    f"(d_model,), got {bias_shape}."
                                )
                            leaves.append(path + ("bias",))
                    else:
                        _walk(value, path)
                    continue
                shape = _shape(value)
                if not shape:
                    continue
                if shape[-1] != d_model:
                    raise ValueError(
                        f"Tensor {'/'.join(path)} with shape {shape} is not "
                        "part of a recognized transformer component and has no "
                        f"trailing d_model axis ({d_model})."
                    )
                leaves.append(path)

        _walk(subtree, ())
        layernorms.sort(key=_path_key)
        dense.sort(key=lambda layer: _path_key(layer.path))
        leaves.sort(key=_path_key)
        return layernorms, dense, leaves

    @staticmethod
    def _chain_dense_layers(
        dense: list[_DenseLayer], d_model: int
    ) -> list[list[_DenseLayer]]:
        """Group trailing dense layers into stream-fed chains, natural order."""

        chains: list[list[_DenseLayer]] = []
        for layer in dense:
            if chains and layer.din == chains[-1][-1].dout:
                chains[-1].append(layer)
            elif layer.din == d_model:
                chains.append([layer])
            else:
                raise ValueError(
                    f"Dense layer {'/'.join(layer.path)} (in={layer.din}) fits "
                    "neither the residual stream nor the preceding classifier "
                    "chain."
                )
        return chains

    # -- composition -------------------------------------------------------

    def build_graph(self, params: Mapping[str, Any]) -> SymmetryGraph:
        root_path = self._root_path()
        subtree = _maybe_descend(params, root_path) if root_path else params
        if not isinstance(subtree, Mapping):
            raise ValueError(
                f"parameter_root '{self.parameter_root}' not found in params."
            )

        attention_paths = self._find_attention_modules(subtree)
        if not attention_paths:
            raise ValueError(
                "No attention modules (query/key/value/out) found; is this a "
                "transformer parameter tree?"
            )

        d_model: int | None = None
        modules: list[_AttentionModule] = []
        for path in attention_paths:
            module, d_model = self._attention_spec(subtree, path, d_model)
            modules.append(module)
        assert d_model is not None

        block_scope_by_module = {
            module.path: self._block_scope(subtree, module.path) for module in modules
        }
        block_scopes = sorted(set(block_scope_by_module.values()), key=_path_key)
        scope_modules: dict[tuple[str, ...], list[_AttentionModule]] = {
            scope: [] for scope in block_scopes
        }
        for module in modules:
            scope_modules[block_scope_by_module[module.path]].append(module)
        for scope, members in scope_modules.items():
            if len(members) != 1:
                raise ValueError(
                    f"Transformer block {'/'.join(scope)} contains "
                    f"{len(members)} attention modules; expected exactly one."
                )

        block_layers = {
            scope: self._collect_block_layers(
                _descend(subtree, scope), scope, scope_modules[scope][0].path
            )
            for scope in block_scopes
        }
        outside_norms, outside_dense, embedding_leaves = self._collect_outside(
            subtree, set(block_scopes), d_model
        )

        def _full(path: tuple[str, ...]) -> tuple[str, ...]:
            return (*root_path, *path)

        layernorm_paths = [
            layernorm for scope in block_scopes for layernorm in block_layers[scope][0]
        ] + outside_norms

        builder = SymmetryGraphBuilder(params, architecture=self.name)
        ResidualStreamRule(
            d_model=d_model,
            layernorm_paths=tuple(_full(path) for path in layernorm_paths),
            feature_leaves=tuple(_full(path) for path in embedding_leaves),
        ).add_to(builder)

        for scope in block_scopes:
            block_name = "/".join(scope)
            module = scope_modules[scope][0]
            MHAAttentionRule(
                component_id=f"{block_name}/attention",
                module_path=_full(module.path),
                stream_group="stream",
            ).add_to(builder)

            ffn_layers = block_layers[scope][1]
            if ffn_layers:
                if ffn_layers[0].din != d_model or ffn_layers[-1].dout != d_model:
                    raise ValueError(
                        f"FFN chain in block {block_name} must run d_model -> "
                        f"... -> d_model ({d_model}), got "
                        f"{ffn_layers[0].din} -> ... -> {ffn_layers[-1].dout}."
                    )
                DenseChainRule(
                    component_id=f"{block_name}/ffn",
                    layer_paths=tuple(_full(layer.path) for layer in ffn_layers),
                    input_group="stream",
                    output_group="stream",
                ).add_to(builder)

        for index, chain in enumerate(self._chain_dense_layers(outside_dense, d_model)):
            DenseChainRule(
                component_id=("classifier" if index == 0 else f"classifier_{index}"),
                layer_paths=tuple(_full(layer.path) for layer in chain),
                input_group="stream",
            ).add_to(builder)

        builder.metadata.update(
            {
                "d_model": d_model,
                "transformer_blocks": ["/".join(scope) for scope in block_scopes],
                "attention_modules": [
                    {
                        "path": "/".join(module.path),
                        "num_heads": module.num_heads,
                        "head_dim": module.head_dim,
                    }
                    for module in modules
                ],
            }
        )
        return builder.finish()
