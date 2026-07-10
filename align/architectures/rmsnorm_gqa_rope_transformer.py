"""LLaMA-style transformer recipe: RMSNorm + RoPE + GQA.

Targets pre-norm decoder trees with RMSNorm (scale only, no bias), rotary
position embeddings on the qk circuit, grouped-query attention, and bias-free
projections. Attention kernels are stored kv-group-structured: ``query`` as
``(d_model, kv_groups, heads_per_group, head_dim)``, ``key``/``value`` as
``(d_model, kv_groups, head_dim)``, ``out`` as ``(kv_groups, heads_per_group,
head_dim, d_model)``. Flat ``(d, H, dk)`` layouts with query head ``i`` in kv
group ``i // (H/G)`` reshape losslessly into this form.

The symmetry model (derived in the RMSNorm/RoPE/GQA subsection of
docs/theory.md) differs from the LayerNorm/GPT family in three ways:

- the RMSNorm residual stream carries a full orthogonal symmetry: the stream
  group declares ``signed_permutation`` by default (``orthogonal`` via
  ``stream_transform_family`` for Procrustes alignment), and the exact per-channel
  post-norm scale symmetry of each RMSNorm ``scale`` against its consumers is
  recorded as ``RMSNormScaleConstraint`` records (handled by canonicalization
  via signed gamma folding),
- RoPE restricts the qk intra-head symmetry to per-frequency-pair scaled
  rotations: the qk groups declare ``rotation_pairs`` (no permutations);
  solvers update them by the closed-form per-pair rotation projection and
  the canonicalization plan balances the pair scales,
- GQA quotients the head symmetry into kv-group permutations times per-group
  query-head permutations (the wreath product), emitted by
  :class:`~align.architectures.rules.GQARoPEAttentionRule`; vo groups
  carry the signed-permutation circuit symmetry.

Discovery is structural and fail-loud: attention modules are found by their
``query/key/value/out`` children with a 4-D query kernel, transformer blocks
are the nearest ancestor scope holding an RMSNorm, and per block the first
RMSNorm (natural order) must feed attention, the second the FFN chain. Every
remaining parameter must be an RMSNorm, a dense chain member, or an
embedding-style tensor with a trailing ``d_model`` axis.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..symmetry import RMSNormScaleConstraint, SymmetryGraph
from ..symmetry.tensor_ops import _descend, _maybe_descend
from .graph_builder import SymmetryGraphBuilder
from .recipe import ArchitectureRecipe, register_recipe
from .rules import (
    DenseChainRule,
    GQARoPEAttentionRule,
    ResidualStreamRule,
    is_attention_module,
    is_dense_module,
    is_layernorm_module,
    is_rmsnorm_module,
)


def _natural_key(value: str) -> list[int | str]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts if part]


def _path_key(path: tuple[str, ...]) -> list[list[int | str]]:
    return [_natural_key(part) for part in path]


def _shape(node: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.shape(node))


@dataclass(frozen=True)
class _DenseLayer:
    path: tuple[str, ...]
    din: int
    dout: int


@register_recipe
@dataclass
class RMSNormGQARoPETransformerRecipe(ArchitectureRecipe):
    """Recipe for RMSNorm + RoPE + GQA pre-norm decoder trees.

    ``stream_transform_family`` selects the modeled stream symmetry class:
    ``signed_permutation`` (default; exact, discrete, no preconditions) or
    ``orthogonal`` (the stream's full symmetry — enables ``procrustes``
    schedule steps, and requires gamma-folded parameters at matching time).
    """

    name: str = "rmsnorm_gqa_rope_transformer"
    parameter_root: str = ""
    stream_transform_family: str = "signed_permutation"

    def __post_init__(self) -> None:
        if self.stream_transform_family not in ("signed_permutation", "orthogonal"):
            raise ValueError(
                "rmsnorm_gqa_rope_transformer stream_transform_family must be "
                "'signed_permutation' or 'orthogonal', got "
                f"{self.stream_transform_family!r}."
            )

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
            if any(is_rmsnorm_module(child) for child in scope.values()):
                return scope_path
        raise ValueError(
            "Could not locate a transformer block scope (an ancestor with an "
            f"RMSNorm) for attention module at {'/'.join(attention_path)}."
        )

    def _require_rmsnorm(self, module: Any, path: tuple[str, ...]) -> None:
        if not is_rmsnorm_module(module):
            if is_layernorm_module(module):
                raise ValueError(
                    f"Norm module {'/'.join(path)} carries a bias; this is a "
                    "LayerNorm stack — use the 'layernorm_mha_transformer' recipe."
                )
            raise ValueError(f"{'/'.join(path)} is not an RMSNorm module.")

    def _collect_block_layers(
        self,
        scope: Mapping[str, Any],
        scope_path: tuple[str, ...],
        attention_path: tuple[str, ...],
    ) -> tuple[list[tuple[str, ...]], list[_DenseLayer]]:
        """Return (RMSNorm paths, dense FFN layers) inside one block scope."""

        norms: list[tuple[str, ...]] = []
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
                    _walk(value, path)
                    continue
                if is_layernorm_module(value):
                    self._require_rmsnorm(value, scope_path + path)
                    norms.append(scope_path + path)
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
        norms.sort(key=_path_key)
        dense.sort(key=lambda layer: _path_key(layer.path))
        return norms, dense

    def _collect_outside(
        self,
        subtree: Mapping[str, Any],
        block_scopes: set[tuple[str, ...]],
        d_model: int,
    ) -> tuple[list[tuple[str, ...]], list[_DenseLayer], list[tuple[str, ...]]]:
        """Return (RMSNorms, dense layers, embedding leaves) outside blocks."""

        norms: list[tuple[str, ...]] = []
        dense: list[_DenseLayer] = []
        leaves: list[tuple[str, ...]] = []

        def _walk(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
            for name, value in node.items():
                path = prefix + (str(name),)
                if path in block_scopes:
                    continue
                if isinstance(value, Mapping):
                    if is_layernorm_module(value):
                        self._require_rmsnorm(value, path)
                        norms.append(path)
                    elif is_dense_module(value):
                        kernel_shape = _shape(value["kernel"])
                        dense.append(
                            _DenseLayer(
                                path=path,
                                din=kernel_shape[0],
                                dout=kernel_shape[1],
                            )
                        )
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
        norms.sort(key=_path_key)
        dense.sort(key=lambda layer: _path_key(layer.path))
        leaves.sort(key=_path_key)
        return norms, dense, leaves

    @staticmethod
    def _chain_dense_layers(
        dense: list[_DenseLayer], d_model: int
    ) -> list[list[_DenseLayer]]:
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
        for path in attention_paths:
            q_shape = _shape(_descend(subtree, path)["query"]["kernel"])
            if len(q_shape) != 4:
                raise ValueError(
                    f"Attention module {'/'.join(path)} query kernel must be "
                    "(d_model, kv_groups, heads_per_group, head_dim), got "
                    f"{q_shape}. Flat-head layouts belong to the "
                    "'layernorm_mha_transformer' recipe."
                )
            if d_model is None:
                d_model = q_shape[0]
            elif q_shape[0] != d_model:
                raise ValueError(
                    f"Attention module {'/'.join(path)} uses d_model "
                    f"{q_shape[0]}, expected {d_model}."
                )
        assert d_model is not None

        block_scope_by_module = {
            path: self._block_scope(subtree, path) for path in attention_paths
        }
        block_scopes = sorted(set(block_scope_by_module.values()), key=_path_key)
        scope_modules: dict[tuple[str, ...], list[tuple[str, ...]]] = {
            scope: [] for scope in block_scopes
        }
        for path in attention_paths:
            scope_modules[block_scope_by_module[path]].append(path)
        for scope, members in scope_modules.items():
            if len(members) != 1:
                raise ValueError(
                    f"Transformer block {'/'.join(scope)} contains "
                    f"{len(members)} attention modules; expected exactly one."
                )

        block_layers = {
            scope: self._collect_block_layers(
                _descend(subtree, scope), scope, scope_modules[scope][0]
            )
            for scope in block_scopes
        }
        outside_norms, outside_dense, embedding_leaves = self._collect_outside(
            subtree, set(block_scopes), d_model
        )
        if len(outside_norms) != 1:
            raise ValueError(
                f"Expected exactly one final RMSNorm outside the blocks, found "
                f"{len(outside_norms)}."
            )
        outside_chains = self._chain_dense_layers(outside_dense, d_model)
        if not outside_chains:
            raise ValueError(
                "Expected at least one stream-fed dense chain (the model head) "
                "outside the blocks."
            )

        def _full(path: tuple[str, ...]) -> tuple[str, ...]:
            return (*root_path, *path)

        norm_paths = [
            norm for scope in block_scopes for norm in block_layers[scope][0]
        ] + outside_norms

        builder = SymmetryGraphBuilder(params, architecture=self.name)
        ResidualStreamRule(
            d_model=d_model,
            layernorm_paths=tuple(_full(path) for path in norm_paths),
            feature_leaves=tuple(_full(path) for path in embedding_leaves),
            transform_family=self.stream_transform_family,
        ).add_to(builder)

        rms_constraints: list[tuple[tuple[str, ...], list[tuple[str, int]]]] = []
        for scope in block_scopes:
            block_name = "/".join(scope)
            module_path = scope_modules[scope][0]
            GQARoPEAttentionRule(
                component_id=f"{block_name}/attention",
                module_path=_full(module_path),
                stream_group="stream",
            ).add_to(builder)

            norms, ffn_layers = block_layers[scope]
            if len(norms) != 2:
                raise ValueError(
                    f"Transformer block {block_name} has {len(norms)} RMSNorms; "
                    "expected exactly two (attention norm, FFN norm)."
                )
            if not ffn_layers:
                raise ValueError(f"Transformer block {block_name} has no FFN chain.")
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

            # Convention: the first RMSNorm (natural order) feeds attention,
            # the second feeds the FFN — the standard pre-norm block layout.
            attention_consumers = [
                ((*_full(module_path), name, "kernel"), 0)
                for name in ("query", "key", "value")
            ]
            ffn_consumers = [((*_full(ffn_layers[0].path), "kernel"), 0)]
            rms_constraints.append((_full(norms[0]), attention_consumers))
            rms_constraints.append((_full(norms[1]), ffn_consumers))

        head_consumers: list[tuple[tuple[str, ...], int]] = []
        for index, chain in enumerate(outside_chains):
            DenseChainRule(
                component_id=("classifier" if index == 0 else f"classifier_{index}"),
                layer_paths=tuple(_full(layer.path) for layer in chain),
                input_group="stream",
            ).add_to(builder)
            head_consumers.append(((*_full(chain[0].path), "kernel"), 0))
        rms_constraints.append((_full(outside_norms[0]), head_consumers))

        for norm_path, consumers in rms_constraints:
            scale_id = builder.tensor((*norm_path, "scale"))
            consumer_ids = [(builder.tensor(path), axis) for path, axis in consumers]
            builder.add_constraint(
                RMSNormScaleConstraint(
                    scale=scale_id,
                    consumers=tuple(consumer_ids),
                )
            )

        builder.metadata.update(
            {
                "d_model": d_model,
                "transformer_blocks": ["/".join(scope) for scope in block_scopes],
            }
        )
        return builder.finish()
