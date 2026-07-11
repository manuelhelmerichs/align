"""Transformer architecture recipe (pre-LN blocks, flax attention layout).

Targets the bundled GPT/ViT families: Flax parameter trees whose attention
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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ..symmetry import SymmetryGraph
from .graph_builder import SymmetryGraphBuilder
from .recipe import ArchitectureRecipe, register_recipe
from .rules import (
    DenseChainRule,
    MHAAttentionRule,
    ResidualStreamRule,
    is_layernorm_module,
)
from .schemas import LAYERNORM_MHA_TRANSFORMER_OPTIONS
from .transformer_inventory import (
    chain_dense_layers,
    discover_transformer_inventory,
)


@register_recipe
@dataclass
class LayerNormMHATransformerRecipe(ArchitectureRecipe):
    """Recipe for pre-LN transformer trees in the bundled GPT/ViT layout."""

    name: str = "layernorm_mha_transformer"
    parameter_root: str = ""
    config_options: ClassVar[frozenset[str]] = LAYERNORM_MHA_TRANSFORMER_OPTIONS

    # -- composition -------------------------------------------------------

    def build_graph(self, params: Mapping[str, Any]) -> SymmetryGraph:
        inventory = discover_transformer_inventory(
            params,
            parameter_root=self.parameter_root,
            norm_predicate=is_layernorm_module,
            validate_norm=lambda _module, _path: None,
            norm_name="LayerNorm",
            query_rank=3,
            query_layout="(d_model, heads, head_dim)",
            recognize_kernel_feature_modules=True,
        )
        d_model = inventory.d_model

        builder = SymmetryGraphBuilder(params, architecture=self.name)
        ResidualStreamRule(
            d_model=d_model,
            layernorm_paths=tuple(
                inventory.full_path(path) for path in inventory.norm_paths
            ),
            feature_leaves=tuple(
                inventory.full_path(path) for path in inventory.feature_leaves
            ),
        ).add_to(builder)

        for block in inventory.blocks:
            block_name = "/".join(block.scope)
            MHAAttentionRule(
                component_id=f"{block_name}/attention",
                module_path=inventory.full_path(block.attention.path),
                stream_group="stream",
            ).add_to(builder)

            ffn_layers = block.dense_layers
            if ffn_layers:
                if ffn_layers[0].din != d_model or ffn_layers[-1].dout != d_model:
                    raise ValueError(
                        f"FFN chain in block {block_name} must run d_model -> "
                        f"... -> d_model ({d_model}), got "
                        f"{ffn_layers[0].din} -> ... -> {ffn_layers[-1].dout}."
                    )
                DenseChainRule(
                    component_id=f"{block_name}/ffn",
                    layer_paths=tuple(
                        inventory.full_path(layer.path) for layer in ffn_layers
                    ),
                    input_group="stream",
                    output_group="stream",
                ).add_to(builder)

        outside_chains = chain_dense_layers(inventory.outside_dense, d_model)
        for index, chain in enumerate(outside_chains):
            DenseChainRule(
                component_id=("classifier" if index == 0 else f"classifier_{index}"),
                layer_paths=tuple(inventory.full_path(layer.path) for layer in chain),
                input_group="stream",
            ).add_to(builder)

        builder.metadata.update(
            {
                "d_model": d_model,
                "transformer_blocks": [
                    "/".join(block.scope) for block in inventory.blocks
                ],
                "attention_modules": [
                    {
                        "path": "/".join(module.path),
                        "num_heads": module.query_shape[1],
                        "head_dim": module.query_shape[2],
                    }
                    for module in inventory.attentions
                ],
            }
        )
        return builder.finish()
