"""LLaMA-style transformer recipe: RMSNorm + RoPE + GQA.

Targets pre-norm decoder trees with RMSNorm (scale only, no bias), rotary
position embeddings on the qk circuit, grouped-query attention, and bias-free
projections. Attention kernels are stored kv-group-structured: ``query`` as
``(d_model, kv_groups, heads_per_group, head_dim)``, ``key``/``value`` as
``(d_model, kv_groups, head_dim)``, ``out`` as ``(kv_groups, heads_per_group,
head_dim, d_model)``. Flat ``(d, H, dk)`` layouts with query head ``i`` in kv
group ``i // (H/G)`` reshape losslessly into this form.

The symmetry model (derived in the RMSNorm/RoPE/GQA subsection of
wiki/Architecture-RMSNorm-GQA-RoPE-Transformer.md) differs from the
LayerNorm/GPT family in three ways:

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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ..symmetry import RMSNormScaleConstraint, SymmetryGraph
from .graph_builder import SymmetryGraphBuilder
from .recipe import ArchitectureRecipe, register_recipe
from .rules import (
    DenseChainRule,
    GQARoPEAttentionRule,
    ResidualStreamRule,
    is_layernorm_module,
    is_rmsnorm_module,
)
from .schemas import RMSNORM_GQA_ROPE_TRANSFORMER_OPTIONS
from .transformer_inventory import (
    chain_dense_layers,
    discover_transformer_inventory,
)


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
    config_options: ClassVar[frozenset[str]] = RMSNORM_GQA_ROPE_TRANSFORMER_OPTIONS

    def __post_init__(self) -> None:
        if self.stream_transform_family not in ("signed_permutation", "orthogonal"):
            raise ValueError(
                "rmsnorm_gqa_rope_transformer stream_transform_family must be "
                "'signed_permutation' or 'orthogonal', got "
                f"{self.stream_transform_family!r}."
            )

    def _require_rmsnorm(self, module: Any, path: tuple[str, ...]) -> None:
        if not is_rmsnorm_module(module):
            if is_layernorm_module(module):
                raise ValueError(
                    f"Norm module {'/'.join(path)} carries a bias; this is a "
                    "LayerNorm stack — use the 'layernorm_mha_transformer' recipe."
                )
            raise ValueError(f"{'/'.join(path)} is not an RMSNorm module.")

    # -- composition -------------------------------------------------------

    def build_graph(self, params: Mapping[str, Any]) -> SymmetryGraph:
        inventory = discover_transformer_inventory(
            params,
            parameter_root=self.parameter_root,
            # LayerNorm is the structural superset (scale with optional bias);
            # the validator below rejects biased norms with a family-specific
            # message while still locating their surrounding block.
            norm_predicate=is_layernorm_module,
            validate_norm=self._require_rmsnorm,
            norm_name="RMSNorm",
            query_rank=4,
            query_layout="(d_model, kv_groups, heads_per_group, head_dim)",
            wrong_layout_hint=(
                "Flat-head layouts belong to the 'layernorm_mha_transformer' recipe."
            ),
        )
        d_model = inventory.d_model
        outside_norms = inventory.outside_norms
        if len(outside_norms) != 1:
            raise ValueError(
                f"Expected exactly one final RMSNorm outside the blocks, found "
                f"{len(outside_norms)}."
            )
        outside_chains = chain_dense_layers(inventory.outside_dense, d_model)
        if not outside_chains:
            raise ValueError(
                "Expected at least one stream-fed dense chain (the model head) "
                "outside the blocks."
            )

        builder = SymmetryGraphBuilder(params, architecture=self.name)
        ResidualStreamRule(
            d_model=d_model,
            layernorm_paths=tuple(
                inventory.full_path(path) for path in inventory.norm_paths
            ),
            feature_leaves=tuple(
                inventory.full_path(path) for path in inventory.feature_leaves
            ),
            transform_family=self.stream_transform_family,
        ).add_to(builder)

        rms_constraints: list[tuple[tuple[str, ...], list[tuple[str, int]]]] = []
        for block in inventory.blocks:
            block_name = "/".join(block.scope)
            module_path = block.attention.path
            GQARoPEAttentionRule(
                component_id=f"{block_name}/attention",
                module_path=inventory.full_path(module_path),
                stream_group="stream",
            ).add_to(builder)

            norms, ffn_layers = block.norms, block.dense_layers
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
                layer_paths=tuple(
                    inventory.full_path(layer.path) for layer in ffn_layers
                ),
                input_group="stream",
                output_group="stream",
            ).add_to(builder)

            # Convention: the first RMSNorm (natural order) feeds attention,
            # the second feeds the FFN — the standard pre-norm block layout.
            attention_consumers = [
                ((*inventory.full_path(module_path), name, "kernel"), 0)
                for name in ("query", "key", "value")
            ]
            ffn_consumers = [((*inventory.full_path(ffn_layers[0].path), "kernel"), 0)]
            rms_constraints.append((inventory.full_path(norms[0]), attention_consumers))
            rms_constraints.append((inventory.full_path(norms[1]), ffn_consumers))

        head_consumers: list[tuple[tuple[str, ...], int]] = []
        for index, chain in enumerate(outside_chains):
            DenseChainRule(
                component_id=("classifier" if index == 0 else f"classifier_{index}"),
                layer_paths=tuple(inventory.full_path(layer.path) for layer in chain),
                input_group="stream",
            ).add_to(builder)
            head_consumers.append(((*inventory.full_path(chain[0].path), "kernel"), 0))
        rms_constraints.append((inventory.full_path(outside_norms[0]), head_consumers))

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
                "transformer_blocks": [
                    "/".join(block.scope) for block in inventory.blocks
                ],
            }
        )
        return builder.finish()
