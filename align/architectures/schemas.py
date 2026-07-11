"""JAX-free configuration schemas for architecture recipe registration."""

from __future__ import annotations

CONVNET_OPTIONS = frozenset({"parameter_root"})
MLP_OPTIONS = frozenset({"parameter_root"})
LAYERNORM_MHA_TRANSFORMER_OPTIONS = frozenset({"parameter_root"})
RMSNORM_GQA_ROPE_TRANSFORMER_OPTIONS = frozenset(
    {"parameter_root", "stream_transform_family"}
)
RESIDUAL_CONVNET_OPTIONS = frozenset(
    {
        "parameter_root",
        "batch_stats_root",
        "residual_topology",
        "residual_connections",
    }
)

__all__ = [
    "CONVNET_OPTIONS",
    "LAYERNORM_MHA_TRANSFORMER_OPTIONS",
    "MLP_OPTIONS",
    "RESIDUAL_CONVNET_OPTIONS",
    "RMSNORM_GQA_ROPE_TRANSFORMER_OPTIONS",
]
