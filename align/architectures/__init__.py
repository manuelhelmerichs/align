"""Architecture recipes and the block adapters they compose.

Block adapters (``align.architectures.blocks``) construct alignment problems
through a shared ``ProblemBuilder``; architecture recipes discover block
locations in a parameter tree and compose them.
"""

import importlib
from typing import Any

from .base import (
    ArchitectureAdapter,
    available_adapters,
    get_adapter,
    register_adapter,
)

_ADAPTER_EXPORTS = {
    "DenseMLPAdapter": "align.architectures.dense_mlp",
    "ResNetAdapter": "align.architectures.resnet",
    "ConvStackBlockAdapter": "align.architectures.resnet",
    "TransformerAdapter": "align.architectures.transformer",
    "ProblemBuilder": "align.architectures.builder",
    "BlockAdapter": "align.architectures.blocks",
    "DenseStackBlockAdapter": "align.architectures.blocks",
    "AttentionBlockAdapter": "align.architectures.blocks",
    "ResidualStreamBlockAdapter": "align.architectures.blocks",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _ADAPTER_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ADAPTER_EXPORTS))


__all__ = [
    "ArchitectureAdapter",
    "AttentionBlockAdapter",
    "BlockAdapter",
    "ConvStackBlockAdapter",
    "DenseMLPAdapter",
    "DenseStackBlockAdapter",
    "ProblemBuilder",
    "ResNetAdapter",
    "ResidualStreamBlockAdapter",
    "TransformerAdapter",
    "available_adapters",
    "get_adapter",
    "register_adapter",
]
