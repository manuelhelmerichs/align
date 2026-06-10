"""Built-in architecture adapters."""

import importlib
from typing import Any

_ADAPTER_EXPORTS = {
    "DenseMLPAdapter": "align.architectures.dense_mlp",
    "ResNetAdapter": "align.architectures.resnet",
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


__all__ = list(_ADAPTER_EXPORTS)
