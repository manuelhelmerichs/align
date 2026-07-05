"""Architecture recipe framework: the recipe contract and registry.

An :class:`ArchitectureAdapter` is a thin *recipe*: it discovers where blocks
live in a parameter tree and composes
:class:`~align.architectures.blocks.BlockAdapter` instances, which perform all
problem construction through a shared
:class:`~align.architectures.builder.ProblemBuilder`. The resulting
:class:`~align.alignment.AlignmentProblem` is consumed by both the rebasin
(permutation) and normalization (scale) stages. Concrete recipes live
alongside this module under ``align.architectures``.
"""

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class ArchitectureAdapter(ABC):
    """Discovery + composition recipe for one architecture family.

    Implementations locate block instances (dense stacks, attention modules,
    conv stacks, the residual stream) in a parameter tree and delegate
    construction to block adapters. Config and CLI select recipes by ``name``.
    """

    name: str

    @abstractmethod
    def build_problem(self, params: Mapping[str, Any]):
        """Return the graph-native alignment problem for ``params``."""


_ADAPTERS: dict[str, type[ArchitectureAdapter]] = {}
_BUILTIN_ADAPTERS: dict[str, tuple[str, str]] = {
    "dense_mlp": ("align.architectures.dense_mlp", "DenseMLPAdapter"),
    "resnet": ("align.architectures.resnet", "ResNetAdapter"),
    "transformer": ("align.architectures.transformer", "TransformerAdapter"),
}


def register_adapter(cls: type[ArchitectureAdapter]) -> type[ArchitectureAdapter]:
    """Decorator to register an ArchitectureAdapter implementation."""

    if not getattr(cls, "name", None):
        raise ValueError("Adapter classes must define a 'name' attribute.")
    _ADAPTERS[cls.name] = cls
    return cls


def _load_builtin_adapter(name: str) -> type[ArchitectureAdapter] | None:
    """Import and return the requested built-in adapter class, if known."""
    module_spec = _BUILTIN_ADAPTERS.get(name)
    if module_spec is None:
        return None

    module_name, class_name = module_spec
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    if not isinstance(adapter_cls, type) or not issubclass(
        adapter_cls, ArchitectureAdapter
    ):
        raise TypeError(
            f"Built-in adapter '{name}' resolved to invalid class {class_name!r}."
        )
    _ADAPTERS.setdefault(name, adapter_cls)
    return adapter_cls


def get_adapter(name: str, **kwargs: Any) -> ArchitectureAdapter:
    adapter_cls = _ADAPTERS.get(name)
    if adapter_cls is None:
        adapter_cls = _load_builtin_adapter(name)
    if adapter_cls is None:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown architecture adapter '{name}'. Available: {available_adapters()}"
        )
    return adapter_cls(**kwargs)


def available_adapters() -> list[str]:
    return sorted(set(_BUILTIN_ADAPTERS) | set(_ADAPTERS))


__all__ = [
    "ArchitectureAdapter",
    "available_adapters",
    "get_adapter",
    "register_adapter",
]
