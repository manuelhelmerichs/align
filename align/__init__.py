"""Align API surface with lazy JAX import.

This defers importing JAX (and the heavy core implementation) until after
platform preferences have been applied by the CLI or caller.
"""

import importlib
from types import ModuleType
from typing import Any

__all__: list[str] = []
_CORE_MODULE: ModuleType | None = None


def _load_core() -> ModuleType:
    global _CORE_MODULE, __all__
    if _CORE_MODULE is None:
        _CORE_MODULE = importlib.import_module("align.rebasin")
        __all__ = list(getattr(_CORE_MODULE, "__all__", []))
    return _CORE_MODULE


def __getattr__(name: str) -> Any:
    module = _load_core()
    return getattr(module, name)


def __dir__() -> list[str]:
    module = _load_core()
    return sorted(set(globals().keys()) | set(dir(module)))
