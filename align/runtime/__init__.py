"""Lazy runtime exports so worker device visibility precedes JAX imports."""

from __future__ import annotations

import importlib
from typing import Any

_PUBLIC_API = {
    "AlignmentRunner": "align.runtime.runner",
    "PrefetchingLoader": "align.runtime.loaders",
    "PreparedRun": "align.runtime.preflight",
    "RunArtifactStore": "align.runtime.artifacts",
    "RunState": "align.run_state",
    "SampleLoader": "align.runtime.loaders",
    "SampleManifest": "align.sample_manifest",
    "SampleRecord": "align.sample_manifest",
    "compute_config_digest": "align.run_state",
    "reference_stability_diagnostic": "align.runtime.diagnostics",
    "prepare_run": "align.runtime.preflight",
}

__all__ = sorted(_PUBLIC_API)


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_API.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
