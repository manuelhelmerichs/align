"""Intentional public API for alignment, loaded without importing JAX eagerly.

The CLI configures platform preferences before touching runtime modules. Keeping
these exports lazy preserves that ordering while making the package root represent
the complete alignment operation rather than only its matching stage.
"""

import importlib
from typing import Any

_PUBLIC_API = {
    "AlignmentRunner": "align.runtime",
    "ArchitectureRecipe": "align.architectures",
    "RunArtifactStore": "align.runtime",
    "RunConfig": "align.config",
    "SampleAlignmentResult": "align.runtime.pipeline",
    "ScaleCanonicalizer": "align.canonicalization",
    "ScaleState": "align.canonicalization",
    "SolverSequence": "align.matching",
    "SolverStep": "align.matching",
    "SymmetryGraph": "align.symmetry",
    "TransformState": "align.matching",
    "available_recipes": "align.architectures",
    "build_solver_sequence": "align.matching",
    "get_recipe": "align.architectures",
    "match_batch": "align.matching",
    "match_component_across": "align.matching",
    "match_sample": "align.matching",
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
