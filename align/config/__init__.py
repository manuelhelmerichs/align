"""Configuration dataclasses and YAML loader for the align CLI.

This package provides the configuration system for the align pipeline.
Configuration is split across submodules for maintainability:

- ``paths``: Filesystem path configuration (PathConfig)
- ``architecture``: Recipe family + discovery options (ArchitectureConfig)
- ``selection``: Sample selection/filtering (SelectionConfig)
- ``stages``: Stage configs (CanonicalizeConfig, MatchConfig, ObjectiveConfig, SolverStep)
- ``runtime``: Runtime execution options (RuntimeConfig)
- ``loader``: YAML loading, merging, and validation
"""

from .architecture import ArchitectureConfig
from .loader import (
    SCHEMA_VERSION,
    RunConfig,
    load_align_config,
    resolve_recipe_defaults,
    validate_method,
    validate_paths,
    validate_ref_sample,
)
from .paths import PathConfig
from .runtime import RuntimeConfig
from .selection import SelectionConfig
from .stages import CanonicalizeConfig, MatchConfig, ObjectiveConfig, SolverStep

__all__ = [
    "SCHEMA_VERSION",
    "ArchitectureConfig",
    "SolverStep",
    "PathConfig",
    "SelectionConfig",
    "ObjectiveConfig",
    "CanonicalizeConfig",
    "MatchConfig",
    "RuntimeConfig",
    "RunConfig",
    "load_align_config",
    "resolve_recipe_defaults",
    "validate_method",
    "validate_paths",
    "validate_ref_sample",
]
