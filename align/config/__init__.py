"""Configuration dataclasses and YAML loader for the align CLI.

This package provides the configuration system for the align pipeline.
Configuration is split across submodules for maintainability:

- ``paths``: Filesystem path configuration (PathConfig)
- ``selection``: Sample selection/filtering (SelectionConfig)
- ``stages``: Stage-specific configs (CanonicalizeConfig, MatchConfig, ScaleCanonicalizationConfig)
- ``runtime``: Runtime execution options (RuntimeConfig)
- ``loader``: YAML loading, merging, and validation
"""

from ..options import SolverStep
from .loader import (
    RunConfig,
    load_align_config,
    resolve_adapter_defaults,
    validate_method,
    validate_paths,
    validate_ref_sample,
)
from .paths import PathConfig
from .runtime import RuntimeConfig
from .selection import SelectionConfig
from .stages import CanonicalizeConfig, MatchConfig, ScaleCanonicalizationConfig

__all__ = [
    "SolverStep",
    "PathConfig",
    "SelectionConfig",
    "ScaleCanonicalizationConfig",
    "CanonicalizeConfig",
    "MatchConfig",
    "RuntimeConfig",
    "RunConfig",
    "load_align_config",
    "resolve_adapter_defaults",
    "validate_method",
    "validate_paths",
    "validate_ref_sample",
]
