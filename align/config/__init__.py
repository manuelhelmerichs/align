"""Configuration dataclasses and YAML loader for the align CLI.

This package provides the configuration system for the align pipeline.
Configuration is split across submodules for maintainability:

- ``paths``: Filesystem path configuration (PathConfig)
- ``selection``: Sample selection/filtering (SelectionConfig)
- ``stages``: Stage-specific configs (NormalizeConfig, RebasinConfig, NormalizationOptions)
- ``runtime``: Runtime execution options (RuntimeConfig)
- ``loader``: YAML loading, merging, and validation
"""

from ..options import RebasinScheduleStep
from .loader import (
    AlignConfig,
    load_align_config,
    resolve_adapter_defaults,
    validate_method,
    validate_paths,
    validate_ref_sample,
)
from .paths import PathConfig
from .runtime import RuntimeConfig
from .selection import SelectionConfig
from .stages import NormalizationOptions, NormalizeConfig, RebasinConfig

__all__ = [
    "RebasinScheduleStep",
    "PathConfig",
    "SelectionConfig",
    "NormalizationOptions",
    "NormalizeConfig",
    "RebasinConfig",
    "RuntimeConfig",
    "AlignConfig",
    "load_align_config",
    "resolve_adapter_defaults",
    "validate_method",
    "validate_paths",
    "validate_ref_sample",
]
