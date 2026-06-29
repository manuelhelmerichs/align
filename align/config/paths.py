"""Path configuration for align."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..sample_formats import DEFAULT_SAMPLE_FORMAT, normalize_sample_format
from ._utils import _to_path, _validate_fields

_PATH_FIELDS = frozenset(
    {
        "experiment_root",
        "samples_dir",
        "tree_path",
        "output_dir",
        "sample_format",
    }
)


@dataclass
class PathConfig:
    """Filesystem locations for manifests, samples, and outputs."""

    experiment_root: Path | None = None
    samples_dir: Path | None = None
    tree_path: Path | None = None
    output_dir: Path | None = None
    sample_format: str = DEFAULT_SAMPLE_FORMAT

    def __post_init__(self) -> None:
        self.sample_format = normalize_sample_format(self.sample_format)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PathConfig:
        _validate_fields("paths", payload, _PATH_FIELDS)
        return cls(
            experiment_root=_to_path(payload.get("experiment_root")),
            samples_dir=_to_path(payload.get("samples_dir")),
            tree_path=_to_path(payload.get("tree_path")),
            output_dir=_to_path(payload.get("output_dir")),
            sample_format=normalize_sample_format(payload.get("sample_format")),
        )


def path_to_dict(config: PathConfig) -> dict[str, str | None]:
    """Convert PathConfig to a serializable dict."""
    return {
        "experiment_root": str(config.experiment_root)
        if config.experiment_root
        else None,
        "samples_dir": str(config.samples_dir) if config.samples_dir else None,
        "tree_path": str(config.tree_path) if config.tree_path else None,
        "output_dir": str(config.output_dir) if config.output_dir else None,
        "sample_format": config.sample_format,
    }


__all__ = ["PathConfig", "path_to_dict"]
