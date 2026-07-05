"""Configuration loader, validation, and the main AlignConfig class."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..sample_formats import normalize_sample_format
from ._utils import _merge_dicts, _validate_fields
from .paths import PathConfig, path_to_dict
from .runtime import RuntimeConfig
from .selection import SelectionConfig
from .stages import NormalizeConfig, RebasinConfig

if TYPE_CHECKING:
    from ..state import SampleManifest

_ALIGN_CONFIG_FIELDS = frozenset(
    {
        "paths",
        "architecture",
        "adapter",
        "selection",
        "order",
        "normalize",
        "rebasin",
        "runtime",
    }
)


@dataclass
class AlignConfig:
    """Configuration for the align CLI."""

    paths: PathConfig = field(default_factory=PathConfig)
    architecture: str = "dense_mlp"
    adapter: dict[str, Any] = field(default_factory=dict)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    order: str | None = None
    normalize: NormalizeConfig | None = field(default_factory=NormalizeConfig)
    rebasin: RebasinConfig | None = field(default_factory=RebasinConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AlignConfig:
        _validate_fields("top-level", payload, _ALIGN_CONFIG_FIELDS)
        normalize_cfg = (
            NormalizeConfig.from_mapping(payload.get("normalize"))
            if "normalize" in payload
            else None
        )
        rebasin_cfg = (
            RebasinConfig.from_mapping(payload.get("rebasin"))
            if "rebasin" in payload
            else None
        )
        return cls(
            paths=PathConfig.from_mapping(payload.get("paths", {})),
            architecture=str(payload.get("architecture", "dense_mlp")),
            adapter=dict(payload.get("adapter", {})),
            selection=SelectionConfig.from_mapping(payload.get("selection", {})),
            order=payload.get("order"),
            normalize=normalize_cfg,
            rebasin=rebasin_cfg,
            runtime=RuntimeConfig.from_mapping(payload.get("runtime", {})),
        )

    def active_stages(self) -> list[str]:
        """Return ordered list of active stages."""
        stages: list[str] = []
        norm_enabled = self.normalize is not None and self.normalize.enabled
        rebasin_enabled = self.rebasin is not None and self.rebasin.enabled

        if not norm_enabled and not rebasin_enabled:
            raise ValueError("At least one of normalize or rebasin must be enabled.")

        order = (self.order or "").lower()
        if order == "rebasin_first":
            if rebasin_enabled:
                stages.append("rebasin")
            if norm_enabled:
                stages.append("normalize")
        else:
            if norm_enabled:
                stages.append("normalize")
            if rebasin_enabled:
                stages.append("rebasin")
        return stages

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "paths": path_to_dict(self.paths),
            "architecture": self.architecture,
            "adapter": self.adapter,
            "selection": asdict(self.selection),
            "order": self.order,
            "runtime": asdict(self.runtime),
        }
        if self.normalize is not None:
            payload["normalize"] = {
                "enabled": self.normalize.enabled,
                "method": self.normalize.method,
                "layer_root": self.normalize.layer_root,
                "task_type": self.normalize.task_type,
                "num_classes": self.normalize.num_classes,
                "scale_normalize": asdict(self.normalize.scale_normalize),
            }
        else:
            payload["normalize"] = None
        if self.rebasin is not None:
            payload["rebasin"] = {
                "enabled": self.rebasin.enabled,
                "objective": self.rebasin.objective,
                "objective_kwargs": self.rebasin.objective_kwargs,
                "schedule": self.rebasin.schedule_payload(),
                "layer_root": self.rebasin.layer_root,
                "seed": self.rebasin.seed,
            }
        else:
            payload["rebasin"] = None
        return payload


def load_align_config(
    config_path: Path | None, overrides: Mapping[str, Any] | None = None
) -> AlignConfig:
    """Load a config YAML and merge CLI overrides."""

    data: dict[str, Any] = {}
    if config_path:
        text = os.path.expandvars(Path(config_path).read_text())
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, Mapping):  # pragma: no cover - defensive
            raise ValueError("Config file must define a mapping at the top level.")
        data = dict(loaded)
    if overrides:
        data = _merge_dicts(data, overrides)
    return AlignConfig.from_mapping(data)


def validate_paths(config: AlignConfig) -> None:
    """Validate that required filesystem paths exist."""

    root = config.paths.experiment_root
    if root is None:
        raise ValueError("experiment_root must be specified via CLI or config file.")
    if not root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")
    samples_dir = config.paths.samples_dir or (root / "samples")
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    normalize_sample_format(config.paths.sample_format)
    tree_path = config.paths.tree_path
    if tree_path is not None:
        resolved_tree = tree_path.resolve()
        if not resolved_tree.exists():
            raise FileNotFoundError(f"Tree path not found: {resolved_tree}")
        if not resolved_tree.is_file():
            raise FileNotFoundError(f"Tree path must be a file: {resolved_tree}")


def resolve_adapter_defaults(config: AlignConfig) -> None:
    """Fill adapter defaults derivable from resolved path configuration."""

    if config.architecture != "resnet":
        return
    root = config.paths.experiment_root
    if root is None:
        return
    adapter_kwargs = dict(config.adapter or {})
    if adapter_kwargs.get("module_graph") is not None:
        return
    module_graph_path = Path(root) / "module_graph.json"
    if module_graph_path.exists():
        adapter_kwargs["module_graph"] = str(module_graph_path)
        config.adapter = adapter_kwargs


def validate_ref_sample(manifest: SampleManifest, selection: SelectionConfig) -> None:
    """Ensure the reference sample exists inside the manifest."""

    ref_label = f"chain{selection.ref_chain}_sample{selection.ref_sample}"
    if not any(record.label == ref_label for record in manifest.records):
        raise ValueError(
            f"Reference sample {ref_label} missing from manifest. Adjust selection filters."
        )


def validate_method(config: AlignConfig) -> None:
    """Validate configured stage methods and architecture compatibility."""

    if config.rebasin is not None:
        config.rebasin.validate_method()
    if config.normalize is not None:
        config.normalize.validate_method()
        if config.normalize.enabled and config.architecture in (
            "resnet",
            "transformer",
        ):
            raise ValueError(
                f"Normalization is not supported for architecture "
                f"'{config.architecture}'. Disable normalize or use "
                "architecture 'dense_mlp'."
            )
    from ..architectures import available_adapters

    if config.architecture not in available_adapters():
        raise ValueError(
            f"Unknown architecture '{config.architecture}'. "
            f"Available: {', '.join(available_adapters())}"
        )


__all__ = [
    "AlignConfig",
    "load_align_config",
    "resolve_adapter_defaults",
    "validate_method",
    "validate_paths",
    "validate_ref_sample",
]
