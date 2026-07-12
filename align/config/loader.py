"""Configuration loader, validation, and the current RunConfig class."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ._utils import _merge_dicts, _validate_fields
from .architecture import ArchitectureConfig
from .paths import PathConfig, path_to_dict
from .runtime import RuntimeConfig
from .selection import SelectionConfig
from .stages import CanonicalizeConfig, CenterSoftmaxHeadConfig, MatchConfig

if TYPE_CHECKING:
    from ..sample_manifest import SampleManifest

_STAGES = ("canonicalize", "center_softmax_head", "match")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "paths",
        "architecture",
        "selection",
        "pipeline",
        "canonicalize",
        "center_softmax_head",
        "match",
        "runtime",
    }
)


@dataclass
class RunConfig:
    """Resolved configuration for one alignment run."""

    paths: PathConfig = field(default_factory=PathConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    pipeline: tuple[str, ...] = ("canonicalize", "match")
    canonicalize: CanonicalizeConfig | None = field(default_factory=CanonicalizeConfig)
    center_softmax_head: CenterSoftmaxHeadConfig | None = None
    match: MatchConfig | None = field(default_factory=MatchConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RunConfig:
        _validate_fields("top-level", payload, _TOP_LEVEL_FIELDS)
        pipeline_raw = payload.get("pipeline", ["canonicalize", "match"])
        if not isinstance(pipeline_raw, list) or not pipeline_raw:
            raise ValueError("Config must define a non-empty 'pipeline' list.")
        pipeline = tuple(str(stage).lower() for stage in pipeline_raw)
        unknown = [stage for stage in pipeline if stage not in _STAGES]
        if unknown:
            raise ValueError(
                "Unknown pipeline stage(s): "
                + ", ".join(unknown)
                + f". Available: {', '.join(_STAGES)}."
            )
        if len(set(pipeline)) != len(pipeline):
            raise ValueError("pipeline stages must be unique.")

        canonicalize_cfg: CanonicalizeConfig | None = None
        if "canonicalize" in pipeline:
            canonicalize_cfg = CanonicalizeConfig.from_mapping(
                payload.get("canonicalize", {})
            )
        elif "canonicalize" in payload:
            raise ValueError(
                "A 'canonicalize' section is present but 'canonicalize' is not in "
                "the pipeline."
            )

        center_softmax_head_cfg: CenterSoftmaxHeadConfig | None = None
        if "center_softmax_head" in pipeline:
            center_softmax_head_cfg = CenterSoftmaxHeadConfig.from_mapping(
                payload.get("center_softmax_head", {})
            )
        elif "center_softmax_head" in payload:
            raise ValueError(
                "A 'center_softmax_head' section is present but "
                "'center_softmax_head' is not in the pipeline."
            )

        match_cfg: MatchConfig | None = None
        if "match" in pipeline:
            match_cfg = MatchConfig.from_mapping(payload.get("match", {}))
        elif "match" in payload:
            raise ValueError(
                "A 'match' section is present but 'match' is not in the pipeline."
            )

        return cls(
            paths=PathConfig.from_mapping(payload.get("paths", {})),
            architecture=ArchitectureConfig.from_mapping(
                payload.get("architecture", {})
            ),
            selection=SelectionConfig.from_mapping(payload.get("selection", {})),
            pipeline=pipeline,
            canonicalize=canonicalize_cfg,
            center_softmax_head=center_softmax_head_cfg,
            match=match_cfg,
            runtime=RuntimeConfig.from_mapping(payload.get("runtime", {})),
        )

    def active_stages(self) -> list[str]:
        """Return the ordered pipeline stages."""

        return list(self.pipeline)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "paths": path_to_dict(self.paths),
            "architecture": self.architecture.as_dict(),
            "selection": asdict(self.selection),
            "pipeline": list(self.pipeline),
            "runtime": asdict(self.runtime),
        }
        payload["canonicalize"] = (
            self.canonicalize.as_dict() if self.canonicalize is not None else None
        )
        payload["center_softmax_head"] = (
            self.center_softmax_head.as_dict()
            if self.center_softmax_head is not None
            else None
        )
        payload["match"] = self.match.as_dict() if self.match is not None else None
        return payload


def load_align_config(
    config_path: Path | None, overrides: Mapping[str, Any] | None = None
) -> RunConfig:
    """Load a config YAML and merge CLI overrides."""

    data: dict[str, Any] = {}
    if config_path:
        text = os.path.expandvars(Path(config_path).read_text())
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, Mapping):  # pragma: no cover - defensive
            raise ValueError("Config file must define a mapping at the top level.")
        data = dict(loaded)
    if overrides:
        override_payload = dict(overrides)
        architecture_override = override_payload.get("architecture")
        if (
            isinstance(architecture_override, Mapping)
            and "family" in architecture_override
        ):
            # Architecture family is a discriminant: options from another recipe
            # are invalid, not recursively mergeable.
            data["architecture"] = dict(architecture_override)
            override_payload.pop("architecture")
        data = _merge_dicts(data, override_payload)
        if "pipeline" in override_payload:
            active = set(override_payload["pipeline"])
            for stage in _STAGES:
                if stage not in active:
                    data.pop(stage, None)
    return RunConfig.from_mapping(data)


def validate_paths(config: RunConfig) -> None:
    """Validate that required filesystem paths exist."""

    root = config.paths.experiment_root
    if root is None:
        raise ValueError("experiment_root must be specified via CLI or config file.")
    if not root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")
    samples_dir = config.paths.samples_dir or (root / "samples")
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    tree_path = config.paths.tree_path
    if tree_path is not None:
        resolved_tree = tree_path.resolve()
        if not resolved_tree.exists():
            raise FileNotFoundError(f"Tree path not found: {resolved_tree}")
        if not resolved_tree.is_file():
            raise FileNotFoundError(f"Tree path must be a file: {resolved_tree}")


def resolve_recipe_defaults(config: RunConfig) -> None:
    """Fill recipe discovery options derivable from resolved paths."""

    if config.architecture.family != "residual_convnet":
        return
    root = config.paths.experiment_root
    if root is None:
        return
    options = config.architecture.options
    if options.get("residual_topology") is not None:
        return
    # The producer emits a "module_graph.json" descriptor; align consumes it as
    # the recipe's residual_topology (an on-disk producer artifact name).
    topology_path = Path(root) / "module_graph.json"
    if topology_path.exists():
        options["residual_topology"] = str(topology_path)


def validate_reference_sample(
    manifest: SampleManifest, selection: SelectionConfig
) -> None:
    """Ensure the reference sample exists inside the manifest."""

    reference_label = (
        f"chain{selection.reference_chain}_sample{selection.reference_sample}"
    )
    if not any(record.label == reference_label for record in manifest.records):
        raise ValueError(
            f"Reference sample {reference_label} missing from manifest. "
            "Adjust selection filters."
        )


def validate_architecture(config: RunConfig) -> None:
    """Validate architecture-family availability (stage options self-validate)."""

    from ..architectures import available_recipes

    if config.architecture.family not in available_recipes():
        raise ValueError(
            f"Unknown architecture family '{config.architecture.family}'. "
            f"Available: {', '.join(available_recipes())}"
        )


__all__ = [
    "RunConfig",
    "load_align_config",
    "resolve_recipe_defaults",
    "validate_architecture",
    "validate_paths",
    "validate_reference_sample",
]
