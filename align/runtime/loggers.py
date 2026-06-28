"""Incremental artifact loggers used by runtime runners."""

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from ..artifacts import (
    write_aux_artifact,
    write_permutations_artifact,
    write_scale_factors_artifact,
)
from ..samples import WeightSample, create_sample_codec
from ..state import ArtifactChecksumStore, SampleManifest, SampleRecord, _file_checksum
from .common import _STATE_SAVE_INTERVAL_SECONDS


class IncrementalArtifactLogger(ABC):
    """Shared checksum and artifact helpers for pipeline loggers."""

    def __init__(
        self,
        manifest: SampleManifest,
        output_dir: Path,
        *,
        checksum_kinds: Sequence[str],
        primary_kind: str = "final",
    ) -> None:
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.state_dir = self.output_dir / "state"
        self.aux_dir = self.state_dir / "aux"
        self.tree_path = Path(self.manifest.tree_path)
        self.sample_codec = create_sample_codec(
            self.manifest.sample_format,
            tree_path=self.tree_path,
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.aux_dir.mkdir(parents=True, exist_ok=True)
        self._checksums = ArtifactChecksumStore(
            self.state_dir / "artifact_checksums.npy",
            self.manifest.total,
            kinds=tuple(checksum_kinds),
        )
        self.primary_kind = primary_kind
        self._last_flush = time.time()
        self._flush_interval = _STATE_SAVE_INTERVAL_SECONDS
        self._tree_copied = False
        self.primary_dir: Path | None = None

    @staticmethod
    def _relative_sample_path(record: SampleRecord) -> Path:
        return Path(record.relative_path)

    def _aux_path_for(self, record: SampleRecord) -> Path:
        rel = self._relative_sample_path(record)
        return self.aux_dir / rel.with_suffix(".json")

    def write_auxiliary(
        self, record: SampleRecord, data: Mapping[str, Any] | None
    ) -> None:
        path = self._aux_path_for(record)
        write_aux_artifact(path, data)

    def _copy_tree_once(self, target_dir: Path) -> None:
        if self._tree_copied:
            return
        self.sample_codec.copy_structure(target_dir, artifact_name="tree")
        self._tree_copied = True

    def _update_checksum(
        self,
        record: SampleRecord,
        *,
        kind: str,
        path: Path | None,
        checksum_value: int | None = None,
    ) -> None:
        if path is None and checksum_value is None:
            value = 0
        elif checksum_value is not None:
            value = int(checksum_value)
        else:
            value = _file_checksum(Path(path)) if path is not None else 0
        self._checksums.update(record.index, **{kind: value})

    def maybe_flush(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_flush) < self._flush_interval:
            return
        self._checksums.flush()
        self._last_flush = now

    @property
    def optional_artifacts(self) -> set[str]:
        """Kinds allowed to be missing (checksum=0)."""
        return set()

    @abstractmethod
    def artifact_paths(self, record: SampleRecord) -> Mapping[str, Path]:
        """Return artifact destinations for ``record`` keyed by checksum kind."""

    def commit_from_scratch(
        self,
        record: SampleRecord,
        *,
        artifacts: Mapping[str, Any],
        checksums: Mapping[str, int] | None = None,
    ) -> None:
        """Move scratch artifacts into place and update checksums."""
        for kind, dest in self.artifact_paths(record).items():
            src_key = f"{kind}_path"
            src_path = artifacts.get(src_key)
            checksum_override = None
            if checksums is not None and kind in checksums:
                checksum_override = int(checksums[kind])
            if (
                kind == self.primary_kind
                and src_path is not None
                and self.primary_dir is not None
            ):
                self._copy_tree_once(self.primary_dir)
            self._move_and_record(
                record,
                kind=kind,
                src_path=Path(src_path) if src_path else None,
                dest_path=dest,
                checksum_override=checksum_override,
                optional=kind in self.optional_artifacts,
            )

    def _move_and_record(
        self,
        record: SampleRecord,
        *,
        kind: str,
        src_path: Path | None,
        dest_path: Path,
        checksum_override: int | None = None,
        optional: bool = False,
    ) -> None:
        dest_path = Path(dest_path)
        if src_path is None:
            if not optional:
                raise FileNotFoundError(
                    f"{kind} artifact missing for sample {record.label}."
                )
            if dest_path.exists():
                dest_path.unlink()
            self._update_checksum(
                record, kind=kind, path=None, checksum_value=checksum_override or 0
            )
            return
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        Path(src_path).replace(dest_path)
        self._update_checksum(
            record, kind=kind, path=dest_path, checksum_value=checksum_override
        )

    def validate_artifacts(
        self, record: SampleRecord, *, autogenerate_missing: bool = True
    ) -> bool:
        entry = self._checksums.entry(record.index)
        for kind, path in self.artifact_paths(record).items():
            expected = entry.get(kind, 0)
            if not path.exists():
                if expected == 0 and (kind in self.optional_artifacts):
                    continue
                return False
            actual = _file_checksum(path)
            if expected == 0:
                if autogenerate_missing:
                    self._checksums.update(record.index, **{kind: actual})
            elif actual != expected:
                return False
        return True

    @abstractmethod
    def write_sample(self, record: SampleRecord, sample: WeightSample) -> None:
        """Persist the primary artifact for ``record``."""

    @abstractmethod
    def finalize(self, **kwargs: Any) -> None:
        """Persist summary metadata."""


_NORMALIZE_STATIC_KEYS = frozenset(
    {
        "task_type",
        "epsilon",
        "degenerate_handling",
        "normalize_biases",
        "activation",
        "classification_head_rescale",
        "num_classes",
    }
)
_NORMALIZE_PER_SAMPLE_KEYS = frozenset({"last_layer_gamma"})
_REBASIN_STATIC_KEYS = frozenset({"objective", "schedule"})
_REBASIN_PER_SAMPLE_KEYS = frozenset({"reference"})


class AlignLogger(IncrementalArtifactLogger):
    """Logger for the align pipeline.

    Extends IncrementalArtifactLogger with support for multi-stage pipelines,
    intermediate artifacts, and stage-specific outputs (permutations, scale factors).
    """

    def __init__(
        self,
        manifest: SampleManifest,
        output_dir: Path,
        *,
        stages: list[str],
        save_intermediate: bool = False,
    ) -> None:
        self.stages = list(stages)
        self.save_intermediate = bool(save_intermediate) and len(self.stages) > 1
        self._static_config: dict[str, dict[str, Any]] = {}

        checksum_kinds = ["final"]
        if "rebasin" in self.stages:
            checksum_kinds.append("permutations")
        if "normalize" in self.stages:
            checksum_kinds.append("scale_factors")
        if self.save_intermediate:
            checksum_kinds.append("intermediate")

        super().__init__(
            manifest,
            output_dir,
            checksum_kinds=checksum_kinds,
            primary_kind="final",
        )

        self.final_dir = self.output_dir / "final"
        self.permutations_dir = (
            self.output_dir / "permutations" if "rebasin" in self.stages else None
        )
        self.scale_factors_dir = (
            self.output_dir / "scale_factors" if "normalize" in self.stages else None
        )
        self.intermediate_dir = (
            self.output_dir / "intermediate" if self.save_intermediate else None
        )

        self.final_dir.mkdir(parents=True, exist_ok=True)
        if self.permutations_dir:
            self.permutations_dir.mkdir(parents=True, exist_ok=True)
        if self.scale_factors_dir:
            self.scale_factors_dir.mkdir(parents=True, exist_ok=True)
        if self.intermediate_dir:
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        self.primary_dir = self.final_dir
        self._intermediate_tree_copied = False

    @property
    def optional_artifacts(self) -> set[str]:
        optional = {"permutations", "scale_factors"}
        if self.save_intermediate:
            optional.add("intermediate")
        return optional

    def artifact_paths(self, record: SampleRecord) -> Mapping[str, Path]:
        rel = self._relative_sample_path(record)
        paths: dict[str, Path] = {"final": self.final_dir / rel}
        if self.permutations_dir is not None:
            paths["permutations"] = self.permutations_dir / rel
        if self.scale_factors_dir is not None:
            paths["scale_factors"] = self.scale_factors_dir / rel
        if self.intermediate_dir is not None:
            paths["intermediate"] = self.intermediate_dir / rel
        return paths

    def write_sample(self, record: SampleRecord, sample: WeightSample) -> None:
        path = self.artifact_paths(record)["final"]
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_codec.save(path, sample)
        self._copy_tree_once(self.final_dir)
        self._update_checksum(record, kind="final", path=path)

    def write_intermediate(self, record: SampleRecord, sample: WeightSample) -> None:
        if self.intermediate_dir is None:
            return
        path = self.artifact_paths(record).get("intermediate")
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_codec.save(path, sample)
        if not self._intermediate_tree_copied and self.tree_path.exists():
            self.sample_codec.copy_structure(
                self.intermediate_dir, artifact_name="tree"
            )
            self._intermediate_tree_copied = True
        self._update_checksum(record, kind="intermediate", path=path)

    def write_permutations(
        self, record: SampleRecord, perms: Mapping[str, Any]
    ) -> None:
        if self.permutations_dir is None:
            return
        path = self.artifact_paths(record).get("permutations")
        if path is None:
            return
        perm_path = write_permutations_artifact(path, perms)
        if perm_path is None:
            self._update_checksum(
                record, kind="permutations", path=None, checksum_value=0
            )
            return
        self._update_checksum(record, kind="permutations", path=perm_path)

    def write_scale_factors(
        self, record: SampleRecord, scales: list[jnp.ndarray]
    ) -> None:
        if self.scale_factors_dir is None:
            return
        path = self.artifact_paths(record).get("scale_factors")
        if path is None:
            return
        scale_path = write_scale_factors_artifact(path, scales)
        if scale_path is None:
            self._update_checksum(
                record, kind="scale_factors", path=None, checksum_value=0
            )
            return
        self._update_checksum(record, kind="scale_factors", path=scale_path)

    def write_aux(self, record: SampleRecord, payload: Mapping[str, Any]) -> None:
        """Write auxiliary data, merging with existing aux if present.

        Static config keys are extracted and stored once in the summary.
        Only per-sample varying data is stored in individual aux files.
        """
        if not payload:
            return

        filtered_payload: dict[str, Any] = {}
        for stage, stage_data in payload.items():
            if stage == "normalize" and isinstance(stage_data, dict):
                if "normalize" not in self._static_config:
                    self._static_config["normalize"] = {
                        k: v
                        for k, v in stage_data.items()
                        if k in _NORMALIZE_STATIC_KEYS
                    }
                per_sample = {
                    k: v
                    for k, v in stage_data.items()
                    if k in _NORMALIZE_PER_SAMPLE_KEYS
                }
                if per_sample:
                    filtered_payload[stage] = per_sample
            elif stage == "rebasin" and isinstance(stage_data, dict):
                if "rebasin" not in self._static_config:
                    self._static_config["rebasin"] = {
                        k: v for k, v in stage_data.items() if k in _REBASIN_STATIC_KEYS
                    }
                per_sample = {
                    k: v for k, v in stage_data.items() if k in _REBASIN_PER_SAMPLE_KEYS
                }
                if per_sample:
                    filtered_payload[stage] = per_sample
            else:
                filtered_payload[stage] = stage_data

        if not filtered_payload:
            return

        path = self._aux_path_for(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = {}
        existing.update({k: v for k, v in filtered_payload.items() if v is not None})
        path.write_text(json.dumps(existing, separators=(",", ":")))

    def finalize(self, *, elapsed_seconds: float) -> None:
        self.maybe_flush(force=True)
        summary: dict[str, Any] = {
            "stages": self.stages,
            "save_intermediate": self.save_intermediate,
            "elapsed_seconds": float(elapsed_seconds),
            "total_samples": self.manifest.total,
        }
        if self._static_config:
            summary["stage_config"] = self._static_config
        (self.output_dir / "align_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        aux_payload: list[dict[str, Any]] = []
        for record in self.manifest.records:
            path = self._aux_path_for(record)
            if path.exists():
                try:
                    aux_payload.append(json.loads(path.read_text()))
                except Exception:
                    aux_payload.append({})
            else:
                aux_payload.append({})
        (self.output_dir / "align_aux.json").write_text(
            json.dumps(aux_payload, separators=(",", ":"))
        )


__all__ = [
    "IncrementalArtifactLogger",
    "AlignLogger",
]
