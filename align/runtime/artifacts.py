"""Alignment-artifact serialization and runtime storage."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..run_state import ArtifactChecksumStore, file_checksum
from ..sample_manifest import SampleManifest, SampleRecord
from ..samples import WeightSample, create_sample_codec
from .resources import STATE_SAVE_INTERVAL_SECONDS

_METADATA_KEY = "__metadata__"

_CANONICALIZE_STATIC_KEYS = frozenset(
    {
        "plan",
        "strategy",
        "epsilon",
        "degenerate_channels",
        "include_bias_in_norm",
        "activation",
    }
)
_MATCH_STATIC_KEYS = frozenset(
    {"objective", "solvers", "barycenter_passes", "transform_families"}
)
_MATCH_PER_SAMPLE_KEYS = frozenset({"reference"})


def _artifact_metadata(payload: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(dict(payload), sort_keys=True))


def _read_npz_artifact(
    path: str | Path, *, expected_type: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        if _METADATA_KEY not in payload.files:
            raise ValueError(f"{expected_type} artifact has no metadata.")
        metadata = json.loads(str(payload[_METADATA_KEY].item()))
        if metadata.get("artifact_type") != expected_type:
            raise ValueError(
                f"Expected a {expected_type} artifact, found "
                f"{metadata.get('artifact_type')!r}."
            )
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != _METADATA_KEY
        }
    return arrays, metadata


def write_transforms_artifact(
    path: str | Path,
    transforms: Mapping[str, Any],
    *,
    transform_families: Mapping[str, str] | None = None,
) -> Path | None:
    """Persist exact hard transforms keyed by stable symmetry-group id."""

    path = Path(path)
    if not transforms:
        if path.exists():
            path.unlink()
        return None

    families = dict(transform_families or {})
    if set(families) != set(transforms):
        raise ValueError(
            "Transform-family metadata must contain exactly the persisted groups."
        )
    payload = {
        str(group_id): np.asarray(matrix) for group_id, matrix in transforms.items()
    }
    payload[_METADATA_KEY] = _artifact_metadata(
        {
            "artifact_type": "transforms",
            "transform_families": families,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def read_transforms_artifact(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a transform artifact and its metadata."""

    return _read_npz_artifact(path, expected_type="transforms")


def write_scales_artifact(path: str | Path, scales: Mapping[str, Any]) -> Path | None:
    """Persist scale vectors keyed by stable group or constraint id."""

    path = Path(path)
    if not scales:
        if path.exists():
            path.unlink()
        return None
    payload = {str(scale_id): np.asarray(scale) for scale_id, scale in scales.items()}
    payload[_METADATA_KEY] = _artifact_metadata(
        {"artifact_type": "scales", "scale_ids": list(payload)}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def read_scales_artifact(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a scale artifact and its metadata."""

    return _read_npz_artifact(path, expected_type="scales")


def write_diagnostics_artifact(
    path: str | Path, diagnostics: Mapping[str, Any] | None
) -> Path | None:
    """Persist optional per-sample diagnostics."""

    path = Path(path)
    if not diagnostics:
        if path.exists():
            path.unlink()
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostics, separators=(",", ":")))
    return path


class RunArtifactStore:
    """Own the output layout, checksums, and run summaries."""

    def __init__(
        self,
        manifest: SampleManifest,
        output_dir: Path,
        *,
        stages: list[str],
        save_intermediate: bool = False,
    ) -> None:
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.state_dir = self.output_dir / "state"
        self.diagnostics_dir = self.state_dir / "sample_diagnostics"
        self.tree_path = Path(self.manifest.tree_path)
        self.sample_codec = create_sample_codec(
            self.manifest.sample_format,
            tree_path=self.tree_path,
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        self.stages = list(stages)
        self.save_intermediate = bool(save_intermediate) and len(self.stages) > 1
        self.stage_output_stages = (
            tuple(self.stages[:-1]) if self.save_intermediate else ()
        )
        self._static_config: dict[str, dict[str, Any]] = {}

        checksum_kinds = ["aligned_sample"]
        if "match" in self.stages:
            checksum_kinds.append("transforms")
        if "canonicalize" in self.stages:
            checksum_kinds.append("scales")
        checksum_kinds.extend(
            self._stage_output_kind(stage) for stage in self.stage_output_stages
        )

        self._checksums = ArtifactChecksumStore(
            self.state_dir / "artifact_checksums.npy",
            self.manifest.total,
            kinds=tuple(checksum_kinds),
        )
        self.primary_kind = "aligned_sample"
        self._last_flush = time.time()
        self._flush_interval = STATE_SAVE_INTERVAL_SECONDS
        self._structure_copied: set[Path] = set()

        self.aligned_samples_dir = self.output_dir / "aligned_samples"
        self.transforms_dir = (
            self.output_dir / "transforms" if "match" in self.stages else None
        )
        self.scales_dir = (
            self.output_dir / "scales" if "canonicalize" in self.stages else None
        )
        self.stage_output_dirs = {
            stage: self.output_dir / "stage_outputs" / stage
            for stage in self.stage_output_stages
        }

        self.aligned_samples_dir.mkdir(parents=True, exist_ok=True)
        if self.transforms_dir is not None:
            self.transforms_dir.mkdir(parents=True, exist_ok=True)
        if self.scales_dir is not None:
            self.scales_dir.mkdir(parents=True, exist_ok=True)
        for directory in self.stage_output_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        self.primary_dir = self.aligned_samples_dir

    @staticmethod
    def _stage_output_kind(stage: str) -> str:
        return f"stage_output_{stage}"

    @staticmethod
    def _relative_sample_path(record: SampleRecord) -> Path:
        return Path(record.relative_path)

    def _diagnostics_path_for(self, record: SampleRecord) -> Path:
        rel = self._relative_sample_path(record)
        return self.diagnostics_dir / rel.with_suffix(".json")

    def _copy_structure_once(self, target_dir: Path) -> None:
        if target_dir in self._structure_copied:
            return
        self.sample_codec.copy_structure(target_dir, artifact_name="tree")
        self._structure_copied.add(target_dir)

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
            value = file_checksum(Path(path)) if path is not None else 0
        self._checksums.update(record.index, **{kind: value})

    def maybe_flush(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_flush) < self._flush_interval:
            return
        self._checksums.flush()
        self._last_flush = now

    @property
    def optional_artifacts(self) -> set[str]:
        return {"transforms", "scales"}

    def artifact_paths(self, record: SampleRecord) -> Mapping[str, Path]:
        rel = self._relative_sample_path(record)
        paths: dict[str, Path] = {"aligned_sample": self.aligned_samples_dir / rel}
        if self.transforms_dir is not None:
            paths["transforms"] = self.transforms_dir / rel
        if self.scales_dir is not None:
            paths["scales"] = self.scales_dir / rel
        for stage, directory in self.stage_output_dirs.items():
            paths[self._stage_output_kind(stage)] = directory / rel
        return paths

    def commit_from_scratch(
        self,
        record: SampleRecord,
        *,
        artifacts: Mapping[str, Any],
        checksums: Mapping[str, int] | None = None,
    ) -> None:
        """Move scratch artifacts into place and update checksums."""

        for kind, destination in self.artifact_paths(record).items():
            source = artifacts.get(f"{kind}_path")
            checksum_override = (
                int(checksums[kind])
                if checksums is not None and kind in checksums
                else None
            )
            if kind == self.primary_kind and source is not None:
                self._copy_structure_once(self.primary_dir)
            self._move_and_record(
                record,
                kind=kind,
                src_path=Path(source) if source else None,
                dest_path=destination,
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
        src_path.replace(dest_path)
        if kind.startswith("stage_output_"):
            stage = kind.removeprefix("stage_output_")
            self._copy_structure_once(self.stage_output_dirs[stage])
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
                if expected == 0 and kind in self.optional_artifacts:
                    continue
                return False
            actual = file_checksum(path)
            if expected == 0 and autogenerate_missing:
                self._checksums.update(record.index, **{kind: actual})
            elif expected != 0 and actual != expected:
                return False
        return True

    def write_aligned_sample(self, record: SampleRecord, sample: WeightSample) -> None:
        path = self.artifact_paths(record)["aligned_sample"]
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_codec.save(path, sample)
        self._copy_structure_once(self.aligned_samples_dir)
        self._update_checksum(record, kind="aligned_sample", path=path)

    def write_stage_output(
        self, record: SampleRecord, stage: str, sample: WeightSample
    ) -> None:
        kind = self._stage_output_kind(stage)
        path = self.artifact_paths(record).get(kind)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_codec.save(path, sample)
        self._copy_structure_once(self.stage_output_dirs[stage])
        self._update_checksum(record, kind=kind, path=path)

    def write_transforms(
        self,
        record: SampleRecord,
        transforms: Mapping[str, Any],
        *,
        transform_families: Mapping[str, str] | None = None,
    ) -> None:
        path = self.artifact_paths(record).get("transforms")
        if path is None:
            return
        written = write_transforms_artifact(
            path,
            transforms,
            transform_families=transform_families,
        )
        self._update_checksum(record, kind="transforms", path=written)

    def write_scales(self, record: SampleRecord, scales: Mapping[str, Any]) -> None:
        path = self.artifact_paths(record).get("scales")
        if path is None:
            return
        written = write_scales_artifact(path, scales)
        self._update_checksum(record, kind="scales", path=written)

    def write_diagnostics(
        self, record: SampleRecord, payload: Mapping[str, Any]
    ) -> None:
        """Persist per-sample diagnostics and hoist static stage data to summary."""

        if not payload:
            return
        filtered_payload: dict[str, Any] = {}
        for stage, stage_data in payload.items():
            if stage == "canonicalize" and isinstance(stage_data, dict):
                self._static_config.setdefault(
                    "canonicalize",
                    {
                        key: value
                        for key, value in stage_data.items()
                        if key in _CANONICALIZE_STATIC_KEYS
                    },
                )
            elif stage == "match" and isinstance(stage_data, dict):
                self._static_config.setdefault(
                    "match",
                    {
                        key: value
                        for key, value in stage_data.items()
                        if key in _MATCH_STATIC_KEYS
                    },
                )
                per_sample = {
                    key: value
                    for key, value in stage_data.items()
                    if key in _MATCH_PER_SAMPLE_KEYS
                }
                if per_sample:
                    filtered_payload[stage] = per_sample
            else:
                filtered_payload[stage] = stage_data

        if not filtered_payload:
            return
        path = self._diagnostics_path_for(record)
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = {}
        existing.update(
            {key: value for key, value in filtered_payload.items() if value is not None}
        )
        write_diagnostics_artifact(path, existing)

    def finalize(
        self,
        *,
        elapsed_seconds: float,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self.maybe_flush(force=True)
        summary: dict[str, Any] = {
            "stages": self.stages,
            "stage_outputs": list(self.stage_output_stages),
            "elapsed_seconds": float(elapsed_seconds),
            "total_samples": self.manifest.total,
        }
        if self._static_config:
            summary["stage_config"] = self._static_config
        if diagnostics:
            summary["diagnostics"] = dict(diagnostics)
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        sample_diagnostics: list[dict[str, Any]] = []
        for record in self.manifest.records:
            path = self._diagnostics_path_for(record)
            if path.exists():
                try:
                    sample_diagnostics.append(json.loads(path.read_text()))
                except Exception:
                    sample_diagnostics.append({})
            else:
                sample_diagnostics.append({})
        (self.output_dir / "sample_diagnostics.json").write_text(
            json.dumps(sample_diagnostics, separators=(",", ":"))
        )


__all__ = [
    "RunArtifactStore",
    "read_scales_artifact",
    "read_transforms_artifact",
    "write_diagnostics_artifact",
    "write_scales_artifact",
    "write_transforms_artifact",
]
