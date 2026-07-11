"""Mutable run state, progress tracking, and artifact checksums."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .sample_manifest import SampleRecord

_CHECKSUM_CHUNK_SIZE = 1 << 20


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def file_checksum(path: Path) -> int:
    """Return a truncated SHA256 checksum for one artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHECKSUM_CHUNK_SIZE):
            digest.update(chunk)
    return int.from_bytes(digest.digest()[:8], "big")


class ProcessedTracker:
    """Bitset-backed tracker storing processed sample indices on disk."""

    def __init__(self, path: Path, total_samples: int):
        self.path = Path(path)
        self.total = int(total_samples)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if self.path.exists() else "w+"
        self._byte_length = (self.total + 7) // 8
        self._buffer = np.memmap(
            self.path, dtype=np.uint8, mode=mode, shape=(self._byte_length,)
        )
        if mode == "w+":
            self._buffer[:] = 0
        # ``mark`` stores index 0 in the least-significant bit of each byte.
        # Match that convention when reconstructing the persisted count.
        bits = np.unpackbits(np.asarray(self._buffer), bitorder="little")[: self.total]
        self._count = int(bits.sum())
        self._dirty = False

    @property
    def count(self) -> int:
        return self._count

    def mark(self, index: int) -> bool:
        self._validate_index(index)
        byte_offset, bit = divmod(index, 8)
        mask = np.uint8(1 << bit)
        current = self._buffer[byte_offset]
        if current & mask:
            return False
        self._buffer[byte_offset] = current | mask
        self._count += 1
        self._dirty = True
        return True

    def is_marked(self, index: int) -> bool:
        if not 0 <= index < self.total:
            return False
        byte_offset, bit = divmod(index, 8)
        return bool(self._buffer[byte_offset] & np.uint8(1 << bit))

    def clear(self, index: int) -> bool:
        self._validate_index(index)
        byte_offset, bit = divmod(index, 8)
        mask = np.uint8(1 << bit)
        current = self._buffer[byte_offset]
        if not current & mask:
            return False
        self._buffer[byte_offset] = current & np.uint8(~mask)
        self._count -= 1
        self._dirty = True
        return True

    def flush(self) -> None:
        if self._dirty:
            self._buffer.flush()
            self._dirty = False

    def _validate_index(self, index: int) -> None:
        if not 0 <= index < self.total:
            raise IndexError(
                f"Sample index {index} outside tracker bounds (total={self.total})."
            )


class ArtifactChecksumStore:
    """Memmapped checksums for persisted artifacts."""

    def __init__(
        self, path: Path, total_samples: int, kinds: Sequence[str] | None = None
    ) -> None:
        self.path = Path(path)
        self.total = int(total_samples)
        self.kinds = tuple(kinds or ("aligned_sample", "transforms"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        columns = max(1, len(self.kinds))
        mode = "r+" if self.path.exists() else "w+"
        if mode == "r+":
            expected_size = self.total * columns * np.dtype(np.uint64).itemsize
            actual_size = self.path.stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"Checksum store {self.path} has size {actual_size} bytes, "
                    f"expected {expected_size} for {self.total} samples and "
                    f"{columns} artifact kind(s)."
                )
        self._buffer = np.memmap(
            self.path,
            dtype=np.uint64,
            mode=mode,
            shape=(self.total, columns),
        )
        if mode == "w+":
            self._buffer[:] = 0
        self._dirty = False
        self._kind_index = {name: index for index, name in enumerate(self.kinds)}

    def update(self, index: int, **artifacts: int | None) -> None:
        self._validate_index(index)
        for name, value in artifacts.items():
            if name in self._kind_index and value is not None:
                self._buffer[index, self._kind_index[name]] = np.uint64(value)
                self._dirty = True

    def entry(self, index: int) -> dict[str, int]:
        self._validate_index(index)
        return {
            name: int(self._buffer[index, column])
            for name, column in self._kind_index.items()
        }

    def flush(self) -> None:
        if self._dirty:
            self._buffer.flush()
            self._dirty = False

    def _validate_index(self, index: int) -> None:
        if not 0 <= index < self.total:
            raise IndexError(
                f"Checksum index {index} outside bounds (total={self.total})."
            )


@dataclass
class RunState:
    """Mutable state persisted between runner invocations."""

    path: Path
    experiment_root: Path
    output_dir: Path
    manifest_path: Path
    config_digest: str
    manifest_digest: str
    stages: list[str]
    total_samples: int
    reference: Mapping[str, Any] = field(default_factory=dict)
    filters: Mapping[str, Any] = field(default_factory=dict)
    tracker_path: Path | None = None
    runtime_settings: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    completed: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.experiment_root = Path(self.experiment_root)
        self.output_dir = Path(self.output_dir)
        self.manifest_path = Path(self.manifest_path)
        self.tracker_path = (
            Path(self.tracker_path)
            if self.tracker_path is not None
            else self.state_dir / "processed_bitmap.bin"
        )
        self._tracker: ProcessedTracker | None = None

    @property
    def state_dir(self) -> Path:
        return self.output_dir / "state"

    def save(self) -> None:
        self.updated_at = time.time()
        if self._tracker is not None:
            self._tracker.flush()
        payload = {
            "experiment_root": str(self.experiment_root),
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "config_digest": self.config_digest,
            "manifest_digest": self.manifest_digest,
            "stages": list(self.stages),
            "reference": dict(self.reference),
            "filters": dict(self.filters),
            "total_samples": self.total_samples,
            "tracker_path": str(self.tracker_path),
            "runtime_settings": dict(self.runtime_settings),
            "metadata": dict(self.metadata),
            "elapsed_seconds": float(self.elapsed_seconds),
            "completed": self.completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        _atomic_write_text(self.path, json.dumps(payload, indent=2))

    def _ensure_tracker(self) -> ProcessedTracker:
        if self._tracker is None:
            self._tracker = ProcessedTracker(self.tracker_path, self.total_samples)
        return self._tracker

    def record_progress(
        self,
        record: SampleRecord | None = None,
        *,
        sample_index: int | None = None,
    ) -> None:
        index = record.index if record is not None else sample_index
        if index is None:
            raise ValueError("sample_index must be provided when recording progress.")
        if self._ensure_tracker().mark(int(index)):
            self.updated_at = time.time()

    def mark_complete(self) -> None:
        self.completed = True
        self.updated_at = time.time()
        if self._tracker is not None:
            self._tracker.flush()

    def processed_count(self) -> int:
        return self._ensure_tracker().count

    def is_processed(self, index: int) -> bool:
        return self._ensure_tracker().is_marked(index)

    def mark_unprocessed(self, index: int) -> None:
        if self._ensure_tracker().clear(index):
            self.updated_at = time.time()

    def add_elapsed(self, seconds: float) -> float:
        self.elapsed_seconds += float(seconds)
        self.updated_at = time.time()
        return self.elapsed_seconds

    @classmethod
    def load(cls, path: Path) -> RunState:
        payload = json.loads(Path(path).read_text())
        return cls(
            path=Path(path),
            experiment_root=Path(payload["experiment_root"]),
            output_dir=Path(payload["output_dir"]),
            manifest_path=Path(payload["manifest_path"]),
            config_digest=str(payload["config_digest"]),
            manifest_digest=str(payload["manifest_digest"]),
            stages=list(payload.get("stages") or []),
            reference=payload.get("reference", {}),
            filters=payload.get("filters", {}),
            total_samples=int(payload["total_samples"]),
            tracker_path=(
                Path(payload["tracker_path"]) if payload.get("tracker_path") else None
            ),
            runtime_settings=payload.get("runtime_settings", {}),
            metadata=payload.get("metadata", {}),
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
            completed=bool(payload.get("completed", False)),
            created_at=float(payload.get("created_at", time.time())),
            updated_at=float(payload.get("updated_at", time.time())),
        )


def compute_config_digest(payload: Mapping[str, Any]) -> str:
    """Return a deterministic hash for the resolved configuration payload."""

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ArtifactChecksumStore",
    "ProcessedTracker",
    "RunState",
    "compute_config_digest",
    "file_checksum",
]
