"""Lightweight manifest and run-state utilities for the align CLI."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as np_format

from .sample_formats import (
    DEFAULT_SAMPLE_FORMAT,
    normalize_sample_format,
    sample_file_suffixes,
)

_INT_PATTERN = re.compile(r"(\d+)(?!.*\d)")
_CHECKSUM_CHUNK_SIZE = 1 << 20
_RECORD_DTYPE = np.dtype(
    [
        ("index", np.int64),
        ("chain_id", np.int64),
        ("sample_id", np.int64),
        ("path_offset", np.int64),
        ("path_length", np.int64),
    ]
)


def _records_array_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}.records.npy")


def _records_blob_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}.paths.bin")


def _tree_copy_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}.treedef.pkl")


def _tree_override_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}.tree_path.txt")


def _read_tree_override(manifest_path: Path, *, discard_stale: bool) -> Path | None:
    override_path = _tree_override_path(Path(manifest_path).resolve())
    if not override_path.exists():
        return None
    recorded = override_path.read_text().strip()
    if not recorded:
        return None
    candidate = Path(recorded).resolve()
    if candidate.exists():
        return candidate
    if discard_stale:
        try:
            override_path.unlink()
        except FileNotFoundError:
            pass
    return None


def _extract_last_int(text: str) -> int:
    match = _INT_PATTERN.search(text)
    if not match:
        raise ValueError(f"Unable to parse integer from '{text}'.")
    return int(match.group(1))


def _iter_chains(samples_dir: Path) -> Iterator[tuple[int, Path]]:
    for child in sorted(samples_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            yield _extract_last_int(child.name), child
        except ValueError:
            continue


def _iter_samples(chain_dir: Path, *, sample_format: str) -> Iterator[tuple[int, Path]]:
    suffixes = sample_file_suffixes(sample_format)
    files = sorted(
        [p for p in chain_dir.iterdir() if p.suffix in suffixes],
        key=lambda path: _extract_last_int(path.stem),
    )
    for file_path in files:
        yield _extract_last_int(file_path.stem), file_path


def _atomic_write_text(path: Path, text: str) -> None:
    """Persist ``text`` atomically to ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _file_checksum(path: Path) -> int:
    """Return the truncated SHA256 checksum for ``path``."""

    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(_CHECKSUM_CHUNK_SIZE)
            if not chunk:
                break
            sha.update(chunk)
    digest = sha.digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class SampleRecord:
    """Metadata for a single sample file."""

    index: int
    chain_id: int
    sample_id: int
    relative_path: str

    @property
    def label(self) -> str:
        return f"chain{self.chain_id}_sample{self.sample_id}"

    def absolute_path(self, samples_dir: Path) -> Path:
        return (samples_dir / self.relative_path).resolve()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chain_id": self.chain_id,
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SampleRecord:
        return cls(
            index=int(payload["index"]),
            chain_id=int(payload["chain_id"]),
            sample_id=int(payload["sample_id"]),
            relative_path=str(payload["relative_path"]),
        )


class _SharedSampleRecordList(Sequence[SampleRecord]):
    """Sequence backed by read-only memmaps for manifest sharing."""

    def __init__(self, records_path: Path, paths_path: Path) -> None:
        self._records = np.load(records_path, mmap_mode="r")
        self._length = int(self._records.shape[0])
        self._paths_handle = paths_path.open("rb")
        size = os.path.getsize(paths_path)
        if size > 0:
            self._paths_map = mmap.mmap(
                self._paths_handle.fileno(), 0, access=mmap.ACCESS_READ
            )
        else:
            self._paths_map = None
        self._cache: dict[int, SampleRecord] = {}

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._length))]
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError("SampleRecord index out of range")
        if idx in self._cache:
            return self._cache[idx]
        entry = self._records[idx]
        offset = int(entry["path_offset"])
        length = int(entry["path_length"])
        if length == 0:
            rel_path = ""
        else:
            if self._paths_map is None:
                raise ValueError("Manifest path blob missing.")
            rel_path = self._paths_map[offset : offset + length].decode("utf-8")
        record = SampleRecord(
            index=int(entry["index"]),
            chain_id=int(entry["chain_id"]),
            sample_id=int(entry["sample_id"]),
            relative_path=rel_path,
        )
        self._cache[idx] = record
        return record

    def __iter__(self):
        for idx in range(self._length):
            yield self[idx]

    def __del__(self):  # pragma: no cover - relies on GC timing
        try:
            if hasattr(self, "_paths_map") and self._paths_map is not None:
                self._paths_map.close()
        finally:
            try:
                if hasattr(self, "_paths_handle"):
                    self._paths_handle.close()
            except Exception:
                pass


@dataclass
class SampleManifest:
    """Immutable list of all samples participating in a rebasin run."""

    samples_dir: Path
    tree_path: Path
    records: Sequence[SampleRecord]
    ref_chain: int
    ref_sample: int
    reference_index: int
    filters: Mapping[str, Any]
    sample_format: str = DEFAULT_SAMPLE_FORMAT
    created_at: float = field(default_factory=lambda: time.time())

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def reference_record(self) -> SampleRecord:
        return self.records[self.reference_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples_dir": str(self.samples_dir),
            "tree_path": str(self.tree_path),
            "sample_format": self.sample_format,
            "records": [record.to_dict() for record in self.records],
            "ref_chain": self.ref_chain,
            "ref_sample": self.ref_sample,
            "reference_index": self.reference_index,
            "filters": dict(self.filters),
            "created_at": self.created_at,
            "digest": self.digest(),
        }

    def digest(self) -> str:
        payload = {
            "records": [record.relative_path for record in self.records],
            "ref_chain": self.ref_chain,
            "ref_sample": self.ref_sample,
            "sample_format": self.sample_format,
            "total": self.total,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_digest(self, expected: str) -> None:
        digest = self.digest()
        if digest != expected:
            raise ValueError(
                f"Sample manifest digest mismatch: expected {expected}, found {digest}"
            )

    def summary(self) -> dict[str, Any]:
        chains = sorted({record.chain_id for record in self.records})
        return {
            "chains": chains,
            "total_samples": self.total,
            "sample_format": self.sample_format,
            "reference": {
                "chain": self.ref_chain,
                "sample": self.ref_sample,
                "index": self.reference_index,
            },
            "filters": dict(self.filters),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")))
        self._write_shared_records(path)
        self._attach_shared_records(path)

    def ensure_shared_records(self, manifest_path: Path) -> None:
        manifest_path = Path(manifest_path).resolve()
        records_path = _records_array_path(manifest_path)
        paths_path = _records_blob_path(manifest_path)
        if not records_path.exists() or not paths_path.exists():
            self._write_shared_records(manifest_path)
        self._attach_shared_records(manifest_path)

    def ensure_shared_tree(self, manifest_path: Path) -> Path:
        manifest_path = Path(manifest_path).resolve()
        shared_tree = _read_tree_override(manifest_path, discard_stale=True)
        if shared_tree is not None:
            self.tree_path = shared_tree
            return shared_tree

        override_path = _tree_override_path(manifest_path)
        copy_path = _tree_copy_path(manifest_path)
        source = Path(self.tree_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Tree path not found: {source}")
        copy_path = copy_path.resolve()
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        if not copy_path.exists() or not source.samefile(copy_path):
            shutil.copy2(source, copy_path)
        override_path.write_text(str(copy_path))
        self.tree_path = copy_path
        return copy_path

    def _write_shared_records(self, manifest_path: Path) -> None:
        manifest_path = Path(manifest_path).resolve()
        records_path = _records_array_path(manifest_path)
        paths_path = _records_blob_path(manifest_path)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        total = self.total
        mmap_array = np_format.open_memmap(
            records_path, mode="w+", dtype=_RECORD_DTYPE, shape=(total,)
        )
        path_blob = bytearray()
        for idx, record in enumerate(self.records):
            encoded = record.relative_path.encode("utf-8")
            mmap_array[idx] = (
                int(record.index),
                int(record.chain_id),
                int(record.sample_id),
                len(path_blob),
                len(encoded),
            )
            path_blob.extend(encoded)
        mmap_array.flush()
        del mmap_array
        with paths_path.open("wb") as handle:
            handle.write(path_blob)

    def _attach_shared_records(self, manifest_path: Path) -> None:
        manifest_path = Path(manifest_path).resolve()
        records_path = _records_array_path(manifest_path)
        paths_path = _records_blob_path(manifest_path)
        if not records_path.exists() or not paths_path.exists():
            return
        self.records = _SharedSampleRecordList(records_path, paths_path)

    def _apply_shared_tree_override(self, manifest_path: Path) -> None:
        shared_tree = _read_tree_override(manifest_path, discard_stale=False)
        if shared_tree is not None:
            self.tree_path = shared_tree

    @classmethod
    def load(cls, path: Path) -> SampleManifest:
        payload = json.loads(Path(path).read_text())
        records = [SampleRecord.from_dict(item) for item in payload["records"]]
        manifest = cls(
            samples_dir=Path(payload["samples_dir"]).resolve(),
            tree_path=Path(payload["tree_path"]).resolve(),
            records=records,
            ref_chain=int(payload["ref_chain"]),
            ref_sample=int(payload["ref_sample"]),
            reference_index=int(payload["reference_index"]),
            filters=payload.get("filters", {}),
            sample_format=normalize_sample_format(str(payload["sample_format"])),
            created_at=float(payload.get("created_at", time.time())),
        )
        manifest._attach_shared_records(Path(path))
        manifest._apply_shared_tree_override(Path(path))
        return manifest

    @classmethod
    def build(
        cls,
        samples_dir: Path,
        tree_path: Path,
        *,
        chain_indices: Iterable[int] | None = None,
        samples_per_chain: int | None = None,
        sample_step: int = 1,
        max_total: int | None = None,
        ref_chain: int = 0,
        ref_sample: int = 0,
        sample_format: str = DEFAULT_SAMPLE_FORMAT,
    ) -> SampleManifest:
        samples_dir = Path(samples_dir).resolve()
        tree_path = Path(tree_path).resolve()
        sample_format = normalize_sample_format(sample_format)
        chain_filter = set(chain_indices) if chain_indices is not None else None
        sample_step = max(sample_step, 1)

        records: list[SampleRecord] = []
        total = 0
        ref_index: int | None = None

        for chain_id, chain_dir in _iter_chains(samples_dir):
            if chain_filter is not None and chain_id not in chain_filter:
                continue
            loaded = 0
            for offset, (sample_id, file_path) in enumerate(
                _iter_samples(chain_dir, sample_format=sample_format)
            ):
                if offset % sample_step != 0:
                    continue
                record = SampleRecord(
                    index=total,
                    chain_id=chain_id,
                    sample_id=sample_id,
                    relative_path=file_path.relative_to(samples_dir).as_posix(),
                )
                records.append(record)
                if record.chain_id == ref_chain and record.sample_id == ref_sample:
                    ref_index = record.index
                total += 1
                loaded += 1
                if max_total is not None and total >= max_total:
                    break
                if samples_per_chain is not None and loaded >= samples_per_chain:
                    break
            if max_total is not None and total >= max_total:
                break

        if not records:
            raise ValueError("No samples were discovered. Check the directory filters.")
        if ref_index is None:
            raise ValueError(
                "Reference sample was filtered out by the current selection options."
            )

        filters = {
            "chain_indices": sorted(chain_filter) if chain_filter else None,
            "samples_per_chain": samples_per_chain,
            "sample_step": sample_step,
            "max_total": max_total,
        }
        return cls(
            samples_dir=samples_dir,
            tree_path=tree_path,
            records=records,
            ref_chain=ref_chain,
            ref_sample=ref_sample,
            reference_index=ref_index,
            filters=filters,
            sample_format=sample_format,
        )


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
        bits = np.unpackbits(np.asarray(self._buffer))
        if bits.size > self.total:
            bits = bits[: self.total]
        self._count = int(bits.sum())
        self._dirty = False

    @property
    def count(self) -> int:
        return self._count

    def mark(self, index: int) -> bool:
        if not (0 <= index < self.total):
            raise IndexError(
                f"Sample index {index} outside tracker bounds (total={self.total})."
            )
        byte_offset = index // 8
        bit = index % 8
        mask = np.uint8(1 << bit)
        current = self._buffer[byte_offset]
        if current & mask:
            return False
        self._buffer[byte_offset] = current | mask
        self._count += 1
        self._dirty = True
        return True

    def is_marked(self, index: int) -> bool:
        if not (0 <= index < self.total):
            return False
        byte_offset = index // 8
        bit = index % 8
        mask = np.uint8(1 << bit)
        return bool(self._buffer[byte_offset] & mask)

    def flush(self) -> None:
        if self._dirty:
            self._buffer.flush()
            self._dirty = False

    def clear(self, index: int) -> bool:
        if not (0 <= index < self.total):
            raise IndexError(
                f"Sample index {index} outside tracker bounds (total={self.total})."
            )
        byte_offset = index // 8
        bit = index % 8
        mask = np.uint8(1 << bit)
        current = self._buffer[byte_offset]
        if not (current & mask):
            return False
        self._buffer[byte_offset] = current & np.uint8(~mask)
        self._count -= 1
        self._dirty = True
        return True


class ArtifactChecksumStore:
    """Memmapped checksums for persisted artifacts."""

    def __init__(
        self, path: Path, total_samples: int, kinds: Sequence[str] | None = None
    ):
        self.path = Path(path)
        self.total = int(total_samples)
        self.kinds: tuple[str, ...] = tuple(kinds or ("final", "permutations"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

        columns = max(1, len(self.kinds))
        mode = "r+" if self.path.exists() else "w+"
        if mode == "r+":
            try:
                self._buffer = np.memmap(
                    self.path, dtype=np.uint64, mode=mode, shape=(self.total, columns)
                )
            except ValueError:
                size = self.path.stat().st_size
                inferred_cols = max(
                    1, size // (self.total * np.dtype(np.uint64).itemsize)
                )
                self._buffer = np.memmap(
                    self.path,
                    dtype=np.uint64,
                    mode=mode,
                    shape=(self.total, inferred_cols),
                )
                if inferred_cols != columns:
                    self.kinds = tuple(self.kinds[:inferred_cols]) + tuple(
                        f"col{idx}" for idx in range(len(self.kinds), inferred_cols)
                    )
        else:
            self._buffer = np.memmap(
                self.path, dtype=np.uint64, mode=mode, shape=(self.total, columns)
            )
            self._buffer[:] = 0
        self._dirty = False
        self._kind_index = {name: idx for idx, name in enumerate(self.kinds)}

    def update(self, index: int, **artifacts: int | None) -> None:
        if not (0 <= index < self.total):
            raise IndexError(
                f"Checksum index {index} outside bounds (total={self.total})."
            )
        for name, value in artifacts.items():
            if name not in self._kind_index or value is None:
                continue
            self._buffer[index, self._kind_index[name]] = np.uint64(value)
            self._dirty = True

    def entry(self, index: int) -> dict[str, int]:
        if not (0 <= index < self.total):
            raise IndexError(
                f"Checksum index {index} outside bounds (total={self.total})."
            )
        values = {
            name: int(self._buffer[index, idx]) for idx, name in enumerate(self.kinds)
        }
        return values

    def flush(self) -> None:
        if self._dirty:
            self._buffer.flush()
            self._dirty = False


@dataclass
class RunManifest:
    """State persisted between runner invocations."""

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
    stage_completion: dict[str, bool] = field(default_factory=dict)
    order: str | None = None
    tracker_path: Path | None = None
    runtime_settings: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    completed: bool = False
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.experiment_root = Path(self.experiment_root)
        self.output_dir = Path(self.output_dir)
        self.manifest_path = Path(self.manifest_path)
        if self.tracker_path is None:
            self.tracker_path = self.state_dir / "processed_bitmap.bin"
        else:
            self.tracker_path = Path(self.tracker_path)
        self.stage_completion = {
            stage: bool(self.stage_completion.get(stage, False))
            for stage in self.stages
        }
        self._tracker: ProcessedTracker | None = None

    @property
    def state_dir(self) -> Path:
        return self.output_dir / "state"

    def save(self) -> None:
        self.updated_at = time.time()
        payload = {
            "experiment_root": str(self.experiment_root),
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "config_digest": self.config_digest,
            "manifest_digest": self.manifest_digest,
            "stages": list(self.stages),
            "stage_completion": dict(self.stage_completion),
            "order": self.order,
            "reference": dict(self.reference),
            "filters": dict(self.filters),
            "total_samples": self.total_samples,
            "tracker_path": str(self.tracker_path) if self.tracker_path else None,
            "runtime_settings": dict(self.runtime_settings),
            "metadata": dict(self.metadata),
            "elapsed_seconds": float(self.elapsed_seconds),
            "completed": self.completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self._tracker is not None:
            self._tracker.flush()
        _atomic_write_text(self.path, json.dumps(payload, indent=2))

    def _ensure_tracker(self) -> ProcessedTracker:
        if self.tracker_path is None:
            self.tracker_path = self.state_dir / "processed_bitmap.bin"
        if self._tracker is None:
            self._tracker = ProcessedTracker(self.tracker_path, self.total_samples)
        return self._tracker

    def record_progress(
        self,
        record: SampleRecord | None = None,
        *,
        sample_index: int | None = None,
    ) -> None:
        if record is not None:
            sample_index = record.index
        if sample_index is None:
            raise ValueError("sample_index must be provided when recording progress.")

        tracker = self._ensure_tracker()
        inserted = tracker.mark(int(sample_index))
        if not inserted:
            return

        self.updated_at = time.time()

    def mark_stage_complete(self, stage: str) -> None:
        if stage not in self.stage_completion:
            self.stage_completion[stage] = True
        else:
            self.stage_completion[stage] = True
        self.updated_at = time.time()

    def mark_complete(self) -> None:
        self.completed = True
        for stage in self.stages:
            self.stage_completion[stage] = True
        self.updated_at = time.time()
        if self._tracker is not None:
            self._tracker.flush()

    def processed_count(self) -> int:
        if self._tracker is not None:
            return self._tracker.count
        return self._ensure_tracker().count

    def is_processed(self, index: int) -> bool:
        tracker = self._tracker or self._ensure_tracker()
        return tracker.is_marked(index)

    def mark_unprocessed(self, index: int) -> None:
        tracker = self._tracker or self._ensure_tracker()
        removed = tracker.clear(index)
        if not removed:
            return
        self.updated_at = time.time()

    def add_elapsed(self, seconds: float) -> float:
        self.elapsed_seconds = float(self.elapsed_seconds) + float(seconds)
        self.updated_at = time.time()
        return self.elapsed_seconds

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        payload = json.loads(Path(path).read_text())
        return cls(
            path=Path(path),
            experiment_root=Path(payload["experiment_root"]),
            output_dir=Path(payload["output_dir"]),
            manifest_path=Path(payload["manifest_path"]),
            config_digest=payload["config_digest"],
            manifest_digest=payload["manifest_digest"],
            stages=list(payload.get("stages") or []),
            stage_completion=payload.get("stage_completion", {}),
            order=payload.get("order"),
            reference=payload.get("reference", {}),
            filters=payload.get("filters", {}),
            total_samples=int(payload["total_samples"]),
            tracker_path=Path(payload["tracker_path"])
            if payload.get("tracker_path")
            else None,
            runtime_settings=payload.get("runtime_settings", {}),
            metadata=payload.get("metadata", {}),
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
            completed=bool(payload.get("completed", False)),
            created_at=float(payload.get("created_at", time.time())),
            updated_at=float(payload.get("updated_at", time.time())),
        )


def compute_config_digest(payload: Mapping[str, Any]) -> str:
    """Return a deterministic hash for the resolved config payload."""

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
