"""Immutable sample discovery and manifest persistence."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import shutil
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as np_format

_INT_PATTERN = re.compile(r"(\d+)(?!.*\d)")
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
        raise ValueError(f"Unable to parse integer from {text!r}.")
    return int(match.group(1))


def _iter_chains(samples_dir: Path) -> Iterator[tuple[int, Path]]:
    for child in sorted(samples_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            yield _extract_last_int(child.name), child
        except ValueError:
            continue


def _iter_samples(chain_dir: Path) -> Iterator[tuple[int, Path]]:
    files = sorted(
        [path for path in chain_dir.iterdir() if path.suffix == ".npz"],
        key=lambda path: _extract_last_int(path.stem),
    )
    for file_path in files:
        yield _extract_last_int(file_path.stem), file_path


@dataclass(frozen=True)
class SampleRecord:
    """Metadata for one posterior sample file."""

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
    """Sequence backed by read-only memmaps for process sharing."""

    def __init__(self, records_path: Path, paths_path: Path) -> None:
        self._records = np.load(records_path, mmap_mode="r")
        self._length = int(self._records.shape[0])
        self._paths_handle = paths_path.open("rb")
        size = os.path.getsize(paths_path)
        self._paths_map = (
            mmap.mmap(self._paths_handle.fileno(), 0, access=mmap.ACCESS_READ)
            if size > 0
            else None
        )
        self._cache: dict[int, SampleRecord] = {}

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self._length))]
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError("SampleRecord index out of range")
        if index in self._cache:
            return self._cache[index]
        entry = self._records[index]
        offset = int(entry["path_offset"])
        length = int(entry["path_length"])
        if length == 0:
            relative_path = ""
        elif self._paths_map is None:
            raise ValueError("Manifest path blob missing.")
        else:
            relative_path = self._paths_map[offset : offset + length].decode("utf-8")
        record = SampleRecord(
            index=int(entry["index"]),
            chain_id=int(entry["chain_id"]),
            sample_id=int(entry["sample_id"]),
            relative_path=relative_path,
        )
        self._cache[index] = record
        return record

    def __iter__(self):
        for index in range(self._length):
            yield self[index]

    def __del__(self):  # pragma: no cover - relies on GC timing
        try:
            if getattr(self, "_paths_map", None) is not None:
                self._paths_map.close()
        finally:
            try:
                self._paths_handle.close()
            except Exception:
                pass


@dataclass
class SampleManifest:
    """Immutable list of all samples participating in one alignment run."""

    samples_dir: Path
    tree_path: Path
    records: Sequence[SampleRecord]
    reference_chain: int
    reference_sample: int
    reference_index: int
    filters: Mapping[str, Any]
    created_at: float = field(default_factory=time.time)

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
            "records": [record.to_dict() for record in self.records],
            "reference_chain": self.reference_chain,
            "reference_sample": self.reference_sample,
            "reference_index": self.reference_index,
            "filters": dict(self.filters),
            "created_at": self.created_at,
            "digest": self.digest(),
        }

    def digest(self) -> str:
        payload = {
            "records": [record.relative_path for record in self.records],
            "reference_chain": self.reference_chain,
            "reference_sample": self.reference_sample,
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
        return {
            "chains": sorted({record.chain_id for record in self.records}),
            "total_samples": self.total,
            "reference": {
                "chain": self.reference_chain,
                "sample": self.reference_sample,
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
        if (
            not _records_array_path(manifest_path).exists()
            or not _records_blob_path(manifest_path).exists()
        ):
            self._write_shared_records(manifest_path)
        self._attach_shared_records(manifest_path)

    def ensure_shared_tree(self, manifest_path: Path) -> Path:
        manifest_path = Path(manifest_path).resolve()
        shared_tree = _read_tree_override(manifest_path, discard_stale=True)
        if shared_tree is not None:
            self.tree_path = shared_tree
            return shared_tree

        override_path = _tree_override_path(manifest_path)
        copy_path = _tree_copy_path(manifest_path).resolve()
        source = Path(self.tree_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Tree path not found: {source}")
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
        mmap_array = np_format.open_memmap(
            records_path, mode="w+", dtype=_RECORD_DTYPE, shape=(self.total,)
        )
        path_blob = bytearray()
        for index, record in enumerate(self.records):
            encoded = record.relative_path.encode("utf-8")
            mmap_array[index] = (
                int(record.index),
                int(record.chain_id),
                int(record.sample_id),
                len(path_blob),
                len(encoded),
            )
            path_blob.extend(encoded)
        mmap_array.flush()
        del mmap_array
        paths_path.write_bytes(path_blob)

    def _attach_shared_records(self, manifest_path: Path) -> None:
        records_path = _records_array_path(Path(manifest_path).resolve())
        paths_path = _records_blob_path(Path(manifest_path).resolve())
        if records_path.exists() and paths_path.exists():
            self.records = _SharedSampleRecordList(records_path, paths_path)

    def _apply_shared_tree_override(self, manifest_path: Path) -> None:
        shared_tree = _read_tree_override(manifest_path, discard_stale=False)
        if shared_tree is not None:
            self.tree_path = shared_tree

    @classmethod
    def load(cls, path: Path) -> SampleManifest:
        payload = json.loads(Path(path).read_text())
        manifest = cls(
            samples_dir=Path(payload["samples_dir"]).resolve(),
            tree_path=Path(payload["tree_path"]).resolve(),
            records=[SampleRecord.from_dict(item) for item in payload["records"]],
            reference_chain=int(payload["reference_chain"]),
            reference_sample=int(payload["reference_sample"]),
            reference_index=int(payload["reference_index"]),
            filters=payload.get("filters", {}),
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
        reference_chain: int = 0,
        reference_sample: int = 0,
    ) -> SampleManifest:
        samples_dir = Path(samples_dir).resolve()
        tree_path = Path(tree_path).resolve()
        chain_filter = set(chain_indices) if chain_indices is not None else None
        sample_step = max(sample_step, 1)

        records: list[SampleRecord] = []
        reference_index: int | None = None
        for chain_id, chain_dir in _iter_chains(samples_dir):
            if chain_filter is not None and chain_id not in chain_filter:
                continue
            loaded = 0
            for offset, (sample_id, file_path) in enumerate(_iter_samples(chain_dir)):
                if offset % sample_step != 0:
                    continue
                record = SampleRecord(
                    index=len(records),
                    chain_id=chain_id,
                    sample_id=sample_id,
                    relative_path=file_path.relative_to(samples_dir).as_posix(),
                )
                records.append(record)
                if chain_id == reference_chain and sample_id == reference_sample:
                    reference_index = record.index
                loaded += 1
                if max_total is not None and len(records) >= max_total:
                    break
                if samples_per_chain is not None and loaded >= samples_per_chain:
                    break
            if max_total is not None and len(records) >= max_total:
                break

        if not records:
            raise ValueError("No samples were discovered. Check the directory filters.")
        if reference_index is None:
            raise ValueError(
                "Reference sample was filtered out by the current selection options."
            )
        return cls(
            samples_dir=samples_dir,
            tree_path=tree_path,
            records=records,
            reference_chain=reference_chain,
            reference_sample=reference_sample,
            reference_index=reference_index,
            filters={
                "chain_indices": sorted(chain_filter) if chain_filter else None,
                "samples_per_chain": samples_per_chain,
                "sample_step": sample_step,
                "max_total": max_total,
            },
        )


__all__ = ["SampleManifest", "SampleRecord"]
