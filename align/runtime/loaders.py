"""Sample loading utilities."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from ..sample_manifest import SampleManifest, SampleRecord
from ..samples import PyTreeNpzCodec, WeightSample


class SampleLoader:
    """Lazy loader that materializes canonical weight samples on demand."""

    def __init__(self, manifest: SampleManifest):
        self.manifest = manifest
        self.codec = PyTreeNpzCodec(Path(manifest.tree_path))

    def load(self, record: SampleRecord) -> WeightSample:
        abs_path = record.absolute_path(self.manifest.samples_dir)
        return self.codec.load(abs_path)

    def load_reference(self) -> WeightSample:
        return self.load(self.manifest.reference_record)


class PrefetchingLoader:
    """Sample loader with background prefetching for I/O overlap.

    This loader prefetches the next batch of samples while the current batch
    is being processed, overlapping I/O with computation.

    Usage:
        loader = PrefetchingLoader(base_loader, prefetch_count=2)
        for batch in batches:
            # Start prefetching next batch
            loader.prefetch(next_batch_records)
            # Process current batch using get() which may return prefetched data
            for record in batch:
                params = loader.get(record)
    """

    def __init__(self, base_loader: SampleLoader, prefetch_count: int = 2):
        self.base_loader = base_loader
        self.prefetch_count = max(1, prefetch_count)
        self._prefetch_futures: dict[int, Future[WeightSample]] = {}
        self._executor = ThreadPoolExecutor(max_workers=min(2, self.prefetch_count))
        self._closed = False

    def prefetch(self, records: Sequence[SampleRecord]) -> None:
        """Start background loading for the given records.

        Only prefetches up to `prefetch_count` samples to limit memory usage.
        """
        active_count = sum(
            1 for fut in self._prefetch_futures.values() if not fut.done()
        )
        for record in records[: self.prefetch_count - active_count]:
            if record.index not in self._prefetch_futures:
                future = self._executor.submit(self.base_loader.load, record)
                self._prefetch_futures[record.index] = future

    def get(self, record: SampleRecord) -> WeightSample:
        """Get sample data, using prefetched result if available."""
        if record.index in self._prefetch_futures:
            future = self._prefetch_futures.pop(record.index)
            return future.result()
        return self.base_loader.load(record)

    def clear(self) -> None:
        """Cancel unused work and release the owned executor."""

        if self._closed:
            return
        for future in self._prefetch_futures.values():
            future.cancel()
        self._prefetch_futures.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def __enter__(self) -> PrefetchingLoader:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.clear()


__all__ = ["SampleLoader", "PrefetchingLoader"]
