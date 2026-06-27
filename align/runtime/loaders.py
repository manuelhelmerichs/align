"""Sample loading utilities."""

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from ..samples import WeightSample, create_sample_codec
from ..state import SampleManifest, SampleRecord

_PREFETCH_EXECUTOR: ThreadPoolExecutor | None = None
_PREFETCH_WORKERS = 2


def _get_prefetch_executor() -> ThreadPoolExecutor:
    global _PREFETCH_EXECUTOR
    if _PREFETCH_EXECUTOR is None:
        _PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS)
    return _PREFETCH_EXECUTOR


class SampleLoader:
    """Lazy loader that materializes canonical weight samples on demand."""

    def __init__(self, manifest: SampleManifest, tree_path: Path | None = None):
        self.manifest = manifest
        self.tree_path = Path(tree_path or manifest.tree_path)
        self.codec = create_sample_codec(
            manifest.sample_format,
            tree_path=self.tree_path,
        )

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
        self._executor = _get_prefetch_executor()

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
        """Clear all prefetch futures (waits for completion)."""
        for future in self._prefetch_futures.values():
            try:
                future.result(timeout=1.0)
            except Exception:
                pass
        self._prefetch_futures.clear()


__all__ = ["SampleLoader", "PrefetchingLoader"]
