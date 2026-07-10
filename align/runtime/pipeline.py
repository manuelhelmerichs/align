"""Shared stage-pipeline execution for local and worker runtimes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..samples import WeightSample
from ..state import SampleRecord
from .loaders import PrefetchingLoader, SampleLoader
from .stages import StageExecutor


@dataclass(frozen=True)
class SampleAlignmentResult:
    """Complete output for one sample after all configured stages."""

    record: SampleRecord
    sample: WeightSample
    aux: dict[str, Any]
    permutations: Any = None
    scale_factors: Any = None
    intermediate_sample: WeightSample | None = None


class PipelineSink(Protocol):
    """Destination for completed pipeline outputs."""

    def write(self, output: SampleAlignmentResult) -> None:
        """Persist or emit one completed sample."""


class StagePipeline:
    """Execute configured stages once, shared by runner and worker paths."""

    def __init__(
        self,
        stages: Sequence[tuple[str, StageExecutor]],
        *,
        save_intermediate: bool,
    ) -> None:
        self.stages = list(stages)
        self.save_intermediate = bool(save_intermediate)

    def can_batch_rebasin(self, batch_size: int) -> bool:
        """Return whether this pipeline can batch the final rebasin stage."""

        if not self.stages or self.stages[-1][0] != "rebasin":
            return False
        rebasin_executor = self.stages[-1][1]
        return rebasin_executor.supports_batching and int(batch_size) > 1

    def run(
        self,
        record: SampleRecord,
        sample: WeightSample,
        sink: PipelineSink,
    ) -> None:
        """Run all stages for one record and send the result to ``sink``."""

        sink.write(self.process_single(record, sample))

    def process_single(
        self,
        record: SampleRecord,
        sample: WeightSample,
    ) -> SampleAlignmentResult:
        """Run all stages for one record and return the completed output."""

        indexed = [(idx, name, ex) for idx, (name, ex) in enumerate(self.stages)]
        return self._run_stages(record, sample, indexed)

    def _run_stages(
        self,
        record: SampleRecord,
        sample: WeightSample,
        stages: Sequence[tuple[int, str, StageExecutor]],
    ) -> SampleAlignmentResult:
        """Thread ``sample`` through ``stages``, collecting artifacts and aux.

        ``stages`` carries each executor's index within the full pipeline so the
        ``save_intermediate`` (skip the final stage) bookkeeping is identical for
        the single and batched paths.
        """

        aux_payload: dict[str, Any] = {}
        current = sample
        permutations = None
        scale_factors = None
        intermediate_sample = None

        for idx, name, executor in stages:
            last_stage = idx == len(self.stages) - 1
            result = executor.process_single(record, current)
            current = result.sample

            if result.aux:
                aux_payload[name] = result.aux
            if name == "normalize":
                scale_factors = result.artifacts.get("scale_factors")
            elif name == "rebasin":
                permutations = result.artifacts.get("permutations")

            if self.save_intermediate and not last_stage:
                intermediate_sample = current

        return SampleAlignmentResult(
            record=record,
            sample=current,
            aux=aux_payload,
            permutations=permutations,
            scale_factors=scale_factors,
            intermediate_sample=intermediate_sample,
        )

    def process_batch(
        self,
        records: Sequence[SampleRecord],
        samples: Sequence[WeightSample],
    ) -> list[SampleAlignmentResult]:
        """Run a batch whose final stage is a batchable rebasin executor."""

        if not self.stages or self.stages[-1][0] != "rebasin":
            raise ValueError("Batched execution requires rebasin as the final stage.")

        rebasin_executor = self.stages[-1][1]
        if not rebasin_executor.supports_batching:
            raise ValueError("Final rebasin executor does not support batching.")

        pre_stages = [
            (idx, name, ex) for idx, (name, ex) in enumerate(self.stages[:-1])
        ]
        partials = [
            self._run_stages(record, sample, pre_stages)
            for record, sample in zip(records, samples, strict=True)
        ]

        rebasin_results = rebasin_executor.process_batch(
            records, [partial.sample for partial in partials]
        )
        outputs: list[SampleAlignmentResult] = []
        for partial, reb_result in zip(partials, rebasin_results, strict=True):
            aux_payload = dict(partial.aux)
            if reb_result.aux:
                aux_payload["rebasin"] = reb_result.aux
            outputs.append(
                SampleAlignmentResult(
                    record=partial.record,
                    sample=reb_result.sample,
                    aux=aux_payload,
                    permutations=reb_result.artifacts.get("permutations"),
                    scale_factors=partial.scale_factors,
                    intermediate_sample=partial.intermediate_sample,
                )
            )
        return outputs

    def run_records(
        self,
        records: Sequence[SampleRecord],
        loader: SampleLoader,
        sink: PipelineSink,
        *,
        batch_size: int,
        on_record_start: Callable[[SampleRecord], None] | None = None,
    ) -> None:
        """Load and run records sequentially or with a batched final rebasin."""

        if self.can_batch_rebasin(batch_size):
            self._run_records_batched(
                records,
                loader,
                sink,
                batch_size=max(1, int(batch_size)),
                on_record_start=on_record_start,
            )
            return

        for record in records:
            if on_record_start is not None:
                on_record_start(record)
            self.run(record, loader.load(record), sink)

    def _run_records_batched(
        self,
        records: Sequence[SampleRecord],
        loader: SampleLoader,
        sink: PipelineSink,
        *,
        batch_size: int,
        on_record_start: Callable[[SampleRecord], None] | None,
    ) -> None:
        prefetch_loader = PrefetchingLoader(loader, prefetch_count=batch_size)
        try:
            for batch_start in range(0, len(records), batch_size):
                batch_records = list(records[batch_start : batch_start + batch_size])
                next_start = batch_start + batch_size
                if next_start < len(records):
                    prefetch_loader.prefetch(
                        records[next_start : next_start + batch_size]
                    )

                sample_batch = []
                for record in batch_records:
                    if on_record_start is not None:
                        on_record_start(record)
                    sample_batch.append(prefetch_loader.get(record))

                for output in self.process_batch(batch_records, sample_batch):
                    sink.write(output)
        finally:
            prefetch_loader.clear()


__all__ = ["SampleAlignmentResult", "PipelineSink", "StagePipeline"]
