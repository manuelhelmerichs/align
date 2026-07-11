"""Shared stage-pipeline execution for local and worker runtimes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..sample_manifest import SampleRecord
from ..samples import WeightSample
from .loaders import PrefetchingLoader, SampleLoader
from .stages import StageExecutor


@dataclass(frozen=True)
class SampleAlignmentResult:
    """Complete output for one sample after all configured stages."""

    record: SampleRecord
    sample: WeightSample
    diagnostics: dict[str, Any]
    transforms: Mapping[str, Any] | None = None
    scales: Mapping[str, Any] | None = None
    transform_families: Mapping[str, str] | None = None
    stage_outputs: Mapping[str, WeightSample] = field(default_factory=dict)


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

    def can_batch_match(self, batch_size: int) -> bool:
        """Return whether this pipeline can batch the final match stage."""

        if not self.stages or self.stages[-1][0] != "match":
            return False
        match_executor = self.stages[-1][1]
        return match_executor.supports_batching and int(batch_size) > 1

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

        diagnostics_payload: dict[str, Any] = {}
        current = sample
        transforms = None
        scales = None
        transform_families = None
        stage_outputs: dict[str, WeightSample] = {}

        for idx, name, executor in stages:
            last_stage = idx == len(self.stages) - 1
            result = executor.process_single(record, current)
            current = result.sample

            if result.diagnostics:
                diagnostics_payload[name] = result.diagnostics
            if name == "canonicalize":
                scales = result.scales
            elif name == "match":
                transforms = result.transforms
                if result.diagnostics:
                    transform_families = result.diagnostics.get("transform_families")

            if self.save_intermediate and not last_stage:
                stage_outputs[name] = current

        return SampleAlignmentResult(
            record=record,
            sample=current,
            diagnostics=diagnostics_payload,
            transforms=transforms,
            scales=scales,
            transform_families=transform_families,
            stage_outputs=stage_outputs,
        )

    def process_batch(
        self,
        records: Sequence[SampleRecord],
        samples: Sequence[WeightSample],
    ) -> list[SampleAlignmentResult]:
        """Run a batch whose final stage is a batchable matching executor."""

        if not self.stages or self.stages[-1][0] != "match":
            raise ValueError("Batched execution requires match as the final stage.")

        match_executor = self.stages[-1][1]
        if not match_executor.supports_batching:
            raise ValueError("Final match executor does not support batching.")

        pre_stages = [
            (idx, name, ex) for idx, (name, ex) in enumerate(self.stages[:-1])
        ]
        partials = [
            self._run_stages(record, sample, pre_stages)
            for record, sample in zip(records, samples, strict=True)
        ]

        match_results = match_executor.process_batch(
            records, [partial.sample for partial in partials]
        )
        outputs: list[SampleAlignmentResult] = []
        for partial, match_result in zip(partials, match_results, strict=True):
            diagnostics_payload = dict(partial.diagnostics)
            if match_result.diagnostics:
                diagnostics_payload["match"] = match_result.diagnostics
            outputs.append(
                SampleAlignmentResult(
                    record=partial.record,
                    sample=match_result.sample,
                    diagnostics=diagnostics_payload,
                    transforms=match_result.transforms,
                    scales=partial.scales,
                    transform_families=(
                        match_result.diagnostics.get("transform_families")
                        if match_result.diagnostics
                        else None
                    ),
                    stage_outputs=partial.stage_outputs,
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
        """Load and run records sequentially or with a batched final matching."""

        if self.can_batch_match(batch_size):
            self._run_records_batched(
                records,
                loader,
                sink,
                batch_size=max(1, int(batch_size)),
                on_record_start=on_record_start,
            )
            return
        with PrefetchingLoader(loader, prefetch_count=1) as prefetch_loader:
            if records:
                prefetch_loader.prefetch(records[:1])
            for index, record in enumerate(records):
                if index + 1 < len(records):
                    prefetch_loader.prefetch(records[index + 1 : index + 2])
                if on_record_start is not None:
                    on_record_start(record)
                self.run(record, prefetch_loader.get(record), sink)

    def _run_records_batched(
        self,
        records: Sequence[SampleRecord],
        loader: SampleLoader,
        sink: PipelineSink,
        *,
        batch_size: int,
        on_record_start: Callable[[SampleRecord], None] | None,
    ) -> None:
        with PrefetchingLoader(loader, prefetch_count=batch_size) as prefetch_loader:
            if records:
                prefetch_loader.prefetch(records[:batch_size])
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


__all__ = ["SampleAlignmentResult", "PipelineSink", "StagePipeline"]
