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
class PipelineOutput:
    """Complete output for one sample after all configured stages."""

    record: SampleRecord
    sample: WeightSample
    aux: dict[str, Any]
    permutations: Any = None
    scale_factors: Any = None
    intermediate_sample: WeightSample | None = None


class PipelineSink(Protocol):
    """Destination for completed pipeline outputs."""

    def write(self, output: PipelineOutput) -> None:
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
    ) -> PipelineOutput:
        """Run all stages for one record and return the completed output."""

        aux_payload: dict[str, Any] = {}
        current = sample
        permutations = None
        scale_factors = None
        intermediate_sample = None

        for idx, (stage, executor) in enumerate(self.stages):
            last_stage = idx == len(self.stages) - 1
            result = executor.process_single(record, current)
            current = result.sample

            if result.aux:
                aux_payload[stage] = result.aux
            if stage == "normalize":
                scale_factors = result.artifacts.get("scale_factors")
            elif stage == "rebasin":
                permutations = result.artifacts.get("permutations")

            if self.save_intermediate and not last_stage:
                intermediate_sample = current

        return PipelineOutput(
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
    ) -> list[PipelineOutput]:
        """Run a batch whose final stage is a batchable rebasin executor."""

        if not self.stages or self.stages[-1][0] != "rebasin":
            raise ValueError("Batched execution requires rebasin as the final stage.")

        rebasin_executor = self.stages[-1][1]
        if not rebasin_executor.supports_batching:
            raise ValueError("Final rebasin executor does not support batching.")
        sample_batch = []
        meta: list[dict[str, Any]] = []

        for record, sample in zip(records, samples, strict=True):
            current = sample
            aux_payload: dict[str, Any] = {}
            scale_factors = None
            intermediate_sample = None

            for idx, (stage, executor) in enumerate(self.stages):
                last_stage = idx == len(self.stages) - 1
                if stage == "rebasin" and last_stage:
                    break
                result = executor.process_single(record, current)
                current = result.sample
                if result.aux:
                    aux_payload[stage] = result.aux
                if stage == "normalize":
                    scale_factors = result.artifacts.get("scale_factors")
                if self.save_intermediate and not last_stage:
                    intermediate_sample = current

            sample_batch.append(current)
            meta.append(
                {
                    "record": record,
                    "aux": aux_payload,
                    "scale_factors": scale_factors,
                    "intermediate_sample": intermediate_sample,
                }
            )

        rebasin_results = rebasin_executor.process_batch(records, sample_batch)
        outputs: list[PipelineOutput] = []
        for meta_entry, reb_result in zip(meta, rebasin_results, strict=True):
            aux_payload = dict(meta_entry["aux"])
            if reb_result.aux:
                aux_payload["rebasin"] = reb_result.aux
            outputs.append(
                PipelineOutput(
                    record=meta_entry["record"],
                    sample=reb_result.sample,
                    aux=aux_payload,
                    permutations=reb_result.artifacts.get("permutations"),
                    scale_factors=meta_entry["scale_factors"],
                    intermediate_sample=meta_entry["intermediate_sample"],
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
                    if on_record_start is not None:
                        on_record_start(output.record)
                    sink.write(output)
        finally:
            prefetch_loader.clear()


__all__ = ["PipelineOutput", "PipelineSink", "StagePipeline"]
