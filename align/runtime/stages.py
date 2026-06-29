"""Stage executor abstractions for the align runner."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax

from ..architectures import get_adapter
from ..config.stages import NormalizeConfig, RebasinConfig
from ..rebasin import (
    build_scheduler,
    rebasin_batch,
    rebasin_single_sample,
)
from ..samples import ParamTree, WeightSample
from ..state import SampleManifest, SampleRecord

_LOG = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Container for per-stage outputs."""

    sample: WeightSample
    artifacts: dict[str, Any]
    aux: dict[str, Any] | None = None

    @property
    def params(self):
        """Convenience access to the canonical parameter PyTree."""

        return self.sample.params


class StageExecutor(ABC):
    """Abstract interface for pipeline stages (normalize, rebasin)."""

    @abstractmethod
    def prepare(self, manifest: SampleManifest, ref_sample: WeightSample) -> None:
        """Perform any one-time setup using the reference sample."""

    @abstractmethod
    def process_single(self, record: SampleRecord, sample: WeightSample) -> StageResult:
        """Run this stage for a single sample."""

    @abstractmethod
    def process_batch(
        self,
        records: Sequence[SampleRecord],
        samples: Sequence[WeightSample],
    ) -> list[StageResult]:
        """Run this stage for a batch of samples."""

    @property
    @abstractmethod
    def supports_batching(self) -> bool:
        """Whether this stage benefits from batched execution."""

    @property
    @abstractmethod
    def prefers_gpu(self) -> bool:
        """Whether this stage prefers GPU execution."""

    def reference_output(
        self, record: SampleRecord, sample: WeightSample
    ) -> WeightSample:
        """Return the reference sample after applying this stage.

        Each stage transforms the reference so the next stage prepares against
        the already-aligned reference.
        """
        return self.process_single(record, sample).sample


class NormalizeExecutor(StageExecutor):
    """Executes the normalization stage."""

    def __init__(
        self,
        config: NormalizeConfig,
        *,
        architecture: str,
        adapter_kwargs: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.architecture = architecture
        self.adapter_kwargs = dict(adapter_kwargs)
        self.adapter = None
        self.problem = None
        self.normalizer = None

    def prepare(self, manifest: SampleManifest, ref_sample: WeightSample) -> None:
        from ..normalization import ScaleNormalizer

        adapter_kwargs = dict(self.adapter_kwargs)
        if self.config.layer_root:
            adapter_kwargs.setdefault("layer_root", self.config.layer_root)
        self.adapter = get_adapter(self.architecture, **adapter_kwargs)
        self.problem = self.adapter.build_problem(ref_sample.params)
        self.normalizer = ScaleNormalizer()

    def process_single(self, record: SampleRecord, sample: WeightSample) -> StageResult:
        if self.adapter is None or self.problem is None or self.normalizer is None:
            raise RuntimeError("NormalizeExecutor not prepared.")

        normalized_params, scale_factors, aux = self.normalizer.normalize(
            self.problem,
            sample.params,
            task_type=self.config.task_type,
            num_classes=self.config.num_classes,
            **self.config.method_kwargs,
        )
        aux_payload = dict(aux or {})
        aux_payload["method"] = self.config.method
        return StageResult(
            sample=sample.with_params(normalized_params),
            artifacts={"scale_factors": scale_factors},
            aux=aux_payload,
        )

    def process_batch(
        self,
        records: Sequence[SampleRecord],
        samples: Sequence[WeightSample],
    ) -> list[StageResult]:
        return [
            self.process_single(record, sample)
            for record, sample in zip(records, samples, strict=True)
        ]

    @property
    def supports_batching(self) -> bool:
        return False

    @property
    def prefers_gpu(self) -> bool:
        return False


class RebasinExecutor(StageExecutor):
    """Executes the rebasin stage."""

    def __init__(
        self,
        config: RebasinConfig,
        *,
        reference_index: int,
        seed: int | None = None,
        batch_size: int = 1,
        architecture: str,
        adapter_kwargs: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.reference_index = int(reference_index)
        self._base_key = jax.random.PRNGKey(seed) if seed is not None else None
        self.batch_size = max(1, int(batch_size))

        self.adapter = None
        self.problem = None
        self.ref_sample: WeightSample | None = None
        self.ref_data = None
        self.ref_backend = None
        self.scheduler = None
        self.architecture = architecture
        self.adapter_kwargs = dict(adapter_kwargs)

    def prepare(self, manifest: SampleManifest, ref_sample: WeightSample) -> None:
        adapter_kwargs = dict(self.adapter_kwargs)
        if self.config.layer_root:
            adapter_kwargs.setdefault("layer_root", self.config.layer_root)
        self.adapter = get_adapter(self.architecture, **adapter_kwargs)
        self.problem = self.adapter.build_problem(ref_sample.params)
        self.ref_sample = ref_sample
        self.scheduler = build_scheduler(
            objective=self.config.objective,
            objective_kwargs=self.config.objective_kwargs,
            schedule=self.config.schedule,
        )
        backend = self.scheduler.backend
        self.ref_backend = backend
        self.ref_data = self.problem.materialize(
            ref_sample.params, backend=backend, cache=True
        )

    def _rng_for_record(self, record: SampleRecord) -> jax.Array | None:
        if self._base_key is None:
            return None
        return jax.random.fold_in(self._base_key, int(record.index))

    def _stage_result(
        self,
        sample: WeightSample,
        params: ParamTree,
        perms: Mapping[str, Any],
        aux: dict[str, Any] | None,
    ) -> StageResult:
        aux_payload = dict(aux or {})
        aux_payload["objective"] = self.config.objective
        aux_payload["schedule"] = self.config.schedule_payload()
        return StageResult(
            sample=sample.with_params(params),
            artifacts={"permutations": perms},
            aux=aux_payload,
        )

    def process_single(self, record: SampleRecord, sample: WeightSample) -> StageResult:
        if (
            self.adapter is None
            or self.problem is None
            or self.ref_sample is None
            or self.scheduler is None
            or self.ref_data is None
        ):
            raise RuntimeError("RebasinExecutor not prepared.")

        folded_params, perms, aux_info = rebasin_single_sample(
            self.problem,
            self.ref_sample.params,
            sample.params,
            scheduler=self.scheduler,
            rng_key=self._rng_for_record(record),
            ref_data=self.ref_data,
            ref_backend=self.ref_backend,
            is_reference=record.index == self.reference_index,
        )
        return self._stage_result(sample, folded_params, perms, aux_info)

    def process_batch(
        self,
        records: Sequence[SampleRecord],
        samples: Sequence[WeightSample],
    ) -> list[StageResult]:
        if (
            self.adapter is None
            or self.problem is None
            or self.ref_sample is None
            or self.scheduler is None
            or self.ref_data is None
        ):
            raise RuntimeError("RebasinExecutor not prepared.")

        record_list = list(records)
        sample_batch = list(samples)
        if not record_list:
            return []

        results: list[StageResult | None] = [None] * len(record_list)

        ref_positions = [
            idx
            for idx, rec in enumerate(record_list)
            if rec.index == self.reference_index
        ]
        for pos in ref_positions:
            results[pos] = self.process_single(record_list[pos], sample_batch[pos])

        non_ref_positions = [
            idx for idx in range(len(record_list)) if results[idx] is None
        ]
        if non_ref_positions:
            if not self.scheduler.supports_batching:
                for pos in non_ref_positions:
                    results[pos] = self.process_single(
                        record_list[pos], sample_batch[pos]
                    )
                return [res for res in results if res is not None]

            target_records = [record_list[idx] for idx in non_ref_positions]
            target_samples = [sample_batch[idx] for idx in non_ref_positions]
            target_params = [sample.params for sample in target_samples]
            rng_keys = [self._rng_for_record(rec) for rec in target_records]
            rng_key = rng_keys[0] if rng_keys else None
            batch_results = rebasin_batch(
                self.problem,
                self.ref_sample.params,
                target_params,
                scheduler=self.scheduler,
                ref_data=self.ref_data,
                ref_backend=self.ref_backend,
                rng_key=rng_key,
            )
            for pos, result in zip(non_ref_positions, batch_results, strict=True):
                folded_params, perms, aux = result
                results[pos] = self._stage_result(
                    sample_batch[pos], folded_params, perms, aux
                )

        return [res for res in results if res is not None]

    @property
    def supports_batching(self) -> bool:
        if self.scheduler is None:
            return False
        return bool(self.scheduler.supports_batching)

    @property
    def prefers_gpu(self) -> bool:
        if self.scheduler is None:
            return any(step.solver == "sinkhorn" for step in self.config.schedule)
        return bool(self.scheduler.prefers_gpu)


__all__ = [
    "StageResult",
    "StageExecutor",
    "NormalizeExecutor",
    "RebasinExecutor",
]
