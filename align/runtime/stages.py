"""Stage executor abstractions for the align runner."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax

from ..architectures import get_recipe
from ..config.stages import CanonicalizeConfig, MatchConfig
from ..matching import (
    build_solver_sequence,
    match_batch,
    match_sample,
)
from ..sample_manifest import SampleManifest, SampleRecord
from ..samples import ParamTree, WeightSample

_LOG = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Container for per-stage outputs."""

    sample: WeightSample
    transforms: Mapping[str, Any] | None = None
    scales: Mapping[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None

    @property
    def params(self):
        """Convenience access to the canonical parameter PyTree."""

        return self.sample.params


class StageExecutor(ABC):
    """Abstract interface for pipeline stages (canonicalize, match)."""

    @abstractmethod
    def prepare(self, manifest: SampleManifest, reference_sample: WeightSample) -> None:
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


class CanonicalizeExecutor(StageExecutor):
    """Executes the canonicalize stage."""

    def __init__(
        self,
        config: CanonicalizeConfig,
        *,
        family: str,
        recipe_kwargs: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.family = family
        self.recipe_kwargs = dict(recipe_kwargs)
        self.recipe = None
        self.graph = None
        self.canonicalizer = None

    def prepare(self, manifest: SampleManifest, reference_sample: WeightSample) -> None:
        from ..canonicalization import ScaleCanonicalizer

        self.recipe = get_recipe(self.family, **self.recipe_kwargs)
        self.graph = self.recipe.build_graph(reference_sample.params)
        self.canonicalizer = ScaleCanonicalizer()

    def process_single(self, record: SampleRecord, sample: WeightSample) -> StageResult:
        if self.recipe is None or self.graph is None or self.canonicalizer is None:
            raise RuntimeError("CanonicalizeExecutor not prepared.")

        canonical_params, scales, diagnostics = self.canonicalizer.canonicalize(
            self.graph,
            sample.params,
            **self.config.canonicalizer_kwargs,
        )
        return StageResult(
            sample=sample.with_params(canonical_params),
            scales=scales,
            diagnostics=dict(diagnostics or {}),
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


class MatchExecutor(StageExecutor):
    """Executes the match stage."""

    def __init__(
        self,
        config: MatchConfig,
        *,
        reference_index: int | None,
        seed: int | None = None,
        rng_offset: int = 0,
        batch_size: int = 1,
        family: str,
        recipe_kwargs: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.reference_index = (
            int(reference_index) if reference_index is not None else None
        )
        self._base_key = jax.random.PRNGKey(seed) if seed is not None else None
        self.rng_offset = int(rng_offset)
        self.batch_size = max(1, int(batch_size))

        self.recipe = None
        self.graph = None
        self.reference_sample: WeightSample | None = None
        self.reference_data = None
        self.reference_backend = None
        self.solver_sequence = None
        self.family = family
        self.recipe_kwargs = dict(recipe_kwargs)

    def prepare(self, manifest: SampleManifest, reference_sample: WeightSample) -> None:
        self.recipe = get_recipe(self.family, **self.recipe_kwargs)
        self.graph = self.recipe.build_graph(reference_sample.params)
        self.reference_sample = reference_sample
        self.solver_sequence = build_solver_sequence(
            objective=self.config.objective.type,
            objective_kwargs=self.config.objective.kwargs,
            schedule=self.config.solvers,
        )
        backend = self.solver_sequence.backend
        self.reference_backend = backend
        self.reference_data = self.graph.materialize(
            reference_sample.params, backend=backend, cache=True
        )

    def _rng_for_record(self, record: SampleRecord) -> jax.Array | None:
        if self._base_key is None:
            return None
        return jax.random.fold_in(self._base_key, self.rng_offset + int(record.index))

    def _stage_result(
        self,
        sample: WeightSample,
        params: ParamTree,
        transforms: Mapping[str, Any],
        diagnostics: dict[str, Any] | None,
    ) -> StageResult:
        diagnostics_payload = dict(diagnostics or {})
        diagnostics_payload["objective"] = self.config.objective.type
        diagnostics_payload["solvers"] = self.config.solvers_payload()
        diagnostics_payload["barycenter_passes"] = self.config.barycenter_passes
        diagnostics_payload["transform_families"] = {
            group_id: group.transform_family
            for group_id, group in self.graph.groups.items()
        }
        return StageResult(
            sample=sample.with_params(params),
            transforms=transforms,
            diagnostics=diagnostics_payload,
        )

    def process_single(self, record: SampleRecord, sample: WeightSample) -> StageResult:
        if (
            self.recipe is None
            or self.graph is None
            or self.reference_sample is None
            or self.solver_sequence is None
            or self.reference_data is None
        ):
            raise RuntimeError("MatchExecutor not prepared.")

        aligned_params, transforms, aux_info = match_sample(
            self.graph,
            self.reference_sample.params,
            sample.params,
            solver_sequence=self.solver_sequence,
            rng_key=self._rng_for_record(record),
            reference_data=self.reference_data,
            reference_backend=self.reference_backend,
            is_reference=(
                self.reference_index is not None
                and record.index == self.reference_index
            ),
        )
        return self._stage_result(sample, aligned_params, transforms, aux_info)

    def process_batch(
        self,
        records: Sequence[SampleRecord],
        samples: Sequence[WeightSample],
    ) -> list[StageResult]:
        if (
            self.recipe is None
            or self.graph is None
            or self.reference_sample is None
            or self.solver_sequence is None
            or self.reference_data is None
        ):
            raise RuntimeError("MatchExecutor not prepared.")

        record_list = list(records)
        sample_batch = list(samples)
        if not record_list:
            return []

        results: list[StageResult | None] = [None] * len(record_list)

        reference_positions = [
            idx
            for idx, rec in enumerate(record_list)
            if self.reference_index is not None and rec.index == self.reference_index
        ]
        for pos in reference_positions:
            results[pos] = self.process_single(record_list[pos], sample_batch[pos])

        target_positions = [
            idx for idx in range(len(record_list)) if results[idx] is None
        ]
        if target_positions:
            if not self.solver_sequence.supports_batching:
                for pos in target_positions:
                    results[pos] = self.process_single(
                        record_list[pos], sample_batch[pos]
                    )
                return [res for res in results if res is not None]

            target_records = [record_list[idx] for idx in target_positions]
            target_samples = [sample_batch[idx] for idx in target_positions]
            target_params = [sample.params for sample in target_samples]
            rng_keys = [self._rng_for_record(rec) for rec in target_records]
            rng_key = rng_keys[0] if rng_keys else None
            batch_results = match_batch(
                self.graph,
                self.reference_sample.params,
                target_params,
                solver_sequence=self.solver_sequence,
                reference_data=self.reference_data,
                reference_backend=self.reference_backend,
                rng_key=rng_key,
            )
            for pos, result in zip(target_positions, batch_results, strict=True):
                aligned_params, transforms, aux = result
                results[pos] = self._stage_result(
                    sample_batch[pos], aligned_params, transforms, aux
                )

        return [res for res in results if res is not None]

    @property
    def supports_batching(self) -> bool:
        if self.solver_sequence is None:
            return False
        return bool(self.solver_sequence.supports_batching)

    @property
    def prefers_gpu(self) -> bool:
        if self.solver_sequence is None:
            return any(step.solver == "sinkhorn" for step in self.config.solvers)
        return bool(self.solver_sequence.prefers_gpu)


__all__ = [
    "StageResult",
    "StageExecutor",
    "CanonicalizeExecutor",
    "MatchExecutor",
]
