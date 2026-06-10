"""Stage executor abstractions for the align runner."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..architecture import get_adapter
from ..config.stages import NormalizeConfig, RebasinConfig
from ..rebasin import ParamTree, rebasin_batch, rebasin_single_sample
from ..state import SampleManifest, SampleRecord
from ..strategies import get_strategy

_LOG = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Container for per-stage outputs."""

    params: ParamTree
    artifacts: dict[str, Any]
    aux: dict[str, Any] | None = None


class StageExecutor(ABC):
    """Abstract interface for pipeline stages (normalize, rebasin)."""

    @abstractmethod
    def prepare(self, manifest: SampleManifest, ref_params: ParamTree) -> None:
        """Perform any one-time setup using the reference sample."""

    @abstractmethod
    def process_single(self, record: SampleRecord, params: ParamTree) -> StageResult:
        """Run this stage for a single sample."""

    @abstractmethod
    def process_batch(
        self,
        records: Sequence[SampleRecord],
        params_list: Sequence[ParamTree],
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

    def reference_output(self, record: SampleRecord, params: ParamTree) -> ParamTree:
        """Return reference params after applying this stage (default: no-op)."""
        return params


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
        self.spec = None

    def prepare(self, manifest: SampleManifest, ref_params: ParamTree) -> None:
        adapter_kwargs = dict(self.adapter_kwargs)
        if self.config.layer_root:
            adapter_kwargs.setdefault("layer_root", self.config.layer_root)
        self.adapter = get_adapter(self.architecture, **adapter_kwargs)
        self.spec = self.adapter.build_spec(ref_params)

    def process_single(self, record: SampleRecord, params: ParamTree) -> StageResult:
        if self.adapter is None or self.spec is None:
            raise RuntimeError("NormalizeExecutor not prepared.")

        normalized_params, scale_factors, aux = self.adapter.normalize(
            params,
            self.spec,
            task_type=self.config.task_type,
            num_classes=self.config.num_classes,
            **self.config.method_kwargs,
        )
        aux_payload = dict(aux or {"method": self.config.method})
        aux_payload["method"] = self.config.method
        return StageResult(
            params=normalized_params,
            artifacts={"scale_factors": scale_factors},
            aux=aux_payload,
        )

    def process_batch(
        self,
        records: Sequence[SampleRecord],
        params_list: Sequence[ParamTree],
    ) -> list[StageResult]:
        return [
            self.process_single(record, params)
            for record, params in zip(records, params_list, strict=True)
        ]

    @property
    def supports_batching(self) -> bool:
        return False

    @property
    def prefers_gpu(self) -> bool:
        return False

    def reference_output(self, record: SampleRecord, params: ParamTree) -> ParamTree:
        return self.process_single(record, params).params


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
        self.spec = None
        self.ref_params = None
        self.ref_views = None
        self.ref_backend = None
        self.strategy = None
        self.architecture = architecture
        self.adapter_kwargs = dict(adapter_kwargs)

    def prepare(self, manifest: SampleManifest, ref_params: ParamTree) -> None:
        adapter_kwargs = dict(self.adapter_kwargs)
        if self.config.layer_root:
            adapter_kwargs.setdefault("layer_root", self.config.layer_root)
        self.adapter = get_adapter(self.architecture, **adapter_kwargs)
        self.spec = self.adapter.build_spec(ref_params)
        self.ref_params = ref_params
        self.strategy = get_strategy(self.config.method, **self.config.method_kwargs)
        backend = (
            "numpy" if getattr(self.strategy, "requires_numpy_views", False) else "jax"
        )
        self.ref_backend = backend
        self.ref_views = self.adapter.permutation_views(
            ref_params, self.spec, backend=backend, cache=True
        )
        try:
            self.strategy.warmup(self.spec, self.ref_views, batch_size=self.batch_size)
        except Exception:
            _LOG.debug(
                "Rebasin warmup failed; continuing without warmup.", exc_info=True
            )

    def _rng_for_record(self, record: SampleRecord) -> jax.Array | None:
        if self._base_key is None:
            return None
        return jax.random.fold_in(self._base_key, int(record.index))

    def _stage_result(
        self,
        params: ParamTree,
        perms: list[jnp.ndarray],
        aux: dict[str, Any] | None,
    ) -> StageResult:
        aux_payload = dict(aux or {})
        aux_payload["method"] = self.config.method
        return StageResult(
            params=params, artifacts={"permutations": perms}, aux=aux_payload
        )

    def process_single(self, record: SampleRecord, params: ParamTree) -> StageResult:
        if (
            self.adapter is None
            or self.spec is None
            or self.ref_params is None
            or self.strategy is None
            or self.ref_views is None
        ):
            raise RuntimeError("RebasinExecutor not prepared.")

        folded_params, perms, aux_info = rebasin_single_sample(
            self.adapter,
            self.spec,
            self.ref_params,
            params,
            method=self.config.method,
            method_kwargs=self.config.method_kwargs,
            rng_key=self._rng_for_record(record),
            ref_views=self.ref_views,
            ref_backend=self.ref_backend,
            is_reference=record.index == self.reference_index,
        )
        return self._stage_result(folded_params, perms, aux_info)

    def process_batch(
        self,
        records: Sequence[SampleRecord],
        params_list: Sequence[ParamTree],
    ) -> list[StageResult]:
        if (
            self.adapter is None
            or self.spec is None
            or self.ref_params is None
            or self.strategy is None
            or self.ref_views is None
        ):
            raise RuntimeError("RebasinExecutor not prepared.")

        record_list = list(records)
        params_batch = list(params_list)
        if not record_list:
            return []

        results: list[StageResult | None] = [None] * len(record_list)

        ref_positions = [
            idx
            for idx, rec in enumerate(record_list)
            if rec.index == self.reference_index
        ]
        for pos in ref_positions:
            results[pos] = self.process_single(record_list[pos], params_batch[pos])

        non_ref_positions = [
            idx for idx in range(len(record_list)) if results[idx] is None
        ]
        if non_ref_positions:
            target_records = [record_list[idx] for idx in non_ref_positions]
            target_params = [params_batch[idx] for idx in non_ref_positions]
            rng_keys = [self._rng_for_record(rec) for rec in target_records]
            batch_results = rebasin_batch(
                self.adapter,
                self.spec,
                self.ref_params,
                target_params,
                method=self.config.method,
                method_kwargs=self.config.method_kwargs,
                ref_views=self.ref_views,
                ref_backend=self.ref_backend,
                rng_keys=rng_keys,
            )
            for pos, result in zip(non_ref_positions, batch_results, strict=True):
                folded_params, perms, aux = result
                results[pos] = self._stage_result(folded_params, perms, aux)

        return [res for res in results if res is not None]

    @property
    def supports_batching(self) -> bool:
        if self.strategy is None:
            return False
        return bool(self.strategy.supports_batching())

    @property
    def prefers_gpu(self) -> bool:
        return self.config.method.lower() == "sinkhorn"

    def reference_output(self, record: SampleRecord, params: ParamTree) -> ParamTree:
        return self.process_single(record, params).params


__all__ = [
    "StageResult",
    "StageExecutor",
    "NormalizeExecutor",
    "RebasinExecutor",
]
