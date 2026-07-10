"""Sample selection configuration for align."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._utils import _maybe_int, _parse_int_list, _validate_fields

_SELECTION_FIELDS = frozenset(
    {
        "chain_indices",
        "samples_per_chain",
        "sample_step",
        "max_total",
        "reference_chain",
        "reference_sample",
    }
)


@dataclass
class SelectionConfig:
    """Subset of samples to operate on."""

    chain_indices: list[int] | None = None
    samples_per_chain: int | None = None
    sample_step: int = 1
    max_total: int | None = None
    reference_chain: int = 0
    reference_sample: int = 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SelectionConfig:
        _validate_fields("selection", payload, _SELECTION_FIELDS)
        chain_indices = payload.get("chain_indices")
        if isinstance(chain_indices, str):
            chain_indices = _parse_int_list(chain_indices)
        return cls(
            chain_indices=list(chain_indices) if chain_indices else None,
            samples_per_chain=_maybe_int(payload.get("samples_per_chain")),
            sample_step=int(payload.get("sample_step", 1)),
            max_total=_maybe_int(payload.get("max_total")),
            reference_chain=int(payload.get("reference_chain", 0)),
            reference_sample=int(payload.get("reference_sample", 0)),
        )


__all__ = ["SelectionConfig"]
