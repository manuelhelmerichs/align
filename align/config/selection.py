"""Sample selection configuration for align."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._utils import _maybe_int, _parse_int_list


@dataclass
class SelectionConfig:
    """Subset of samples to operate on."""

    chain_indices: list[int] | None = None
    samples_per_chain: int | None = None
    sample_step: int = 1
    max_total: int | None = None
    ref_chain: int = 0
    ref_sample: int = 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SelectionConfig:
        chain_indices = payload.get("chain_indices")
        if isinstance(chain_indices, str):
            chain_indices = _parse_int_list(chain_indices)
        return cls(
            chain_indices=list(chain_indices) if chain_indices else None,
            samples_per_chain=_maybe_int(payload.get("samples_per_chain")),
            sample_step=int(payload.get("sample_step", 1)),
            max_total=_maybe_int(payload.get("max_total")),
            ref_chain=int(payload.get("ref_chain", 0)),
            ref_sample=int(payload.get("ref_sample", 0)),
        )


__all__ = ["SelectionConfig"]
