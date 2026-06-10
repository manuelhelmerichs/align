"""Runtime execution configuration for align."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._utils import _maybe_int, _parse_int_list


@dataclass
class RuntimeConfig:
    """Runtime execution options for the align CLI."""

    resume: bool = False
    dry_run: bool = False
    list_samples: bool = False
    validate_only: bool = False
    force_cpu: bool = False
    force_gpu: bool = False
    parallelism: int | None = None
    device_ids: list[int] | None = None
    per_device_batch: int | None = None
    verbosity: str = "info"
    validate_artifacts_on_resume: bool = True
    save_intermediate: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RuntimeConfig:
        device_ids = payload.get("device_ids")
        if isinstance(device_ids, str):
            device_ids = _parse_int_list(device_ids)
        return cls(
            resume=bool(payload.get("resume", False)),
            dry_run=bool(payload.get("dry_run", False)),
            list_samples=bool(payload.get("list_samples", False)),
            validate_only=bool(payload.get("validate_only", False)),
            force_cpu=bool(payload.get("force_cpu", False)),
            force_gpu=bool(payload.get("force_gpu", False)),
            parallelism=_maybe_int(payload.get("parallelism")),
            device_ids=list(device_ids) if device_ids else None,
            per_device_batch=_maybe_int(payload.get("per_device_batch")),
            verbosity=str(payload.get("verbosity", "info")),
            validate_artifacts_on_resume=bool(
                payload.get("validate_artifacts_on_resume", True)
            ),
            save_intermediate=bool(payload.get("save_intermediate", False)),
        )


__all__ = ["RuntimeConfig"]
