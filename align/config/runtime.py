"""Runtime execution configuration for align."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._utils import _maybe_int, _parse_int_list, _require_bool, _validate_fields

_VALID_VERBOSITY = frozenset({"quiet", "info", "debug"})
_RUNTIME_FIELDS = frozenset(
    {
        "resume",
        "dry_run",
        "list_samples",
        "describe_symmetry",
        "validate_only",
        "force_cpu",
        "force_gpu",
        "parallelism",
        "device_ids",
        "per_device_batch",
        "verbosity",
        "validate_artifacts_on_resume",
        "save_intermediate",
    }
)


def _validate_verbosity(value: Any) -> str:
    verbosity = str(value)
    if verbosity not in _VALID_VERBOSITY:
        raise ValueError(
            f"Unknown runtime.verbosity '{verbosity}'. "
            f"Available: {', '.join(sorted(_VALID_VERBOSITY))}"
        )
    return verbosity


@dataclass
class RuntimeConfig:
    """Runtime execution options for the align CLI."""

    resume: bool = False
    dry_run: bool = False
    list_samples: bool = False
    describe_symmetry: bool = False
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
        _validate_fields("runtime", payload, _RUNTIME_FIELDS)
        device_ids = payload.get("device_ids")
        if isinstance(device_ids, str):
            device_ids = _parse_int_list(device_ids)
        return cls(
            resume=_require_bool("runtime.resume", payload.get("resume", False)),
            dry_run=_require_bool("runtime.dry_run", payload.get("dry_run", False)),
            list_samples=_require_bool(
                "runtime.list_samples", payload.get("list_samples", False)
            ),
            describe_symmetry=_require_bool(
                "runtime.describe_symmetry", payload.get("describe_symmetry", False)
            ),
            validate_only=_require_bool(
                "runtime.validate_only", payload.get("validate_only", False)
            ),
            force_cpu=_require_bool(
                "runtime.force_cpu", payload.get("force_cpu", False)
            ),
            force_gpu=_require_bool(
                "runtime.force_gpu", payload.get("force_gpu", False)
            ),
            parallelism=_maybe_int(payload.get("parallelism")),
            device_ids=list(device_ids) if device_ids else None,
            per_device_batch=_maybe_int(payload.get("per_device_batch")),
            verbosity=_validate_verbosity(payload.get("verbosity", "info")),
            validate_artifacts_on_resume=_require_bool(
                "runtime.validate_artifacts_on_resume",
                payload.get("validate_artifacts_on_resume", True),
            ),
            save_intermediate=_require_bool(
                "runtime.save_intermediate",
                payload.get("save_intermediate", False),
            ),
        )


__all__ = ["RuntimeConfig"]
