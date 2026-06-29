"""Utility functions for configuration parsing."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _to_path(value: Any) -> Path | None:
    """Convert a value to a Path, or None if empty."""
    if value is None or value == "":
        return None
    return Path(str(value)).expanduser()


def _maybe_int(value: Any) -> int | None:
    """Convert a value to int, or None if empty."""
    if value is None or value == "":
        return None
    return int(value)


def _require_bool(name: str, value: Any) -> bool:
    """Validate that a config value is a real boolean."""
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean, got {type(value).__name__}.")
    return value


def _validate_fields(
    section: str, payload: Mapping[str, Any], allowed: frozenset[str]
) -> None:
    """Reject unknown keys in a config section."""
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section} config option(s): " + ", ".join(unknown))


def _parse_int_list(value: Sequence[int] | str) -> list[int]:
    """Parse a comma-separated string or sequence into a list of ints."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [int(v) for v in value]
    return [int(chunk) for chunk in str(value).split(",") if chunk]


def _merge_dicts(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge overrides into base dict."""
    merged = dict(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


__all__ = [
    "_to_path",
    "_maybe_int",
    "_require_bool",
    "_validate_fields",
    "_parse_int_list",
    "_merge_dicts",
]
