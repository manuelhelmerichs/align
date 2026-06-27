"""Sample artifact format metadata that is safe to import before JAX."""

from __future__ import annotations

DEFAULT_SAMPLE_FORMAT = "pytree_npz"

_SAMPLE_SUFFIXES: dict[str, tuple[str, ...]] = {
    DEFAULT_SAMPLE_FORMAT: (".npz",),
}


def normalize_sample_format(name: str | None) -> str:
    """Normalize and validate a sample format name."""

    sample_format = (name or DEFAULT_SAMPLE_FORMAT).lower()
    if sample_format not in _SAMPLE_SUFFIXES:
        available = ", ".join(available_sample_formats())
        raise ValueError(
            f"Unknown sample format '{sample_format}'. Available: {available}"
        )
    return sample_format


def available_sample_formats() -> tuple[str, ...]:
    """Return configured sample artifact formats."""

    return tuple(sorted(_SAMPLE_SUFFIXES))


def sample_file_suffixes(sample_format: str) -> tuple[str, ...]:
    """Return file suffixes that belong to ``sample_format``."""

    return _SAMPLE_SUFFIXES[normalize_sample_format(sample_format)]


__all__ = [
    "DEFAULT_SAMPLE_FORMAT",
    "available_sample_formats",
    "normalize_sample_format",
    "sample_file_suffixes",
]
