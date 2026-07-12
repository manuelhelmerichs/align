"""Small shared helpers for deterministic architecture inventory."""

from __future__ import annotations

import re


def natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Return a total-order key that sorts embedded decimal integers naturally."""

    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", str(value))
        if part
    )


def path_key(
    path: tuple[str, ...],
) -> tuple[tuple[tuple[int, int | str], ...], ...]:
    return tuple(natural_key(part) for part in path)


def path_tokens(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split(".") if part)


__all__ = ["natural_key", "path_key", "path_tokens"]
