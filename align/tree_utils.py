"""Persistent parameter-tree updates that share all untouched leaves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from flax.core import frozen_dict


def replace_paths(
    params: Mapping[str, Any],
    replacements: Sequence[tuple[tuple[str, ...], Any]],
) -> Mapping[str, Any]:
    """Path-copy mappings and replace leaves without copying untouched arrays."""

    root: dict[str, Any] = dict(params)
    copied: dict[tuple[str, ...], dict[str, Any]] = {(): root}
    for path, value in replacements:
        if not path:
            raise ValueError("Cannot replace the root parameter mapping.")
        source: Mapping[str, Any] = params
        prefix: tuple[str, ...] = ()
        target = root
        for key in path[:-1]:
            prefix = (*prefix, key)
            source_child = source[key]
            if not isinstance(source_child, Mapping):
                raise KeyError(
                    f"Parameter path {path!r} descends through non-mapping {key!r}."
                )
            child = copied.get(prefix)
            if child is None:
                child = dict(source_child)
                copied[prefix] = child
                target[key] = child
            target = child
            source = source_child
        target[path[-1]] = value
    return (
        frozen_dict.freeze(root) if isinstance(params, frozen_dict.FrozenDict) else root
    )


__all__ = ["replace_paths"]
