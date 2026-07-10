"""Small parameter-tree editing helper shared by canonicalization plans."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from flax.core import frozen_dict

from ..symmetry.tensor_ops import _set_path


def replace_paths(
    params: Mapping[str, Any],
    replacements: Sequence[tuple[tuple[str, ...], Any]],
) -> Mapping[str, Any]:
    """Return a copy of ``params`` with the given path -> value replacements."""

    is_frozen = isinstance(params, frozen_dict.FrozenDict)
    mutable = frozen_dict.unfreeze(params) if is_frozen else copy.deepcopy(params)
    for path, value in replacements:
        _set_path(mutable, path, value)
    return frozen_dict.freeze(mutable) if is_frozen else mutable


__all__ = ["replace_paths"]
