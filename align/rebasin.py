"""Rebasin API surface and shared utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

# NOTE: Platform configuration must happen BEFORE this module is imported.
# The CLI handles this via _configure_platform_preferences() before importing runtime.
# Do NOT call configure_jax_platforms() here - it would override CLI preferences.
import jax
import numpy as np
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

if TYPE_CHECKING:
    from .strategies.base import RebasinStrategy

ParamTree: TypeAlias = Mapping[str, Any]


_STRATEGY_CACHE: dict[tuple, RebasinStrategy] = {}
_STRATEGY_CACHE_MAX_SIZE = 16
_CACHE_KEY_UNSUPPORTED = object()


def _is_hashable(value: Any) -> bool:
    """Check if a value is hashable."""
    try:
        hash(value)
        return True
    except TypeError:
        return False


def _array_checksum(value: Any) -> str | None:
    """Return a content-based checksum for array-like inputs."""
    if isinstance(value, (str, bytes, bytearray)):
        return None
    if not (isinstance(value, (np.ndarray, jax.Array)) or hasattr(value, "__array__")):
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if getattr(arr, "dtype", None) is None or arr.dtype == object:
        return None
    try:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        hasher.update(arr.tobytes())
        return hasher.hexdigest()
    except Exception:
        return None


def _fingerprint_cache_value(value: Any):
    """Produce a hashable fingerprint for caching; return sentinel on failure."""
    if _is_hashable(value):
        return value

    if is_dataclass(value):
        return _fingerprint_cache_value(asdict(value))

    checksum = _array_checksum(value)
    if checksum is not None:
        return ("array", checksum)

    if isinstance(value, Mapping):
        items = []
        for key in sorted(value):
            fingerprint = _fingerprint_cache_value(value[key])
            if fingerprint is _CACHE_KEY_UNSUPPORTED:
                return _CACHE_KEY_UNSUPPORTED
            items.append((key, fingerprint))
        return tuple(items)

    if isinstance(value, (set, frozenset)):
        items = []
        for item in sorted(value, key=repr):
            fingerprint = _fingerprint_cache_value(item)
            if fingerprint is _CACHE_KEY_UNSUPPORTED:
                return _CACHE_KEY_UNSUPPORTED
            items.append(fingerprint)
        return tuple(items)

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, np.ndarray, jax.Array)
    ):
        fingerprints = []
        for item in value:
            fingerprint = _fingerprint_cache_value(item)
            if fingerprint is _CACHE_KEY_UNSUPPORTED:
                return _CACHE_KEY_UNSUPPORTED
            fingerprints.append(fingerprint)
        return tuple(fingerprints)

    return _CACHE_KEY_UNSUPPORTED


def _build_strategy_cache_key(
    method: str, method_kwargs: Mapping[str, Any]
) -> tuple | None:
    """Build a deterministic cache key or return None if kwargs can't be fingerprinted."""
    try:
        hashed_items = []
        for key in sorted(method_kwargs):
            fingerprint = _fingerprint_cache_value(method_kwargs[key])
            if fingerprint is _CACHE_KEY_UNSUPPORTED:
                return None
            hashed_items.append((key, fingerprint))
        return (method.lower(), tuple(hashed_items))
    except Exception:
        return None


def _get_cached_strategy(
    method: str, method_kwargs: Mapping[str, Any] | None = None
) -> RebasinStrategy:
    """Get or create a cached strategy instance.

    This ensures that JIT-compiled functions within the strategy are reused
    across multiple calls, avoiding expensive recompilation.

    Kwargs containing unhashable values (e.g., calibration_data arrays) are
    fingerprinted with a content checksum when possible. If any kwargs cannot
    be deterministically fingerprinted, caching is skipped to avoid stale data.
    """
    from .strategies import get_strategy

    method_kwargs = method_kwargs or {}
    cache_key = _build_strategy_cache_key(method, method_kwargs)
    if cache_key is None:
        return get_strategy(method, **method_kwargs)

    if cache_key not in _STRATEGY_CACHE:
        if len(_STRATEGY_CACHE) >= _STRATEGY_CACHE_MAX_SIZE:
            first_key = next(iter(_STRATEGY_CACHE))
            del _STRATEGY_CACHE[first_key]
        _STRATEGY_CACHE[cache_key] = get_strategy(method, **method_kwargs)

    return _STRATEGY_CACHE[cache_key]


def clear_strategy_cache() -> None:
    """Clear the strategy cache. Useful for testing or memory management."""
    _STRATEGY_CACHE.clear()


def tree_leaves_with_names(
    params: ParamTree,
) -> tuple[list[str], list[Any], jax.tree_util.PyTreeDef]:
    """Return flattened names (dotted paths) and leaves for ``params``."""

    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
    names: list[str] = []
    leaves: list[Any] = []
    for path, leaf in leaves_with_path:
        parts: list[str] = []
        for key in path:
            if isinstance(key, DictKey):
                parts.append(str(key.key))
            elif isinstance(key, GetAttrKey):
                parts.append(str(key.name))
            elif isinstance(key, SequenceKey):
                parts.append(str(key.idx))
            else:
                parts.append(str(key))
        names.append(".".join(parts))
        leaves.append(leaf)
    return names, leaves, treedef


def save_pytree_npz(path: str | Path, params: ParamTree):
    """Persist a PyTree using dotted keys, compatible with existing tooling."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names, leaves, _ = tree_leaves_with_names(params)
    arrays = {
        name or f"leaf_{idx}": np.asarray(leaf)
        for idx, (name, leaf) in enumerate(zip(names, leaves, strict=True))
    }
    np.savez_compressed(path, **arrays)


def _compute_views(
    adapter,
    spec,
    params: ParamTree,
    *,
    strategy,
    backend: str | None = None,
    cache: bool = False,
) -> Mapping[str, Sequence[Any]]:
    requested_backend = backend or (
        "numpy" if getattr(strategy, "requires_numpy_views", False) else "jax"
    )
    return adapter.permutation_views(
        params,
        spec,
        backend=requested_backend,  # type: ignore[arg-type]
        cache=cache,
    )


def rebasin_single_sample(
    adapter,
    spec,
    ref_params: ParamTree,
    params: ParamTree,
    *,
    method: str,
    method_kwargs: Mapping[str, Any] | None = None,
    strategy: RebasinStrategy | None = None,
    rng_key: jax.Array | None = None,
    ref_views: Mapping[str, Sequence[Any]] | None = None,
    ref_backend: str | None = None,
    is_reference: bool = False,
) -> tuple[ParamTree, Mapping[str, Any], dict[str, Any] | None]:
    """Rebasin ``params`` using an architecture adapter and alignment spec."""
    method_kwargs = dict(method_kwargs or {})
    strategy = strategy or _get_cached_strategy(method, method_kwargs)

    backend = ref_backend or (
        "numpy" if getattr(strategy, "requires_numpy_views", False) else "jax"
    )
    if ref_views is None:
        ref_views = _compute_views(
            adapter,
            spec,
            ref_params,
            strategy=strategy,
            backend=backend,
            cache=True,
        )

    if is_reference:
        perms = strategy.identity_permutations(spec, ref_views)
        return params, perms, {"reference": True}

    target_views = _compute_views(
        adapter,
        spec,
        params,
        strategy=strategy,
        backend=backend,
        cache=False,
    )
    perms, aux_info = strategy.match(spec, ref_views, target_views, rng_key=rng_key)

    folded_params = adapter.permute(params, spec, perms)
    return folded_params, perms, aux_info


def rebasin_batch(
    adapter,
    spec,
    ref_params: ParamTree,
    params_batch: Sequence[ParamTree],
    *,
    method: str,
    method_kwargs: Mapping[str, Any] | None = None,
    rng_keys: Sequence[jax.Array | None] | None = None,
    ref_views: Mapping[str, Sequence[Any]] | None = None,
    ref_backend: str | None = None,
) -> list[tuple[ParamTree, Mapping[str, Any], dict[str, Any] | None]]:
    """Rebasin a batch of samples using adapter-driven alignment."""
    if not params_batch:
        return []

    method_kwargs = dict(method_kwargs or {})
    strategy = _get_cached_strategy(method, method_kwargs)
    backend = ref_backend or (
        "numpy" if getattr(strategy, "requires_numpy_views", False) else "jax"
    )
    if ref_views is None:
        ref_views = _compute_views(
            adapter,
            spec,
            ref_params,
            strategy=strategy,
            backend=backend,
            cache=True,
        )

    if not strategy.supports_batching():
        results = []
        for idx, params in enumerate(params_batch):
            rng_key = rng_keys[idx] if rng_keys else None
            results.append(
                rebasin_single_sample(
                    adapter,
                    spec,
                    ref_params,
                    params,
                    method=method,
                    method_kwargs=method_kwargs,
                    strategy=strategy,
                    rng_key=rng_key,
                    ref_views=ref_views,
                    ref_backend=backend,
                    is_reference=False,
                )
            )
        return results

    target_views_batch = [
        _compute_views(
            adapter,
            spec,
            params,
            strategy=strategy,
            backend=backend,
            cache=False,
        )
        for params in params_batch
    ]
    perms_batch, aux_batch = strategy.batch_match(
        spec, ref_views, target_views_batch, rng_keys=rng_keys
    )

    results: list[tuple[ParamTree, Mapping[str, Any], dict[str, Any] | None]] = []
    for params, perms, aux in zip(params_batch, perms_batch, aux_batch, strict=True):
        folded_params = adapter.permute(params, spec, perms)
        results.append((folded_params, perms, aux))
    return results


__all__ = [
    "rebasin_batch",
    "tree_leaves_with_names",
    "save_pytree_npz",
    "rebasin_single_sample",
    "clear_strategy_cache",
]
