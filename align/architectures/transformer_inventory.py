"""Shared structural inventory for transformer architecture recipes.

The inventory pass deliberately stops before emitting symmetry groups.  It
locates blocks, attention modules, norms, dense edges, and stream-shaped leaves
once; family-specific recipes then validate their attention/norm semantics and
compose the appropriate rules.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..symmetry.tensor_ops import _descend, _maybe_descend
from .rules import is_attention_module, is_dense_module


def natural_key(value: str) -> list[int | str]:
    """Natural-sort one path segment."""

    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts if part]


def path_key(path: tuple[str, ...]) -> list[list[int | str]]:
    """Natural-sort a parameter path segment by segment."""

    return [natural_key(part) for part in path]


def array_shape(node: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.shape(node))


@dataclass(frozen=True)
class AttentionInventory:
    path: tuple[str, ...]
    query_shape: tuple[int, ...]


@dataclass(frozen=True)
class DenseLayerInventory:
    path: tuple[str, ...]
    din: int
    dout: int


@dataclass(frozen=True)
class TransformerBlockInventory:
    scope: tuple[str, ...]
    attention: AttentionInventory
    norms: tuple[tuple[str, ...], ...]
    dense_layers: tuple[DenseLayerInventory, ...]


@dataclass(frozen=True)
class TransformerInventory:
    """Typed, naturally ordered topology shared by transformer recipes."""

    root_path: tuple[str, ...]
    d_model: int
    attentions: tuple[AttentionInventory, ...]
    blocks: tuple[TransformerBlockInventory, ...]
    outside_norms: tuple[tuple[str, ...], ...]
    outside_dense: tuple[DenseLayerInventory, ...]
    feature_leaves: tuple[tuple[str, ...], ...]

    def full_path(self, path: tuple[str, ...]) -> tuple[str, ...]:
        return (*self.root_path, *path)

    @property
    def norm_paths(self) -> tuple[tuple[str, ...], ...]:
        return tuple(norm for block in self.blocks for norm in block.norms) + tuple(
            self.outside_norms
        )


def chain_dense_layers(
    dense: tuple[DenseLayerInventory, ...] | list[DenseLayerInventory],
    d_model: int,
) -> list[list[DenseLayerInventory]]:
    """Group naturally ordered dense edges into stream-fed chains."""

    chains: list[list[DenseLayerInventory]] = []
    for layer in dense:
        if chains and layer.din == chains[-1][-1].dout:
            chains[-1].append(layer)
        elif layer.din == d_model:
            chains.append([layer])
        else:
            raise ValueError(
                f"Dense layer {'/'.join(layer.path)} (in={layer.din}) fits neither "
                "the residual stream nor the preceding classifier chain."
            )
    return chains


def discover_transformer_inventory(
    params: Mapping[str, Any],
    *,
    parameter_root: str,
    norm_predicate: Callable[[Any], bool],
    validate_norm: Callable[[Any, tuple[str, ...]], None],
    norm_name: str,
    query_rank: int,
    query_layout: str,
    wrong_layout_hint: str | None = None,
    recognize_kernel_feature_modules: bool = False,
) -> TransformerInventory:
    """Discover the common topology of one pre-norm transformer tree."""

    root_path = tuple(part for part in parameter_root.split(".") if part)
    subtree = _maybe_descend(params, root_path) if root_path else params
    if not isinstance(subtree, Mapping):
        raise ValueError(f"parameter_root '{parameter_root}' not found in params.")

    attention_paths = _find_attention_modules(subtree)
    if not attention_paths:
        raise ValueError(
            "No attention modules (query/key/value/out) found; is this a "
            "transformer parameter tree?"
        )

    attentions: list[AttentionInventory] = []
    d_model: int | None = None
    for path in attention_paths:
        query_shape = array_shape(_descend(subtree, path)["query"]["kernel"])
        if len(query_shape) != query_rank:
            hint = f" {wrong_layout_hint}" if wrong_layout_hint else ""
            raise ValueError(
                f"Attention module {'/'.join(path)} query kernel must be "
                f"{query_layout}, got {query_shape}.{hint}"
            )
        if d_model is None:
            d_model = query_shape[0]
        elif query_shape[0] != d_model:
            raise ValueError(
                f"Attention module {'/'.join(path)} uses d_model {query_shape[0]}, "
                f"expected {d_model}."
            )
        attentions.append(AttentionInventory(path=path, query_shape=query_shape))
    assert d_model is not None

    block_scope_by_attention = {
        attention.path: _block_scope(
            subtree, attention.path, norm_predicate=norm_predicate, norm_name=norm_name
        )
        for attention in attentions
    }
    block_scopes = sorted(set(block_scope_by_attention.values()), key=path_key)
    attentions_by_scope: dict[tuple[str, ...], list[AttentionInventory]] = {
        scope: [] for scope in block_scopes
    }
    for attention in attentions:
        attentions_by_scope[block_scope_by_attention[attention.path]].append(attention)

    blocks: list[TransformerBlockInventory] = []
    for scope in block_scopes:
        members = attentions_by_scope[scope]
        if len(members) != 1:
            raise ValueError(
                f"Transformer block {'/'.join(scope)} contains {len(members)} "
                "attention modules; expected exactly one."
            )
        norms, dense = _collect_block_layers(
            _descend(subtree, scope),
            scope_path=scope,
            attention_path=members[0].path,
            norm_predicate=norm_predicate,
            validate_norm=validate_norm,
        )
        blocks.append(
            TransformerBlockInventory(
                scope=scope,
                attention=members[0],
                norms=tuple(norms),
                dense_layers=tuple(dense),
            )
        )

    outside_norms, outside_dense, feature_leaves = _collect_outside(
        subtree,
        block_scopes=set(block_scopes),
        d_model=d_model,
        norm_predicate=norm_predicate,
        validate_norm=validate_norm,
        recognize_kernel_feature_modules=recognize_kernel_feature_modules,
    )
    return TransformerInventory(
        root_path=root_path,
        d_model=d_model,
        attentions=tuple(attentions),
        blocks=tuple(blocks),
        outside_norms=tuple(outside_norms),
        outside_dense=tuple(outside_dense),
        feature_leaves=tuple(feature_leaves),
    )


def _find_attention_modules(subtree: Mapping[str, Any]) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []

    def _walk(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        if is_attention_module(node):
            found.append(prefix)
            return
        for name, value in node.items():
            if isinstance(value, Mapping):
                _walk(value, prefix + (str(name),))

    _walk(subtree, ())
    return sorted(found, key=path_key)


def _block_scope(
    subtree: Mapping[str, Any],
    attention_path: tuple[str, ...],
    *,
    norm_predicate: Callable[[Any], bool],
    norm_name: str,
) -> tuple[str, ...]:
    for depth in range(len(attention_path) - 1, 0, -1):
        scope_path = attention_path[:depth]
        scope = _descend(subtree, scope_path)
        if any(norm_predicate(child) for child in scope.values()):
            return scope_path
    article = "an" if norm_name[:1].lower() in "aeiou" else "a"
    raise ValueError(
        f"Could not locate a transformer block scope (an ancestor with {article} "
        f"{norm_name}) for attention module at {'/'.join(attention_path)}."
    )


def _dense_layer(path: tuple[str, ...], module: Mapping[str, Any]):
    kernel_shape = array_shape(module["kernel"])
    return DenseLayerInventory(path=path, din=kernel_shape[0], dout=kernel_shape[1])


def _collect_block_layers(
    scope: Mapping[str, Any],
    *,
    scope_path: tuple[str, ...],
    attention_path: tuple[str, ...],
    norm_predicate: Callable[[Any], bool],
    validate_norm: Callable[[Any, tuple[str, ...]], None],
) -> tuple[list[tuple[str, ...]], list[DenseLayerInventory]]:
    norms: list[tuple[str, ...]] = []
    dense: list[DenseLayerInventory] = []
    attention_rel = attention_path[len(scope_path) :]

    def _walk(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        for name, value in node.items():
            if not isinstance(value, Mapping):
                continue
            path = prefix + (str(name),)
            if path == attention_rel:
                continue
            if attention_rel[: len(path)] == path:
                _walk(value, path)
                continue
            full_path = scope_path + path
            if norm_predicate(value):
                validate_norm(value, full_path)
                norms.append(full_path)
            elif is_dense_module(value):
                dense.append(_dense_layer(full_path, value))
            else:
                _walk(value, path)

    _walk(scope, ())
    norms.sort(key=path_key)
    dense.sort(key=lambda layer: path_key(layer.path))
    return norms, dense


def _collect_outside(
    subtree: Mapping[str, Any],
    *,
    block_scopes: set[tuple[str, ...]],
    d_model: int,
    norm_predicate: Callable[[Any], bool],
    validate_norm: Callable[[Any, tuple[str, ...]], None],
    recognize_kernel_feature_modules: bool,
) -> tuple[list[tuple[str, ...]], list[DenseLayerInventory], list[tuple[str, ...]]]:
    norms: list[tuple[str, ...]] = []
    dense: list[DenseLayerInventory] = []
    leaves: list[tuple[str, ...]] = []

    def _walk(node: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        for name, value in node.items():
            path = prefix + (str(name),)
            if path in block_scopes:
                continue
            if isinstance(value, Mapping):
                if norm_predicate(value):
                    validate_norm(value, path)
                    norms.append(path)
                elif is_dense_module(value):
                    dense.append(_dense_layer(path, value))
                elif (
                    recognize_kernel_feature_modules
                    and "kernel" in value
                    and not isinstance(value["kernel"], Mapping)
                ):
                    _collect_kernel_feature_module(value, path, d_model, leaves)
                else:
                    _walk(value, path)
                continue
            shape = array_shape(value)
            if not shape:
                continue
            if shape[-1] != d_model:
                raise ValueError(
                    f"Tensor {'/'.join(path)} with shape {shape} is not part of a "
                    "recognized transformer component and has no trailing d_model "
                    f"axis ({d_model})."
                )
            leaves.append(path)

    _walk(subtree, ())
    norms.sort(key=path_key)
    dense.sort(key=lambda layer: path_key(layer.path))
    leaves.sort(key=path_key)
    return norms, dense, leaves


def _collect_kernel_feature_module(
    module: Mapping[str, Any],
    path: tuple[str, ...],
    d_model: int,
    leaves: list[tuple[str, ...]],
) -> None:
    kernel_shape = array_shape(module["kernel"])
    if kernel_shape[-1] != d_model:
        raise ValueError(
            f"Module {'/'.join(path)} has kernel shape {kernel_shape}; expected a "
            f"trailing d_model axis of {d_model}."
        )
    leaves.append((*path, "kernel"))
    if "bias" not in module:
        return
    bias_shape = array_shape(module["bias"])
    if bias_shape != (d_model,):
        raise ValueError(
            f"Module {'/'.join(path)} bias must be (d_model,), got {bias_shape}."
        )
    leaves.append((*path, "bias"))


__all__ = [
    "AttentionInventory",
    "DenseLayerInventory",
    "TransformerBlockInventory",
    "TransformerInventory",
    "array_shape",
    "chain_dense_layers",
    "discover_transformer_inventory",
    "natural_key",
    "path_key",
]
