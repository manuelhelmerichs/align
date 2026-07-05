"""Block-level views of alignment problems.

Blocks partition a problem's permutation groups into named sub-problems (the FCN
stack, one attention module, ...). This module derives block listings, extracts
self-contained sub-problems, and matches structurally identical blocks across
different networks so they can be aligned against each other.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from typing import Any

import numpy as np

from .problem import AlignmentProblem
from .tensor_ops import _canonical_axis, binding_axis_interval


def tensors_for_block(problem: AlignmentProblem, block_id: str) -> tuple[str, ...]:
    """Return tensor ids bound to any of the block's groups, in problem order."""

    block = problem.blocks[block_id]
    members = set(block.groups)
    return tuple(
        tensor_id
        for tensor_id in problem.tensors
        if any(
            binding.group in members
            for binding in problem.bindings_for_tensor(tensor_id)
        )
    )


def resolve_block_patterns(
    problem: AlignmentProblem, patterns: Sequence[str]
) -> tuple[str, ...]:
    """Resolve fnmatch ``patterns`` to block ids; every pattern must match."""

    if not problem.blocks:
        raise ValueError(
            "Problem declares no blocks; block selectors cannot be resolved."
        )
    resolved: list[str] = []
    for pattern in patterns:
        matches = [
            block_id
            for block_id in problem.blocks
            if fnmatch.fnmatchcase(block_id, pattern)
        ]
        if not matches:
            available = ", ".join(problem.blocks)
            raise ValueError(
                f"Block pattern {pattern!r} matches no blocks. Available: {available}"
            )
        for block_id in matches:
            if block_id not in resolved:
                resolved.append(block_id)
    return tuple(resolved)


def groups_for_blocks(
    problem: AlignmentProblem, patterns: Sequence[str]
) -> tuple[str, ...]:
    """Resolve block ``patterns`` to the union of their groups, in group order."""

    block_ids = resolve_block_patterns(problem, patterns)
    members = {
        group_id
        for block_id in block_ids
        for group_id in problem.blocks[block_id].groups
    }
    return tuple(gid for gid in problem.group_order if gid in members)


def describe_problem(problem: AlignmentProblem) -> dict[str, Any]:
    """Return a JSON-friendly listing of the problem's blocks.

    Problems without declared blocks are reported as one implicit block ``all``
    so the listing is uniform for older adapters.
    """

    repeated = problem.repeated_group_terms()
    attention_groups = {
        group_id
        for constraint in problem.constraints
        if constraint.kind == "attention_block"
        for group_id in constraint.groups
    }

    def _block_entry(
        block_id: str, kind: str, group_ids: Sequence[str]
    ) -> dict[str, Any]:
        tensor_ids = sorted(
            {
                binding.tensor_id
                for binding in problem.axis_bindings
                if binding.group in set(group_ids)
            }
        )
        param_count = int(
            sum(
                int(np.prod(problem.tensors[tensor_id].shape))
                for tensor_id in tensor_ids
            )
        )
        notes: list[str] = []
        if any(group_id in repeated for group_id in group_ids):
            notes.append("repeated-group terms: exact LAP unavailable, use sinkhorn")
        if any(group_id in attention_groups for group_id in group_ids):
            notes.append("attention-coupled: lap uses the structured head update")
        return {
            "id": block_id,
            "kind": kind,
            "groups": [
                {"id": group_id, "size": int(problem.groups[group_id].size)}
                for group_id in group_ids
            ],
            "num_tensors": len(tensor_ids),
            "num_params": param_count,
            "tensors": tensor_ids,
            "notes": notes,
        }

    if problem.blocks:
        blocks = [
            _block_entry(
                block.id,
                block.kind,
                [gid for gid in problem.group_order if gid in set(block.groups)],
            )
            for block in problem.blocks.values()
        ]
    else:
        blocks = [_block_entry("all", "unstructured", list(problem.group_order))]
    return {
        "architecture": problem.metadata.get("architecture"),
        "num_groups": len(problem.groups),
        "num_tensors": len(problem.tensors),
        "blocks": blocks,
    }


def format_problem_listing(problem: AlignmentProblem) -> str:
    """Render :func:`describe_problem` as human-readable text."""

    description = describe_problem(problem)
    lines = [
        f"architecture: {description['architecture']}",
        f"groups: {description['num_groups']}  tensors: {description['num_tensors']}",
        "blocks:",
    ]
    for block in description["blocks"]:
        group_summary = ", ".join(
            f"{group['id']}({group['size']})" for group in block["groups"]
        )
        lines.append(
            f"  {block['id']} [{block['kind']}]"
            f"  groups: {group_summary}"
            f"  tensors: {block['num_tensors']}"
            f"  params: {block['num_params']}"
        )
        for note in block["notes"]:
            lines.append(f"    note: {note}")
    return "\n".join(lines)


def extract_block_problem(
    problem: AlignmentProblem, block_ids: Sequence[str]
) -> AlignmentProblem:
    """Return the self-contained sub-problem covering ``block_ids``.

    The sub-problem keeps only the selected blocks' groups, their bindings, and
    the tensors those bindings touch. Constraints referencing dropped groups or
    tensors are dropped. Aligning the sub-problem permutes only internal units of
    the selected blocks, so it stays function-preserving for the full network.
    """

    unknown = sorted(set(block_ids) - set(problem.blocks))
    if unknown:
        raise ValueError("Unknown block id(s): " + ", ".join(unknown))
    if not block_ids:
        raise ValueError("extract_block_problem requires at least one block id.")

    kept_blocks = {block_id: problem.blocks[block_id] for block_id in block_ids}
    kept_groups = {
        group_id for block in kept_blocks.values() for group_id in block.groups
    }
    bindings = tuple(
        binding for binding in problem.axis_bindings if binding.group in kept_groups
    )
    kept_tensor_ids = {binding.tensor_id for binding in bindings}
    constraints = tuple(
        constraint
        for constraint in problem.constraints
        if set(constraint.groups) <= kept_groups
        and set(constraint.tensors) <= kept_tensor_ids
    )
    metadata: dict[str, Any] = {
        "architecture": problem.metadata.get("architecture"),
        "group_order": [gid for gid in problem.group_order if gid in kept_groups],
        "extracted_blocks": list(block_ids),
    }
    sub_problem = AlignmentProblem(
        groups={gid: problem.groups[gid] for gid in kept_groups},
        tensors={tid: problem.tensors[tid] for tid in kept_tensor_ids},
        axis_bindings=bindings,
        constraints=constraints,
        blocks=kept_blocks,
        metadata=metadata,
    )
    sub_problem.validate()
    return sub_problem


def _binding_profile(
    problem: AlignmentProblem,
    tensor_id: str,
    group_index: dict[str, int],
) -> tuple[Any, ...]:
    """Canonical per-tensor profile: shape + block-group binding structure."""

    tensor = problem.tensors[tensor_id]
    entries = []
    for binding in problem.bindings_for_tensor(tensor_id):
        if binding.group not in group_index:
            continue
        axis, start, stop = binding_axis_interval(tensor.shape, binding)
        selector = tuple(
            sorted(
                (_canonical_axis(len(tensor.shape), sel_axis), sel_index)
                for sel_axis, sel_index in getattr(binding, "selector", ())
            )
        )
        entries.append(
            (axis, start, stop, binding.role, group_index[binding.group], selector)
        )
    return (tuple(tensor.shape), tuple(sorted(entries)))


def block_signature(problem: AlignmentProblem, block_id: str) -> tuple[Any, ...]:
    """Canonical structural signature of one block.

    Two blocks with equal signatures have identical group sizes and identically
    bound tensors (shapes, axes, roles), so their parameters can be aligned
    against each other regardless of the surrounding architecture.
    """

    block = problem.blocks[block_id]
    ordered_groups = [gid for gid in problem.group_order if gid in set(block.groups)]
    group_index = {gid: idx for idx, gid in enumerate(ordered_groups)}
    profiles = sorted(
        _binding_profile(problem, tensor_id, group_index)
        for tensor_id in tensors_for_block(problem, block_id)
    )
    return (
        block.kind,
        tuple(int(problem.groups[gid].size) for gid in ordered_groups),
        tuple(profiles),
    )


def match_block_tensors(
    ref_problem: AlignmentProblem,
    ref_block: str,
    target_problem: AlignmentProblem,
    target_block: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Match tensors and groups of two structurally identical blocks.

    Returns ``(tensor_map, group_map)`` mapping reference ids to target ids.
    Raises if the block signatures differ or the correspondence is ambiguous.
    """

    ref_signature = block_signature(ref_problem, ref_block)
    target_signature = block_signature(target_problem, target_block)
    if ref_signature != target_signature:
        raise ValueError(
            f"Block {ref_block!r} and {target_block!r} have incompatible "
            "structures; cannot align across networks."
        )

    ref_groups = [
        gid
        for gid in ref_problem.group_order
        if gid in set(ref_problem.blocks[ref_block].groups)
    ]
    target_groups = [
        gid
        for gid in target_problem.group_order
        if gid in set(target_problem.blocks[target_block].groups)
    ]
    group_map = dict(zip(ref_groups, target_groups, strict=True))

    ref_index = {gid: idx for idx, gid in enumerate(ref_groups)}
    target_index = {gid: idx for idx, gid in enumerate(target_groups)}

    def _profiles(problem, block_id, group_index):
        profiles: dict[tuple[Any, ...], list[str]] = {}
        for tensor_id in tensors_for_block(problem, block_id):
            profile = _binding_profile(problem, tensor_id, group_index)
            profiles.setdefault(profile, []).append(tensor_id)
        return profiles

    ref_profiles = _profiles(ref_problem, ref_block, ref_index)
    target_profiles = _profiles(target_problem, target_block, target_index)

    tensor_map: dict[str, str] = {}
    for profile, ref_ids in ref_profiles.items():
        target_ids = target_profiles.get(profile, [])
        if len(ref_ids) != 1 or len(target_ids) != 1:
            raise ValueError(
                f"Ambiguous tensor correspondence between blocks {ref_block!r} "
                f"and {target_block!r}: profile {profile!r} matches "
                f"{ref_ids} vs {target_ids}."
            )
        tensor_map[ref_ids[0]] = target_ids[0]
    return tensor_map, group_map


__all__ = [
    "block_signature",
    "describe_problem",
    "extract_block_problem",
    "format_problem_listing",
    "groups_for_blocks",
    "match_block_tensors",
    "resolve_block_patterns",
    "tensors_for_block",
]
