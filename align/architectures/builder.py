"""Shared problem accumulator used by block adapters.

A :class:`ProblemBuilder` collects groups, tensors, bindings, constraints, and
block specs while block adapters run, then assembles and validates the final
:class:`~align.alignment.AlignmentProblem`. Tensor specs are derived from the
parameter tree the builder was created for, so block adapters only pass paths.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..alignment import (
    AlignmentProblem,
    AxisBinding,
    BlockSpec,
    GraphConstraint,
    PermutationGroup,
    TensorSpec,
)
from ..alignment.tensor_ops import _descend


class ProblemBuilder:
    """Accumulates one alignment problem across block adapters."""

    def __init__(self, params: Mapping[str, Any], *, architecture: str) -> None:
        self.params = params
        self.groups: dict[str, PermutationGroup] = {}
        self.tensors: dict[str, TensorSpec] = {}
        self.bindings: list[AxisBinding] = []
        self.constraints: list[GraphConstraint] = []
        self.blocks: dict[str, BlockSpec] = {}
        self.metadata: dict[str, Any] = {"architecture": architecture}
        self.group_order: list[str] = []

    def add_group(
        self, group_id: str, size: int, *, transforms: str = "permutation"
    ) -> str:
        if group_id in self.groups:
            raise ValueError(f"Group {group_id!r} was already added.")
        self.groups[group_id] = PermutationGroup(
            id=group_id, size=int(size), transforms=transforms
        )
        self.group_order.append(group_id)
        return group_id

    def tensor(self, path: tuple[str, ...]) -> str:
        """Register (or fetch) the tensor at ``path``; returns its id."""

        tensor_id = "/".join(path)
        if tensor_id not in self.tensors:
            value = _descend(self.params, path)
            shape = tuple(int(dim) for dim in np.shape(value))
            self.tensors[tensor_id] = TensorSpec(
                id=tensor_id,
                path=tuple(path),
                shape=shape,
                metadata={"architecture": self.metadata["architecture"]},
            )
        return tensor_id

    def bind(
        self,
        path: tuple[str, ...],
        axis: int,
        group: str,
        *,
        start: int | None = None,
        stop: int | None = None,
        role: str = "out",
        scale_power: float = 1.0,
        selector: tuple[tuple[int, int], ...] = (),
        transform_scope: str = "linear",
    ) -> str:
        """Bind one axis (or a half-open interval of it) of ``path`` to ``group``."""

        tensor_id = self.tensor(path)
        self.bindings.append(
            AxisBinding(
                tensor_id=tensor_id,
                axis=axis,
                group=group,
                start=start,
                stop=stop,
                role=role,
                scale_power=scale_power,
                selector=selector,
                transform_scope=transform_scope,
            )
        )
        return tensor_id

    def add_constraint(self, constraint: GraphConstraint) -> None:
        self.constraints.append(constraint)

    def add_block(
        self,
        block_id: str,
        kind: str,
        groups: tuple[str, ...],
        metadata: Mapping[str, Any] | None = None,
    ) -> BlockSpec:
        if block_id in self.blocks:
            raise ValueError(f"Block {block_id!r} was already added.")
        block = BlockSpec(
            id=block_id,
            kind=kind,
            groups=tuple(groups),
            metadata=dict(metadata or {}),
        )
        self.blocks[block_id] = block
        return block

    def finish(self) -> AlignmentProblem:
        """Assemble and validate the accumulated problem."""

        metadata = dict(self.metadata)
        metadata.setdefault("group_order", list(self.group_order))
        problem = AlignmentProblem(
            groups=self.groups,
            tensors=self.tensors,
            axis_bindings=tuple(self.bindings),
            constraints=tuple(self.constraints),
            blocks=self.blocks,
            metadata=metadata,
        )
        problem.validate(self.params)
        return problem


__all__ = ["ProblemBuilder"]
