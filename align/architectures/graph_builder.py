"""Shared symmetry-graph accumulator used by component rules.

A :class:`SymmetryGraphBuilder` collects groups, tensors, bindings, constraints, and
component specs while component rules run, then assembles and validates the final
:class:`~align.symmetry.SymmetryGraph`. Tensor specs are derived from the
parameter tree the builder was created for, so component rules only pass paths.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..symmetry import (
    AxisBinding,
    ComponentSpec,
    ConstraintRecord,
    SymmetryGraph,
    SymmetryGroup,
    TensorSpec,
)
from ..symmetry.tensor_ops import _descend


class SymmetryGraphBuilder:
    """Accumulates one symmetry graph across component rules."""

    def __init__(self, params: Mapping[str, Any], *, architecture: str) -> None:
        self.params = params
        self.groups: dict[str, SymmetryGroup] = {}
        self.tensors: dict[str, TensorSpec] = {}
        self.bindings: list[AxisBinding] = []
        self.constraints: list[ConstraintRecord] = []
        self.components: dict[str, ComponentSpec] = {}
        self.metadata: dict[str, Any] = {"architecture": architecture}
        self.group_order: list[str] = []

    def add_group(
        self, group_id: str, size: int, *, transform_family: str = "permutation"
    ) -> str:
        if group_id in self.groups:
            raise ValueError(f"Group {group_id!r} was already added.")
        self.groups[group_id] = SymmetryGroup(
            id=group_id, size=int(size), transform_family=transform_family
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

    def add_constraint(self, constraint: ConstraintRecord) -> None:
        self.constraints.append(constraint)

    def add_component(
        self,
        component_id: str,
        kind: str,
        groups: tuple[str, ...],
        metadata: Mapping[str, Any] | None = None,
    ) -> ComponentSpec:
        if component_id in self.components:
            raise ValueError(f"Component {component_id!r} was already added.")
        component = ComponentSpec(
            id=component_id,
            kind=kind,
            groups=tuple(groups),
            metadata=dict(metadata or {}),
        )
        self.components[component_id] = component
        return component

    def finish(self) -> SymmetryGraph:
        """Assemble and validate the accumulated graph."""

        metadata = dict(self.metadata)
        metadata.setdefault("group_order", list(self.group_order))
        graph = SymmetryGraph(
            groups=self.groups,
            tensors=self.tensors,
            axis_bindings=tuple(self.bindings),
            constraints=tuple(self.constraints),
            components=self.components,
            metadata=metadata,
        )
        graph.validate(self.params)
        return graph


__all__ = ["SymmetryGraphBuilder"]
