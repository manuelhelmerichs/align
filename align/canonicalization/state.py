"""Per-group scale state for the graph-native scale action."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ScaleState:
    """Solver-free state holding one positive scale vector per group.

    This is the scaling counterpart of :class:`TransformState`. It is consumed
    by :meth:`SymmetryGraph.apply_scales`, which divides ``out``-role axes and
    multiplies ``in``-role axes by the group's scale.
    """

    group_order: tuple[str, ...]
    scales: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def identity(
        cls,
        graph,
        *,
        dtype=np.float32,
    ) -> ScaleState:
        scales = {
            group_id: np.ones(group.size, dtype=dtype)
            for group_id, group in graph.groups.items()
        }
        return cls(group_order=graph.group_order, scales=scales)

    @classmethod
    def from_scales(
        cls,
        graph,
        scales: Mapping[str, Any],
    ) -> ScaleState:
        """Build a state, defaulting unspecified groups to an identity scale."""

        state = cls.identity(graph)
        state.scales.update(dict(scales))
        return state

    def validate(self, graph) -> None:
        """Validate group coverage, vector shapes, and positivity."""

        if len(self.group_order) != len(graph.groups) or set(self.group_order) != set(
            graph.groups
        ):
            raise ValueError(
                "ScaleState group_order must contain exactly the graph groups."
            )
        if set(self.scales) != set(graph.groups):
            raise ValueError("ScaleState scales must contain exactly the graph groups.")
        for group_id, group in graph.groups.items():
            vector = np.asarray(self.scales[group_id])
            if vector.shape != (int(group.size),):
                raise ValueError(
                    f"Group {group_id!r} scale has shape {vector.shape}, expected "
                    f"({group.size},)."
                )
            if not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
                raise ValueError(
                    f"Group {group_id!r} scale must be strictly positive and finite."
                )


__all__ = ["ScaleState"]
