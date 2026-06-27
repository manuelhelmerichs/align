"""Per-group scale state for the graph-native scale action."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class ScaleState:
    """Solver-free state holding one positive scale vector per group.

    This is the scaling counterpart of :class:`PermutationState`. It is consumed
    by :meth:`AlignmentProblem.apply_scales`, which divides ``out``-role axes and
    multiplies ``in``-role axes by the group's scale.
    """

    group_order: tuple[str, ...]
    scales: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def identity(
        cls,
        problem,
        *,
        backend: Literal["jax", "numpy"] = "numpy",
        dtype=np.float32,
    ) -> ScaleState:
        if backend == "jax":
            import jax.numpy as jnp

            ones = jnp.ones
        else:
            ones = np.ones
        scales = {
            group_id: ones(group.size, dtype=dtype)
            for group_id, group in problem.groups.items()
        }
        return cls(group_order=problem.group_order, scales=scales)

    @classmethod
    def from_scales(
        cls,
        problem,
        scales: Mapping[str, Any],
        *,
        backend: Literal["jax", "numpy"] = "numpy",
    ) -> ScaleState:
        """Build a state, defaulting unspecified groups to an identity scale."""

        state = cls.identity(problem, backend=backend)
        state.scales.update(dict(scales))
        return state

    def copy(self) -> ScaleState:
        return ScaleState(
            group_order=tuple(self.group_order),
            scales=dict(self.scales),
            metadata=dict(self.metadata),
        )

    def validate(self, problem) -> None:
        """Validate group coverage, vector shapes, and positivity."""

        if set(self.group_order) != set(problem.groups):
            raise ValueError(
                "ScaleState group_order must contain exactly the problem groups."
            )
        if set(self.scales) != set(problem.groups):
            raise ValueError(
                "ScaleState scales must contain exactly the problem groups."
            )
        for group_id, group in problem.groups.items():
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
