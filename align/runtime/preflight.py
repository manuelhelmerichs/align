"""Side-effect-free run preparation shared by validation and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from ..config import RunConfig
from ..sample_manifest import SampleManifest
from .loaders import SampleLoader
from .stages import MatchExecutor, StageExecutor, build_stage_executors


@dataclass(frozen=True)
class PreparedRun:
    """Validated executors plus a serializable static execution summary."""

    executors: tuple[tuple[str, StageExecutor], ...]
    summary: dict[str, Any]


def prepare_run(config: RunConfig, manifest: SampleManifest) -> PreparedRun:
    """Load the reference and validate every declared stage without writing state."""

    reference = SampleLoader(manifest).load_reference()
    batch_size = max(1, int(config.runtime.per_device_batch or 1))
    executors = tuple(
        build_stage_executors(
            stage_order=config.active_stages(),
            manifest=manifest,
            reference_sample=reference,
            family=config.architecture.family,
            recipe_kwargs=config.architecture.recipe_kwargs,
            canonicalize_config=config.canonicalize,
            center_softmax_head_config=config.center_softmax_head,
            match_config=config.match,
            match_reference_index=manifest.reference_index,
            seed=config.match.seed if config.match is not None else None,
            batch_size=batch_size,
        )
    )

    phases: list[dict[str, Any]] = []
    stage_plans: list[dict[str, Any]] = []
    match_graph = None
    for name, executor in executors:
        backend = "device" if executor.prefers_gpu else "host"
        if not phases or phases[-1]["backend"] != backend:
            phases.append({"backend": backend, "stages": [name]})
        else:
            phases[-1]["stages"].append(name)
        graph = getattr(executor, "graph", None)
        stage_plan = {
            "stage": name,
            "backend": backend,
            "supports_batching": bool(executor.supports_batching),
        }
        if graph is not None:
            stage_plan.update(
                {
                    "groups": len(graph.groups),
                    "tensors": len(graph.tensors),
                    "bindings": len(graph.axis_bindings),
                }
            )
        if isinstance(executor, MatchExecutor):
            match_graph = graph
            stage_plan["solvers"] = executor.config.solvers_payload()
            stage_plan["objective"] = executor.config.objective.type
            stage_plan["backend_phases"] = [
                {"backend": backend, "steps": list(indices)}
                for backend, indices in executor.solver_sequence.backend_phases
            ]
        stage_plans.append(stage_plan)

    reference_bytes = sum(
        int(np.asarray(leaf).nbytes)
        for leaf in jax.tree_util.tree_leaves(reference.params)
    )
    tensor_shapes = (
        {tensor_id: list(spec.shape) for tensor_id, spec in match_graph.tensors.items()}
        if match_graph is not None
        else {}
    )
    return PreparedRun(
        executors=executors,
        summary={
            "stages": stage_plans,
            "backend_phases": phases,
            "batch_size": batch_size,
            "reference_bytes": reference_bytes,
            "match_tensor_shapes": tensor_shapes,
        },
    )


__all__ = ["PreparedRun", "prepare_run"]
