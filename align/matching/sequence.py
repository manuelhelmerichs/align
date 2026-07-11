"""Optimization schedules for graph-native alignment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import numpy as np

from ..symmetry import materialize_many
from .attention import (
    attention_module_specs,
    gqa_module_specs,
    update_attention_module,
    update_gqa_attention_module,
)
from .solvers import (
    LAPGroupSolver,
    ProcrustesGroupSolver,
    SinkhornSolver,
    SolverStep,
    _scheduled_groups,
)
from .state import (
    MatrixTransform,
    PermutationTransform,
    SignedPermutationTransform,
    TransformState,
)


def _check_permute_only_folded(graph, *data_mappings) -> None:
    """Preflight for orthogonal groups with ``permute_only`` bindings.

    General orthogonal maps skip permute-only bindings (RMSNorm scales), which
    is exact only when the bound tensors are the folded constant 1. Enforced
    once per solve so a schedule on unfolded parameters fails loudly instead
    of silently corrupting the symmetry.
    """

    for binding in graph.axis_bindings:
        if binding.transform_scope != "permute_only":
            continue
        if graph.groups[binding.group].transform_family != "orthogonal":
            continue
        for data in data_mappings:
            tensor = np.asarray(data[binding.tensor_id])
            if float(np.max(np.abs(tensor - 1.0))) > 1e-3:
                raise ValueError(
                    f"Group {binding.group!r} declares the orthogonal transform "
                    f"class, but {binding.tensor_id!r} is not folded to 1 "
                    "(max deviation "
                    f"{float(np.max(np.abs(tensor - 1.0))):.3g}). Run scale "
                    "canonicalization (RMSNorm gamma folding) before matching, or "
                    "declare the stream as 'signed_permutation'."
                )


class SolverSequence:
    """Run an explicit sequence of hard and soft alignment solvers."""

    def __init__(self, objective, steps: Sequence[SolverStep]) -> None:
        if not steps:
            raise ValueError("Solver sequence must contain at least one step.")
        self.objective = objective
        self.steps = tuple(steps)
        for step in self.steps:
            if step.solver not in {"lap", "procrustes", "sinkhorn"}:
                raise ValueError(f"Unknown matching solver {step.solver!r}.")
        self._sinkhorn_solvers = {
            index: SinkhornSolver(self.objective, step)
            for index, step in enumerate(self.steps)
            if step.solver == "sinkhorn"
        }

    @classmethod
    def from_steps(cls, objective, steps) -> SolverSequence:
        steps = [
            step if isinstance(step, SolverStep) else SolverStep.from_mapping(step)
            for step in steps
        ]
        return cls(objective, steps)

    @property
    def prefers_gpu(self) -> bool:
        return any(step.solver == "sinkhorn" for step in self.steps)

    @property
    def supports_batching(self) -> bool:
        return all(step.solver == "sinkhorn" for step in self.steps)

    @property
    def backend(self) -> str:
        return "jax" if self.prefers_gpu else "numpy"

    def validate_graph(self, graph) -> None:
        """Validate objective/solver/group compatibility before any execution."""

        if getattr(self.objective, "name", None) == "relative_fisher":
            invalid = [step.solver for step in self.steps if step.solver != "sinkhorn"]
            if invalid:
                raise ValueError(
                    "relative_fisher is Sinkhorn-only; incompatible solver steps: "
                    + ", ".join(invalid)
                )
        for index, step in enumerate(self.steps):
            groups = _scheduled_groups(graph, step)
            if step.solver == "sinkhorn":
                unsupported = [
                    group_id
                    for group_id in groups
                    if graph.groups[group_id].transform_family != "permutation"
                ]
                if unsupported:
                    detail = ", ".join(
                        f"{group_id} ({graph.groups[group_id].transform_family})"
                        for group_id in unsupported
                    )
                    raise ValueError(
                        f"Solver step {index} uses sinkhorn on unsupported transform "
                        f"families: {detail}. Sinkhorn currently represents only "
                        "plain permutations; schedule lap/procrustes for the other "
                        "families or restrict this step's groups/components."
                    )
            if step.solver == "procrustes":
                ineligible = [
                    group_id
                    for group_id in groups
                    if graph.groups[group_id].transform_family != "orthogonal"
                ]
                if (
                    step.groups is not None or step.components is not None
                ) and ineligible:
                    raise ValueError(
                        f"Solver step {index} uses procrustes on non-orthogonal groups: "
                        + ", ".join(ineligible)
                    )

    def solve(
        self,
        graph,
        reference_data,
        target_data,
        *,
        rng_key=None,
        initial_state: TransformState | None = None,
        rng_keys=None,
    ) -> tuple[TransformState, dict[str, Any]]:
        self.validate_graph(graph)
        state = initial_state or TransformState.identity(
            graph, backend="jax" if self.prefers_gpu else "numpy"
        )
        _check_permute_only_folded(graph, reference_data, target_data)
        aux_steps: list[dict[str, Any]] = []
        for index, step in enumerate(self.steps):
            if step.solver == "lap":
                state, aux = self._run_lap_step(
                    graph, reference_data, target_data, state, step
                )
            elif step.solver == "procrustes":
                state, aux = self._run_procrustes_step(
                    graph, reference_data, target_data, state, step
                )
            elif step.solver == "sinkhorn":
                step_key = None
                if rng_key is not None:
                    step_key = jax.random.fold_in(rng_key, index)
                per_record_step_keys = (
                    jax.vmap(
                        lambda key, step_index=index: jax.random.fold_in(
                            key, step_index
                        )
                    )(rng_keys)
                    if rng_keys is not None
                    else None
                )
                state, aux = self._sinkhorn_solvers[index].solve(
                    graph,
                    reference_data,
                    target_data,
                    state,
                    rng_key=step_key,
                    rng_keys=per_record_step_keys,
                )
                state = _unbatch_state(state, sample_index=0)
            else:  # pragma: no cover - guarded in __init__
                raise ValueError(step.solver)
            aux_steps.append({"index": index, **aux})

        values = self.objective.value(graph, reference_data, target_data, state)
        aux_payload = {
            "objective": getattr(self.objective, "name", type(self.objective).__name__),
            "solvers": [step.to_dict() for step in self.steps],
            "steps": aux_steps,
            "objective_final": float(np.ravel(np.asarray(values))[0]),
        }
        state.validate(graph, hard=True)
        return state, aux_payload

    def solve_batch(
        self,
        graph,
        reference_data,
        target_params_batch,
        *,
        rng_key=None,
        rng_keys=None,
        backend: str | None = None,
    ) -> tuple[list[TransformState], list[dict[str, Any]]]:
        if not target_params_batch:
            return [], []
        self.validate_graph(graph)
        if not self.supports_batching:
            results: list[TransformState] = []
            aux_payload: list[dict[str, Any]] = []
            for idx, params in enumerate(target_params_batch):
                data = graph.materialize(
                    params, backend=backend or self.backend, cache=False
                )
                sample_key = (
                    jax.random.fold_in(rng_key, idx) if rng_key is not None else None
                )
                if rng_keys is not None:
                    sample_key = rng_keys[idx]
                state, aux = self.solve(
                    graph,
                    reference_data,
                    data,
                    rng_key=sample_key,
                )
                results.append(state)
                aux_payload.append(aux)
            return results, aux_payload

        target_data = materialize_many(
            graph, target_params_batch, backend=backend or self.backend
        )
        _check_permute_only_folded(graph, reference_data, target_data)
        state = TransformState.identity(graph, backend="jax")
        aux_steps: list[dict[str, Any]] = []
        for index, _step in enumerate(self.steps):
            step_key = (
                jax.random.fold_in(rng_key, index) if rng_key is not None else None
            )
            per_record_step_keys = (
                jax.vmap(
                    lambda key, step_index=index: jax.random.fold_in(key, step_index)
                )(rng_keys)
                if rng_keys is not None
                else None
            )
            state, aux = self._sinkhorn_solvers[index].solve(
                graph,
                reference_data,
                target_data,
                state,
                rng_key=step_key,
                rng_keys=per_record_step_keys,
            )
            aux_steps.append({"index": index, **aux})

        batch_size = len(target_params_batch)
        states = [_unbatch_state(state, sample_index=idx) for idx in range(batch_size)]
        losses = np.ravel(
            np.asarray(self.objective.value(graph, reference_data, target_data, state))
        )
        aux_batch: list[dict[str, Any]] = []
        for sample_idx in range(batch_size):
            sample_steps = []
            for step_aux in aux_steps:
                sample_aux = dict(step_aux)
                for key in ("loss_initial", "loss_final", "loss_history"):
                    value = sample_aux.get(key)
                    if isinstance(value, list) and len(value) == batch_size:
                        sample_aux[key] = value[sample_idx]
                ambiguity = sample_aux.get("ambiguity")
                if isinstance(ambiguity, dict):
                    sample_aux["ambiguity"] = {
                        group_id: group_values[sample_idx]
                        for group_id, group_values in ambiguity.items()
                    }
                sample_steps.append(sample_aux)
            aux_batch.append(
                {
                    "objective": getattr(
                        self.objective, "name", type(self.objective).__name__
                    ),
                    "solvers": [step.to_dict() for step in self.steps],
                    "steps": sample_steps,
                    "objective_final": float(losses[sample_idx]),
                }
            )
        return states, aux_batch

    def _run_lap_step(self, graph, reference_data, target_data, state, step):
        self.objective.begin_lap_execution(graph, target_data)
        try:
            return self._run_lap_step_cached(
                graph, reference_data, target_data, state, step
            )
        finally:
            self.objective.end_lap_execution()

    def _run_lap_step_cached(self, graph, reference_data, target_data, state, step):
        groups = _scheduled_groups(graph, step)
        scheduled = set(groups)
        module_by_head = {
            spec.head_group: spec
            for spec in attention_module_specs(graph)
            if spec.head_group in scheduled
        }
        gqa_by_kv = {
            spec.kv_group: spec
            for spec in gqa_module_specs(graph)
            if spec.kv_group in scheduled
        }
        # Intra groups of a scheduled head/kv group are updated inside the
        # structured module update, not as standalone LAP groups.
        structured_intras = {
            group_id
            for spec in module_by_head.values()
            for group_id in (*spec.qk_groups, *spec.vo_groups)
            if group_id in scheduled
        } | {
            group_id
            for spec in gqa_by_kv.values()
            for group_id in (*spec.query_head_groups, *spec.vo_groups)
            if group_id in scheduled
        }
        solver = LAPGroupSolver(self.objective, record_ambiguity=step.record_ambiguity)
        sweeps = 0
        max_delta = 0.0
        ambiguity: dict[str, Any] = {}
        for _ in range(step.max_sweeps):
            sweeps += 1
            sweep_delta = 0.0
            for group_id in groups:
                if group_id in structured_intras:
                    continue
                if group_id in module_by_head:
                    state, update_aux = update_attention_module(
                        graph,
                        self.objective,
                        reference_data,
                        target_data,
                        state,
                        module_by_head[group_id],
                        scheduled_groups=scheduled,
                        record_ambiguity=step.record_ambiguity,
                    )
                elif group_id in gqa_by_kv:
                    state, update_aux = update_gqa_attention_module(
                        graph,
                        self.objective,
                        reference_data,
                        target_data,
                        state,
                        gqa_by_kv[group_id],
                        scheduled_groups=scheduled,
                        record_ambiguity=step.record_ambiguity,
                    )
                else:
                    state, update_aux = solver.update(
                        graph, reference_data, target_data, state, group_id
                    )
                sweep_delta = max(sweep_delta, float(update_aux["delta"]))
                ambiguity.update(update_aux.get("ambiguity", {}))
            max_delta = sweep_delta
            if sweep_delta <= step.tolerance:
                break
        aux = {
            "solver": "lap",
            "sweeps": sweeps,
            "max_delta": max_delta,
            "converged": max_delta <= step.tolerance,
        }
        if ambiguity:
            aux["ambiguity"] = ambiguity
        return state, aux

    def _run_procrustes_step(self, graph, reference_data, target_data, state, step):
        scheduled = _scheduled_groups(graph, step)
        eligible = tuple(
            group_id
            for group_id in scheduled
            if graph.groups[group_id].transform_family == "orthogonal"
        )
        if step.groups is not None or step.components is not None:
            ineligible = sorted(set(scheduled) - set(eligible))
            if ineligible:
                raise ValueError(
                    "procrustes steps only update 'orthogonal' groups; "
                    "scheduled non-orthogonal group(s): " + ", ".join(ineligible)
                )
        if not eligible:
            raise ValueError(
                "procrustes step scheduled, but the graph declares no "
                "'orthogonal' groups."
            )
        solver = ProcrustesGroupSolver(self.objective)
        sweeps = 0
        max_delta = 0.0
        for _ in range(step.max_sweeps):
            sweeps += 1
            sweep_delta = 0.0
            for group_id in eligible:
                state, update_aux = solver.update(
                    graph, reference_data, target_data, state, group_id
                )
                sweep_delta = max(sweep_delta, float(update_aux["delta"]))
            max_delta = sweep_delta
            if sweep_delta <= step.tolerance:
                break
        return state, {
            "solver": "procrustes",
            "sweeps": sweeps,
            "max_delta": max_delta,
            "groups": list(eligible),
        }


def _unbatch_state(state: TransformState, *, sample_index: int) -> TransformState:
    transforms = {}
    for group_id, transform in state.transforms.items():
        if isinstance(transform, PermutationTransform):
            indices = np.asarray(transform.indices)
            transforms[group_id] = PermutationTransform(
                indices[sample_index] if indices.ndim == 2 else indices
            )
        elif isinstance(transform, SignedPermutationTransform):
            indices = np.asarray(transform.indices)
            signs = np.asarray(transform.signs)
            transforms[group_id] = SignedPermutationTransform(
                indices[sample_index] if indices.ndim == 2 else indices,
                signs[sample_index] if signs.ndim == 2 else signs,
            )
        else:
            matrix = np.asarray(transform.matrix)
            transforms[group_id] = MatrixTransform(
                matrix[sample_index] if matrix.ndim == 3 else matrix,
                family=transform.family,
            )
    return TransformState(
        group_order=state.group_order,
        transforms=transforms,
        logits=None,
        metadata=dict(state.metadata),
    )


__all__ = ["SolverSequence"]
