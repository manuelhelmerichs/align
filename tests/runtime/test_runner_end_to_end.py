"""End-to-end runner tests for staged alignment and artifact output."""

import json
import logging
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from align.architectures.mlp import MLPRecipe
from align.architectures.residual_convnet import ResidualConvNetRecipe
from align.config import (
    ArchitectureConfig,
    CanonicalizeConfig,
    MatchConfig,
    PathConfig,
    RunConfig,
    RuntimeConfig,
    SelectionConfig,
    SolverStep,
)
from align.matching import TransformState
from align.run_state import RunState
from align.runtime import RunArtifactStore
from align.runtime.artifacts import read_transforms_artifact
from align.runtime.runner import AlignmentRunner
from align.sample_manifest import SampleManifest
from align.symmetry import ResidualChannelTie

_FAMILY_BY_SCENARIO = {"mlp": "mlp", "residual_convnet": "residual_convnet"}


def calculate_r_hat(
    flattened: np.ndarray,
    chain_labels: np.ndarray,
    num_chains: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Gelman-Rubin R-hat without pulling in visualization deps."""
    overall_mean = np.mean(flattened, axis=0)
    within_chain_vars = []
    chain_means = []
    for chain_id in range(num_chains):
        chain_samples = flattened[chain_labels == chain_id]
        if len(chain_samples) < 2:
            continue
        chain_var = np.var(chain_samples, axis=0, ddof=1)
        within_chain_vars.append(chain_var)
        chain_mean = np.mean(chain_samples, axis=0)
        chain_means.append((chain_mean - overall_mean) ** 2)
    within_chain_var = np.mean(within_chain_vars, axis=0)
    chain_means = np.array(chain_means)
    M = num_chains
    N = len(flattened) / M
    between_chain_var = np.sum(chain_means, axis=0) * (N / max(M - 1, 1))
    v_hat = ((N - 1) / N) * within_chain_var + ((M + 1) / (M * N)) * between_chain_var
    r_hat = np.sqrt(v_hat / within_chain_var)
    return within_chain_var, between_chain_var, r_hat


def _save_pytree_npz(path: Path, pytree) -> None:
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    payload = {f"arr_{idx}": np.asarray(leaf) for idx, leaf in enumerate(leaves)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _load_pytree_npz(path: Path, treedef):
    with np.load(path, allow_pickle=False) as data:
        leaves = [jnp.asarray(data[key]) for key in data.files]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _flatten_params(params) -> np.ndarray:
    leaves, _ = jax.tree_util.tree_flatten(params)
    return np.concatenate([np.ravel(np.asarray(leaf)) for leaf in leaves], axis=0)


def _perm_matrix(indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    n = int(indices.shape[0])
    mat = np.zeros((n, n), dtype=np.float64)
    mat[np.arange(n), indices] = 1.0
    return mat


def _residual_convnet_residual_topology() -> dict[str, object]:
    return {
        "nodes": [
            {"id": "input", "kind": "input"},
            {"id": "core/Conv_0", "kind": "conv"},
            {"id": "core/Conv_1", "kind": "conv"},
            {"id": "core/Conv_2", "kind": "conv"},
            {"id": "residual_add", "kind": "add"},
            {"id": "core/Dense_0", "kind": "dense"},
        ],
        "edges": [
            {"source": "input", "target": "core/Conv_0"},
            {"source": "core/Conv_0", "target": "core/Conv_1"},
            {"source": "core/Conv_1", "target": "core/Conv_2"},
            {"source": "core/Conv_0", "target": "residual_add"},
            {"source": "core/Conv_2", "target": "residual_add"},
            {"source": "residual_add", "target": "core/Dense_0"},
        ],
    }


def _mlp_apply(params, x: jnp.ndarray) -> jnp.ndarray:
    subtree = params["params"]["fcn"]
    h = x
    layer_names = sorted(subtree.keys())
    for name in layer_names[:-1]:
        w = jnp.asarray(subtree[name]["kernel"])
        b = jnp.asarray(subtree[name]["bias"])
        z = (w.T @ h.T).T + b
        h = jnp.maximum(0.0, z)
    last = layer_names[-1]
    w = jnp.asarray(subtree[last]["kernel"])
    b = jnp.asarray(subtree[last]["bias"])
    return (w.T @ h.T).T + b


def _conv1x1(x: jnp.ndarray, kernel: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    w = kernel[0, 0, :, :]
    return jnp.einsum("oc,bhwc->bhwo", w.T, x) + bias


def _tiny_residual_convnet_apply(params, x: jnp.ndarray) -> jnp.ndarray:
    core = params["core"]
    residual = _conv1x1(
        x,
        jnp.asarray(core["Conv_0"]["kernel"]),
        jnp.asarray(core["Conv_0"]["bias"]),
    )
    residual = jnp.maximum(0.0, residual)
    h = _conv1x1(
        residual,
        jnp.asarray(core["Conv_1"]["kernel"]),
        jnp.asarray(core["Conv_1"]["bias"]),
    )
    h = jnp.maximum(0.0, h)
    h = _conv1x1(
        h,
        jnp.asarray(core["Conv_2"]["kernel"]),
        jnp.asarray(core["Conv_2"]["bias"]),
    )
    h = jnp.maximum(0.0, residual + h)
    h = jnp.mean(h, axis=(1, 2))
    w = jnp.asarray(core["Dense_0"]["kernel"])
    b = jnp.asarray(core["Dense_0"]["bias"])
    return (w.T @ h.T).T + b


def _make_mlp_params(
    *, key: jax.Array, in_dim: int, hidden: tuple[int, ...], out_dim: int
):
    sizes = (in_dim, *hidden, out_dim)
    keys = jax.random.split(key, len(sizes) - 1)
    fcn = {}
    for idx, (k, din, dout) in enumerate(zip(keys, sizes[:-1], sizes[1:], strict=True)):
        w_key, b_key = jax.random.split(k)
        w = jax.random.normal(w_key, (din, dout), dtype=jnp.float32) * 0.7
        b = jax.random.normal(b_key, (dout,), dtype=jnp.float32) * 0.1
        fcn[f"dense{idx}"] = {"kernel": w, "bias": b}
    return {"params": {"fcn": fcn}}


def _build_experiment(
    tmp_path: Path, *, architecture: str, num_samples: int
) -> tuple[Path, Path, dict[str, np.ndarray], jnp.ndarray, callable]:
    exp_root = tmp_path / f"exp_{architecture}"
    samples_dir = exp_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(0 if architecture == "mlp" else 7)

    baseline_preds: dict[str, np.ndarray] = {}

    if architecture == "mlp":
        in_dim = 3
        hidden = (5, 4)
        out_dim = 2

        base_key, noise_key, x_key = jax.random.split(key, 3)
        ref = _make_mlp_params(
            key=base_key, in_dim=in_dim, hidden=hidden, out_dim=out_dim
        )
        recipe = MLPRecipe()
        graph = recipe.build_graph(ref)
        perm = np.array([2, 4, 1, 0, 3], dtype=np.int32)
        pmat = _perm_matrix(perm)

        x = jax.random.normal(x_key, (9, in_dim), dtype=jnp.float32)
        predict = _mlp_apply

        chain0: list = []
        chain1: list = []
        for s in range(num_samples):
            k = jax.random.fold_in(noise_key, s)
            delta = jax.tree_util.tree_map(
                lambda a, key=k: (
                    jax.random.normal(key, a.shape, dtype=jnp.float32) * 0.02
                ),
                ref,
            )
            sample0 = jax.tree_util.tree_map(lambda a, d: a + d, ref, delta)
            sample1 = graph.apply_transforms(
                sample0, TransformState.from_transforms(graph, {"mlp/h0": pmat})
            )
            chain0.append(sample0)
            chain1.append(sample1)

        treedef = jax.tree_util.tree_structure(ref)

        for chain_id, chain in enumerate((chain0, chain1)):
            chain_dir = samples_dir / str(chain_id)
            chain_dir.mkdir(parents=True, exist_ok=True)
            for sample_id, params in enumerate(chain):
                sample_path = chain_dir / f"sample_{sample_id}.npz"
                _save_pytree_npz(sample_path, params)
                pred = np.asarray(predict(params, x))
                baseline_preds[f"chain{chain_id}_sample{sample_id}"] = pred

    elif architecture == "residual_convnet":
        k0, k1, k2, kd, xk, noise_key = jax.random.split(key, 6)
        ref = {
            "core": {
                "Conv_0": {
                    "kernel": jax.random.normal(k0, (1, 1, 1, 2), dtype=jnp.float32),
                    "bias": jax.random.normal(k0, (2,), dtype=jnp.float32) * 0.1,
                },
                "Conv_1": {
                    "kernel": jax.random.normal(k1, (1, 1, 2, 3), dtype=jnp.float32),
                    "bias": jax.random.normal(k1, (3,), dtype=jnp.float32) * 0.1,
                },
                "Conv_2": {
                    "kernel": jax.random.normal(k2, (1, 1, 3, 2), dtype=jnp.float32),
                    "bias": jax.random.normal(k2, (2,), dtype=jnp.float32) * 0.1,
                },
                "Dense_0": {
                    "kernel": jax.random.normal(kd, (2, 1), dtype=jnp.float32),
                    "bias": jax.random.normal(kd, (1,), dtype=jnp.float32) * 0.1,
                },
            }
        }

        residual_topology = _residual_convnet_residual_topology()
        (exp_root / "module_graph.json").write_text(json.dumps(residual_topology))

        recipe = ResidualConvNetRecipe(residual_topology=residual_topology)
        graph = recipe.build_graph(ref)
        forward_perms = {
            group_id: _perm_matrix(np.roll(np.arange(group.size), 1))
            for group_id, group in graph.groups.items()
        }

        x = jax.random.normal(xk, (4, 3, 3, 1), dtype=jnp.float32)
        predict = _tiny_residual_convnet_apply

        chain0: list = []
        chain1: list = []
        for s in range(num_samples):
            k = jax.random.fold_in(noise_key, s)
            delta = jax.tree_util.tree_map(
                lambda a, key=k: (
                    jax.random.normal(key, a.shape, dtype=jnp.float32) * 0.02
                ),
                ref,
            )
            sample0 = jax.tree_util.tree_map(lambda a, d: a + d, ref, delta)
            sample1 = graph.apply_transforms(
                sample0, TransformState.from_transforms(graph, forward_perms)
            )
            chain0.append(sample0)
            chain1.append(sample1)

        treedef = jax.tree_util.tree_structure(ref)

        for chain_id, chain in enumerate((chain0, chain1)):
            chain_dir = samples_dir / str(chain_id)
            chain_dir.mkdir(parents=True, exist_ok=True)
            for sample_id, params in enumerate(chain):
                sample_path = chain_dir / f"sample_{sample_id}.npz"
                _save_pytree_npz(sample_path, params)
                pred = np.asarray(predict(params, x))
                baseline_preds[f"chain{chain_id}_sample{sample_id}"] = pred

    else:  # pragma: no cover
        raise ValueError(f"Unsupported architecture '{architecture}'.")

    tree_path = exp_root / "tree"
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    with tree_path.open("wb") as handle:
        pickle.dump(treedef, handle)

    return exp_root, tree_path, baseline_preds, x, predict


def _run_align(
    *, exp_root: Path, tree_path: Path, architecture: str, solver: str
) -> tuple[Path, SampleManifest]:
    samples_dir = exp_root / "samples"
    output_dir = exp_root / f"align_{solver}"
    if solver == "lap":
        solvers = (SolverStep(solver="lap", max_sweeps=15, tolerance=0.0),)
    elif solver == "sinkhorn":
        solvers = (
            SolverStep(
                solver="sinkhorn",
                max_steps=25,
                tolerance=1e-4,
                sinkhorn_iterations=25,
                record_loss_history=False,
                init_scale=0.0,
            ),
            SolverStep(solver="lap", max_sweeps=5, tolerance=0.0),
        )
    else:  # pragma: no cover
        raise ValueError(solver)

    selection = SelectionConfig(
        chain_indices=[0, 1],
        samples_per_chain=3,
        reference_chain=0,
        reference_sample=0,
    )
    canonicalize_stage = architecture == "mlp"
    pipeline = ("canonicalize", "match") if canonicalize_stage else ("match",)
    config = RunConfig(
        paths=PathConfig(
            experiment_root=exp_root,
            samples_dir=samples_dir,
            tree_path=tree_path,
            output_dir=output_dir,
        ),
        architecture=ArchitectureConfig(family=_FAMILY_BY_SCENARIO[architecture]),
        selection=selection,
        pipeline=pipeline,
        canonicalize=(
            CanonicalizeConfig(activation="relu") if canonicalize_stage else None
        ),
        match=MatchConfig(solvers=solvers),
        runtime=RuntimeConfig(
            parallelism=1, save_intermediate=False, validate_artifacts_on_resume=False
        ),
    )

    manifest_path = output_dir / "state" / "sample_manifest.json"
    manifest = SampleManifest.build(
        samples_dir=samples_dir,
        tree_path=tree_path,
        chain_indices=selection.chain_indices,
        samples_per_chain=selection.samples_per_chain,
        sample_step=selection.sample_step,
        max_total=selection.max_total,
        reference_chain=selection.reference_chain,
        reference_sample=selection.reference_sample,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(manifest_path)

    state_dir = output_dir / "state"
    run_state = RunState(
        path=state_dir / "run_state.json",
        experiment_root=exp_root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        config_digest="test-digest",
        manifest_digest=manifest.digest(),
        stages=config.active_stages(),
        reference={
            "chain": manifest.reference_chain,
            "sample": manifest.reference_sample,
            "index": manifest.reference_index,
        },
        filters=manifest.filters,
        total_samples=manifest.total,
        runtime_settings={},
        metadata={
            "architecture": config.architecture.family,
            "match_objective": config.match.objective.type,
            "match_solvers": config.match.solvers_payload(),
        },
    )
    run_state.save()

    artifact_store = RunArtifactStore(
        manifest=manifest,
        output_dir=output_dir,
        stages=config.active_stages(),
        save_intermediate=False,
    )
    runner = AlignmentRunner(
        config=config,
        manifest=manifest,
        run_state=run_state,
        artifact_store=artifact_store,
        progress_logger=logging.getLogger("test_end_to_end"),
    )
    runner.execute()

    assert run_state.completed
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["stages"] == list(pipeline)
    assert (output_dir / "sample_diagnostics.json").exists()
    diagnostics = json.loads((output_dir / "sample_diagnostics.json").read_text())
    assert all("objective_final" in item["match"] for item in diagnostics)
    assert all(item["match"]["steps"] for item in diagnostics)
    return output_dir, manifest


@pytest.mark.parametrize("architecture", ["mlp", "residual_convnet"])
@pytest.mark.parametrize("solver", ["lap", "sinkhorn"])
def test_align_runner_end_to_end_preserves_predictions(
    tmp_path: Path, architecture: str, solver: str
) -> None:
    exp_root, tree_path, baseline_preds, x, predict = _build_experiment(
        tmp_path, architecture=architecture, num_samples=3
    )

    output_dir, manifest = _run_align(
        exp_root=exp_root,
        tree_path=tree_path,
        architecture=architecture,
        solver=solver,
    )

    # Load the producer treedef (the output also copies it into aligned_samples/tree).
    with tree_path.open("rb") as handle:
        treedef = pickle.load(handle)

    # Artifacts exist.
    assert (output_dir / "aligned_samples").exists()
    assert (output_dir / "transforms").exists()
    if architecture == "mlp":
        assert (output_dir / "scales").exists()
    else:
        assert not (output_dir / "scales").exists()

    if architecture == "residual_convnet":
        residual_topology_path = exp_root / "module_graph.json"
        assert residual_topology_path.exists()
        reference_params = _load_pytree_npz(
            exp_root / "samples" / "0" / "sample_0.npz", treedef
        )
        graph = ResidualConvNetRecipe(
            residual_topology=str(residual_topology_path)
        ).build_graph(reference_params)
        ties = [
            constraint
            for constraint in graph.constraints
            if isinstance(constraint, ResidualChannelTie)
        ]
        assert len(ties) == 1
        assert ties[0].groups == (graph.metadata["group_order"][0],)
        assert set(ties[0].members) == {"core/Conv_0", "core/Conv_2"}

        non_reference = next(
            record for record in manifest.records if record.chain_id == 1
        )
        transform_path = output_dir / "transforms" / non_reference.relative_path
        assert transform_path.exists()
        transforms, metadata = read_transforms_artifact(transform_path)
        assert set(transforms) == set(graph.groups)
        assert metadata["transform_families"] == {
            group_id: group.transform_family for group_id, group in graph.groups.items()
        }
        for indices in transforms.values():
            indices = np.asarray(indices, dtype=np.int64)
            np.testing.assert_array_equal(
                np.sort(indices), np.arange(indices.size, dtype=np.int64)
            )

    # Aligned predictions match pre-align predictions for every sample.
    for record in manifest.records:
        aligned_path = output_dir / "aligned_samples" / record.relative_path
        assert aligned_path.exists(), f"Missing aligned sample for {record.label}"
        params = _load_pytree_npz(aligned_path, treedef)
        pred_after = np.asarray(predict(params, x))
        pred_before = baseline_preds[record.label]
        np.testing.assert_allclose(pred_after, pred_before, rtol=0, atol=1e-5)

    # Weight-space mixing should improve after alignment when chains differ by a symmetry
    # (here: a fixed hidden-unit/channel permutation).
    flat_before = []
    flat_after = []
    chain_labels = []
    for record in manifest.records:
        before_path = exp_root / "samples" / record.relative_path
        after_path = output_dir / "aligned_samples" / record.relative_path
        before_params = _load_pytree_npz(before_path, treedef)
        after_params = _load_pytree_npz(after_path, treedef)
        flat_before.append(_flatten_params(before_params))
        flat_after.append(_flatten_params(after_params))
        chain_labels.append(int(record.chain_id))

    flat_before_arr = np.stack(flat_before, axis=0)
    flat_after_arr = np.stack(flat_after, axis=0)
    chain_labels_arr = np.asarray(chain_labels)

    _, between0, rhat0 = calculate_r_hat(
        flat_before_arr, chain_labels_arr, num_chains=2
    )
    _, between1, rhat1 = calculate_r_hat(flat_after_arr, chain_labels_arr, num_chains=2)

    assert float(np.mean(between1)) < float(np.mean(between0))
    assert float(np.mean(rhat1)) < float(np.mean(rhat0))

    # At least one transform artifact should exist for the non-reference chain.
    transform_chain_dir = output_dir / "transforms" / "1"
    assert transform_chain_dir.exists()
    assert any(path.suffix == ".npz" for path in transform_chain_dir.iterdir())
