"""Tests for the ``python -m align.benchmarks`` CLI."""

import json
import pickle
from pathlib import Path

import jax
import numpy as np
import pytest

from align.benchmarks import __main__ as benchmarks_cli
from align.benchmarks.harness import BenchmarkRecord
from align.benchmarks.synthetic import make_mlp_orbit_case, permutation_matrix
from align.matching import TransformState


def _save_pytree_npz(path: Path, pytree) -> None:
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    payload = {f"arr_{idx}": np.asarray(leaf) for idx, leaf in enumerate(leaves)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _write_experiment(tmp_path: Path) -> Path:
    case = make_mlp_orbit_case(seed=3)
    graph, base = case.graph, case.reference
    perms = {
        "mlp/h0": permutation_matrix([2, 4, 1, 0, 3]),
        "mlp/h1": permutation_matrix([3, 1, 0, 2]),
    }
    root = tmp_path / "run42"
    rng = np.random.default_rng(0)
    for chain_id in range(2):
        mode = (
            base
            if chain_id == 0
            else graph.apply_transforms(
                base, TransformState.from_transforms(graph, perms)
            )
        )
        for sample_id in range(4):
            noisy = jax.tree_util.tree_map(
                lambda leaf: (
                    np.asarray(leaf)
                    + 0.01 * rng.standard_normal(np.shape(leaf)).astype(np.float32)
                ),
                mode,
            )
            _save_pytree_npz(
                root / "samples" / str(chain_id) / f"sample_{sample_id}.npz", noisy
            )
    tree_path = root / "tree"
    with tree_path.open("wb") as handle:
        pickle.dump(jax.tree_util.tree_structure(base), handle)
    return root


def test_compare_writes_report_and_markdown(tmp_path):
    output = tmp_path / "report.json"
    markdown = tmp_path / "tables" / "compare.md"

    code = benchmarks_cli.main(
        [
            "compare",
            "--objective",
            "euclidean",
            "--schedules",
            "lap",
            "--cases",
            "mlp",
            "--seeds",
            "0",
            "--no-posterior",
            "--no-performance",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ]
    )

    assert code == 0
    report = json.loads(output.read_text())
    assert [record["name"] for record in report["records"]] == [
        "orbit/mlp/lap/euclidean/seed0"
    ]
    assert "## orbit" in markdown.read_text()


def test_compare_rejects_unknown_schedule():
    with pytest.raises(SystemExit):
        benchmarks_cli.main(
            ["compare", "--objective", "euclidean", "--schedules", "bogus"]
        )


def test_compare_parses_inverse_fisher_synthetic_posterior():
    args = benchmarks_cli.build_parser().parse_args(
        [
            "compare",
            "--objective",
            'relative_fisher={"calibration":"relative_fisher"}',
            "--synthetic-posterior",
            "mlp",
            "--posterior-noise-mode",
            "inverse_fisher",
            "--posterior-noise-scale",
            "0.6",
        ]
    )
    assert args.synthetic_posterior == "mlp"
    assert args.posterior_noise_mode == "inverse_fisher"
    assert args.posterior_noise_scale == 0.6


def _stub_evaluator(experiment_root, *, split, n_inputs, batch_size):
    """Stand in for a sampler-backed evaluator factory (see ``--evaluator``)."""

    import jax.numpy as jnp

    from align.benchmarks.synthetic import mlp_apply

    return {
        "apply_fn": mlp_apply,
        "inputs": jnp.ones((n_inputs, 3)),
        "activation_fn": None,
        "prediction_transform": None,
        "functional_batch_size": 3,
    }


def test_posterior_command_runs_a_real_experiment_directory(tmp_path):
    root = _write_experiment(tmp_path)
    output = tmp_path / "posterior.json"

    code = benchmarks_cli.main(
        [
            "posterior",
            "--experiment-root",
            str(root),
            "--architecture",
            "mlp",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    (record,) = json.loads(output.read_text())["records"]
    assert record["kind"] == "posterior"
    assert record["name"] == "posterior/experiment_run42/lap/euclidean"
    metrics = record["metrics"]
    assert metrics["after_split_rhat_max"] < metrics["before_split_rhat_max"]

    custom_output = tmp_path / "posterior_custom.json"
    code = benchmarks_cli.main(
        [
            "posterior",
            "--experiment-root",
            str(root),
            "--architecture",
            "mlp",
            "--schedule",
            '[{"solver": "lap", "max_sweeps": 5, "tolerance": 0.0}]',
            "--output",
            str(custom_output),
        ]
    )
    assert code == 0
    (record,) = json.loads(custom_output.read_text())["records"]
    assert record["name"] == "posterior/experiment_run42/custom/euclidean"


def test_posterior_command_reports_functional_metrics(tmp_path):
    root = _write_experiment(tmp_path)
    output = tmp_path / "posterior_functional.json"

    code = benchmarks_cli.main(
        [
            "posterior",
            "--experiment-root",
            str(root),
            "--architecture",
            "mlp",
            "--evaluator",
            "tests.benchmarks.test_cli:_stub_evaluator",
            "--functional-inputs",
            "7",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    (record,) = json.loads(output.read_text())["records"]
    metrics = record["metrics"]
    assert metrics["before_chain_mean_prediction_rmse_mean"] > 0.0
    assert metrics["before_function_cross_rms_distance"] > 0.0
    assert metrics["before_weight_averaging_gap"] is not None
    assert metrics["function_drift_max"] < 1e-5
    assert metrics["function_drift_relative_rmse"] < 1e-6


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("no_colon_here", "module:function"),
        ("align.benchmarks.harness:not_a_real_name", "has no"),
        ("align.benchmarks.does_not_exist:thing", "not importable"),
    ],
)
def test_posterior_command_rejects_a_bad_evaluator_spec(tmp_path, spec, match):
    root = _write_experiment(tmp_path)

    with pytest.raises(SystemExit, match=match):
        benchmarks_cli.main(
            [
                "posterior",
                "--experiment-root",
                str(root),
                "--architecture",
                "mlp",
                "--evaluator",
                spec,
            ]
        )


def test_posterior_command_requires_architecture(tmp_path):
    root = _write_experiment(tmp_path)

    with pytest.raises(SystemExit, match="--architecture"):
        benchmarks_cli.main(["posterior", "--experiment-root", str(root)])


def test_regression_exit_code_reflects_violations(monkeypatch, capsys):
    skipped = BenchmarkRecord(
        name="orbit/fake/lap/euclidean/seed0", kind="orbit", skipped="unsupported"
    )
    monkeypatch.setattr(
        benchmarks_cli, "run_regression_suite", lambda *, fast: [skipped]
    )

    code = benchmarks_cli.main(["regression", "--fast"])

    assert code == 1
    assert "threshold violation" in capsys.readouterr().err
