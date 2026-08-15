"""Tests for CLI config resolution, pipeline flags, and command behavior."""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from align import _jax_platforms
from align.cli import main as align_main

ROOT = Path(__file__).resolve().parents[1]


def test_cli_pipeline_override_prunes_inactive_stage_sections(tmp_path):
    from align.cli import _cli_overrides, build_parser
    from align.config import load_align_config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "pipeline": ["canonicalize", "match"],
                "canonicalize": {"activation": "relu"},
                "match": {"objective": {"type": "euclidean"}},
            }
        )
    )
    args = build_parser().parse_args([str(path), "--match-only"])
    config = load_align_config(path, _cli_overrides(args))
    assert config.pipeline == ("match",)
    assert config.canonicalize is None
    assert config.match is not None


def test_cli_architecture_override_replaces_prior_family_options(tmp_path):
    from align.cli import _cli_overrides, build_parser
    from align.config import load_align_config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "architecture": {
                    "family": "residual_convnet",
                    "parameter_root": "core",
                    "linear_residual_free": True,
                },
                "pipeline": ["match"],
            }
        )
    )
    args = build_parser().parse_args([str(path), "--architecture", "mlp"])
    config = load_align_config(path, _cli_overrides(args))
    assert config.architecture.family == "mlp"
    assert config.architecture.options == {}


def test_config_digest_includes_every_active_stage_config():
    from align.cli import _digest_payload

    payload = {
        "architecture": {"family": "mlp"},
        "selection": {},
        "pipeline": ["center_softmax_head", "match"],
        "center_softmax_head": {"head": "core.Dense_0"},
        "match": {"objective": {"type": "euclidean"}},
        "resolved_paths": {},
    }
    digest_payload = _digest_payload(payload)
    assert digest_payload["center_softmax_head"] == {"head": "core.Dense_0"}
    assert digest_payload["match"] == payload["match"]


def test_force_gpu_fails_when_no_gpu_backend_is_available(monkeypatch):
    from align.cli import _configure_platform_preferences
    from align.config import RunConfig

    config = RunConfig.from_mapping(
        {"pipeline": ["match"], "runtime": {"force_gpu": True}}
    )
    monkeypatch.setattr("align.cli.configure_jax_platforms", lambda **kwargs: "cpu")
    with pytest.raises(RuntimeError, match="force-gpu"):
        _configure_platform_preferences(config)


def test_package_root_declares_full_lazy_api_without_importing_jax():
    import subprocess

    code = """
import sys
import align

expected = {
    'AlignmentRunner',
    'ArchitectureRecipe',
    'RunConfig',
    'ScaleCanonicalizer',
    'SymmetryGraph',
    'TransformState',
    'match_sample',
}
missing = expected - set(align.__all__)
if missing:
    raise SystemExit(f'missing root exports: {sorted(missing)}')
if any(name == 'jax' or name.startswith('jax.') for name in sys.modules):
    raise SystemExit('import align initialized JAX')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_gpu_platform_configuration_supports_mps(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(_jax_platforms, "_preferred_gpu_platform", lambda: "mps")
    monkeypatch.setattr(_jax_platforms, "_probe_platform", lambda platform: True)

    preference = _jax_platforms.configure_jax_platforms("gpu")

    assert preference == "gpu"
    assert os.environ["JAX_PLATFORMS"] == "mps,cpu"
    assert _jax_platforms.is_gpu_platform("mps")


def test_cuda_hardware_without_jax_plugin_is_not_selected(monkeypatch):
    monkeypatch.setattr(_jax_platforms, "_cuda_plugin_installed", lambda: False)
    monkeypatch.setattr(_jax_platforms.shutil, "which", lambda _name: "/bin/nvidia-smi")
    assert not _jax_platforms._looks_like_cuda_system()


def test_broken_accelerator_probe_falls_back_before_jax_import(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(_jax_platforms, "_preferred_gpu_platform", lambda: "mps")
    monkeypatch.setattr(_jax_platforms, "_probe_platform", lambda _platform: False)
    assert _jax_platforms.configure_jax_platforms("gpu") == "cpu"
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def _write_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weight=np.array([1.0], dtype=np.float32))


def _make_experiment(tmp_path: Path) -> Path:
    exp_root = tmp_path / "experiment"
    samples_dir = exp_root / "samples" / "0"
    _write_sample(samples_dir / "sample_0.npz")
    tree_path = exp_root / "tree"
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    with tree_path.open("wb") as handle:
        pickle.dump({"treedef": True}, handle)
    return exp_root


def test_cli_dry_run_produces_state(tmp_path) -> None:
    exp_root = _make_mlp_experiment(tmp_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  experiment_root: {exp_root}
pipeline: [canonicalize]
canonicalize:
  activation: relu
selection:
  reference_chain: 0
  reference_sample: 0
runtime:
  resume: false
        """
    )

    output_dir = exp_root / "align_run"
    args = [
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ]

    align_main(args)

    state_dir = output_dir / "state"
    manifest_path = state_dir / "sample_manifest.json"
    run_state_path = state_dir / "run_state.json"
    summary_path = state_dir / "dry_run_summary.json"

    assert manifest_path.exists()
    assert run_state_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["pending_samples"] == 1
    assert summary["manifest"]["total_samples"] == 1
    assert summary["output_dir"] == str(output_dir)
    assert summary["stages"] == ["canonicalize"]


def test_cli_match_only_override(tmp_path):
    exp_root = _make_mlp_experiment(tmp_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  experiment_root: {exp_root}
pipeline: [canonicalize, match]
selection:
  reference_chain: 0
  reference_sample: 0
runtime:
  resume: false
        """
    )

    output_dir = exp_root / "align_override"
    args = [
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--match-only",
        "--dry-run",
    ]

    align_main(args)

    run_state_path = output_dir / "state" / "run_state.json"
    assert run_state_path.exists()
    run_state = json.loads(run_state_path.read_text())
    assert run_state["stages"] == ["match"]
    assert "stage_completion" not in run_state


def test_config_import_does_not_trigger_jax_initialization():
    """Verify that importing config.py doesn't prematurely initialize JAX.

    This is critical for the CLI to work correctly: the platform preferences
    (CPU vs GPU) must be set BEFORE JAX is imported. If config.py or its
    imports trigger JAX initialization, the CLI's platform selection would
    be ignored.
    """
    import subprocess

    # Run in a subprocess to ensure a clean environment
    code = """
import os
import sys

# Track if JAX gets imported
class JaxImportDetector:
    jax_imported = False

    def find_module(self, name, path=None):
        if name == 'jax' or (isinstance(name, str) and name.startswith('jax.')):
            JaxImportDetector.jax_imported = True
        return None

sys.meta_path.insert(0, JaxImportDetector())

# Import config - this should NOT trigger JAX
from align.config import RunConfig, load_align_config

# Check result
if JaxImportDetector.jax_imported:
    print("FAIL: JAX was imported during config import")
    sys.exit(1)
else:
    print("PASS: config import did not trigger JAX")
    sys.exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    assert result.returncode == 0, (
        f"Config import triggered JAX: {result.stdout}\n{result.stderr}"
    )


def test_jax_platform_respects_cpu_preference():
    """Verify that setting CPU preference before JAX import works correctly.

    This ensures the platform configuration mechanism works as expected,
    particularly that LAP schedules (CPU-based) don't fail on systems
    with GPUs when configured for CPU-only operation.
    """
    import subprocess

    code = """
import os
import sys

# Set CPU preference before any JAX imports
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Now import JAX
import jax

# Verify we're on CPU
devices = jax.devices()
if len(devices) != 1 or devices[0].platform != "cpu":
    print(f"FAIL: Expected CPU device, got {devices}")
    sys.exit(1)

# Verify PRNGKey works
key = jax.random.PRNGKey(42)
print(f"PASS: CPU platform active, PRNGKey works: {key}")
sys.exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    assert result.returncode == 0, (
        f"CPU platform setup failed: {result.stdout}\n{result.stderr}"
    )


def test_cli_print_config_exits_without_state(tmp_path, capsys) -> None:
    exp_root = _make_experiment(tmp_path)

    config_path = tmp_path / "config.yaml"
    output_dir = exp_root / "align_print"
    config_path.write_text(
        f"""
paths:
  experiment_root: {exp_root}
  output_dir: {output_dir}
pipeline: [canonicalize]
selection:
  reference_chain: 0
  reference_sample: 0
runtime:
  resume: false
        """
    )

    align_main([str(config_path), "--print-config"])

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    resolved = payload["resolved_paths"]
    assert resolved["experiment_root"] == str(exp_root.resolve())
    assert resolved["output_dir"] == str(output_dir.resolve())
    assert payload["stages"] == ["canonicalize"]
    state_dir = output_dir / "state"
    assert (state_dir / "sample_manifest.json").exists()
    assert not (state_dir / "run_state.json").exists()


def test_cli_list_samples_outputs_manifest(tmp_path, capsys) -> None:
    exp_root = _make_experiment(tmp_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  experiment_root: {exp_root}
pipeline: [canonicalize]
selection:
  reference_chain: 0
  reference_sample: 0
runtime:
  resume: false
        """
    )

    output_dir = exp_root / "align_list"
    align_main([str(config_path), "--output-dir", str(output_dir), "--list-samples"])

    out = capsys.readouterr().out
    assert "chain0_sample0" in out
    assert "0/sample_0.npz" in out
    assert (output_dir / "state" / "run_state.json").exists()
    assert not (output_dir / "aligned_samples").exists()


def _make_mlp_experiment(tmp_path: Path) -> Path:
    import jax

    exp_root = tmp_path / "experiment_mlp"
    rng = np.random.default_rng(0)
    fcn = {}
    for idx, (din, dout) in enumerate(((3, 5), (5, 4), (4, 2))):
        fcn[f"dense{idx}"] = {
            "kernel": rng.normal(size=(din, dout)).astype(np.float32),
            "bias": rng.normal(size=(dout,)).astype(np.float32),
        }
    params = {"params": {"fcn": fcn}}
    leaves, treedef = jax.tree_util.tree_flatten(params)
    sample_path = exp_root / "samples" / "0" / "sample_0.npz"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        sample_path,
        **{f"arr_{idx}": np.asarray(leaf) for idx, leaf in enumerate(leaves)},
    )
    tree_path = exp_root / "tree"
    with tree_path.open("wb") as handle:
        pickle.dump(treedef, handle)
    return exp_root


@pytest.mark.parametrize("parallelism", [1, 2])
def test_cli_refinement_writes_final_pass_artifacts_and_diagnostic(
    tmp_path, parallelism
) -> None:
    exp_root = _make_mlp_experiment(tmp_path)

    source_path = exp_root / "samples" / "0" / "sample_0.npz"
    with np.load(source_path) as source:
        base = {key: np.asarray(source[key]) for key in source.files}
    rng = np.random.default_rng(11)
    for sample_id in (1, 2):
        perturbed = {
            key: value
            + 0.03 * rng.standard_normal(value.shape).astype(value.dtype, copy=False)
            for key, value in base.items()
        }
        np.savez(
            exp_root / "samples" / "0" / f"sample_{sample_id}.npz",
            **perturbed,
        )

    output_dir = exp_root / "align_refined"
    config_path = tmp_path / "refined.yaml"
    config_path.write_text(
        f"""
paths:
  experiment_root: {exp_root}
  output_dir: {output_dir}
architecture:
  family: mlp
pipeline: [match]
selection:
  reference_chain: 0
  reference_sample: 0
match:
  barycenter_passes: 2
  solvers:
    - solver: lap
      max_sweeps: 10
      tolerance: 0.0
runtime:
  resume: false
  parallelism: {parallelism}
  force_cpu: true
        """
    )

    align_main([str(config_path)])

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["stage_config"]["match"]["barycenter_passes"] == 2
    diagnostic = summary["diagnostics"]["reference_stability"]
    assert diagnostic["previous_pass"] == 1
    assert diagnostic["current_pass"] == 2
    assert diagnostic["convergence_ratio"] is not None
    assert np.isfinite(diagnostic["convergence_ratio"])
    assert len(diagnostic["history"]) == 1

    for sample_id in range(3):
        assert (
            output_dir / "aligned_samples" / "0" / f"sample_{sample_id}.npz"
        ).exists()
        assert (output_dir / "transforms" / "0" / f"sample_{sample_id}.npz").exists()

    # On the final pass the configured reference sample is a normal target:
    # the refined barycenter, not that sample, is the actual reference.
    sample_diagnostics = json.loads(
        (output_dir / "sample_diagnostics.json").read_text()
    )
    assert "objective_final" in sample_diagnostics[0]["match"]
    assert sample_diagnostics[0]["match"]["steps"]
    assert not (output_dir / "state" / "refinement").exists()


def test_cli_describe_symmetry_outputs_component_listing(tmp_path, capsys) -> None:

    exp_root = _make_mlp_experiment(tmp_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  experiment_root: {exp_root}
architecture:
  family: mlp
pipeline: [match]
selection:
  reference_chain: 0
  reference_sample: 0
runtime:
  resume: false
  force_cpu: true
        """
    )

    output_dir = exp_root / "align_problems"
    align_main(
        [str(config_path), "--output-dir", str(output_dir), "--describe-symmetry"]
    )

    out = capsys.readouterr().out
    assert "architecture: mlp" in out
    assert "mlp [dense_chain]" in out
    assert "mlp/h0(5)" in out
    assert not (output_dir / "aligned_samples").exists()


def test_cli_validate_only_exits_without_running(tmp_path, capsys) -> None:
    exp_root = _make_mlp_experiment(tmp_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  experiment_root: {exp_root}
pipeline: [canonicalize]
canonicalize:
  activation: relu
selection:
  reference_chain: 0
  reference_sample: 0
runtime:
  resume: false
        """
    )

    output_dir = exp_root / "align_validate"
    align_main([str(config_path), "--output-dir", str(output_dir), "--validate-only"])

    out = capsys.readouterr().out
    assert "Validation successful. No alignment executed." in out
    assert not output_dir.exists()
    assert not (output_dir / "aligned_samples").exists()


def _minimal_params_for_family(family: str):
    import jax
    import jax.numpy as jnp

    from align.benchmarks.synthetic import (
        make_layernorm_mha_transformer_params,
        make_rmsnorm_gqa_rope_transformer_params,
    )

    if family == "mlp":
        return {
            "params": {
                "fcn": {
                    "Dense_0": {
                        "kernel": jnp.ones((2, 3), dtype=jnp.float32),
                        "bias": jnp.zeros((3,), dtype=jnp.float32),
                    },
                    "Dense_1": {
                        "kernel": jnp.ones((3, 1), dtype=jnp.float32),
                        "bias": jnp.zeros((1,), dtype=jnp.float32),
                    },
                }
            }
        }
    if family in {"convnet", "residual_convnet"}:
        return {
            "core": {
                "Conv_0": {
                    "kernel": jnp.ones((1, 1, 2, 3), dtype=jnp.float32),
                    "bias": jnp.zeros((3,), dtype=jnp.float32),
                },
                "Dense_0": {
                    "kernel": jnp.ones((3, 2), dtype=jnp.float32),
                    "bias": jnp.zeros((2,), dtype=jnp.float32),
                },
            }
        }
    if family == "layernorm_mha_transformer":
        return make_layernorm_mha_transformer_params(
            key=jax.random.PRNGKey(1), num_blocks=1
        )
    if family == "rmsnorm_gqa_rope_transformer":
        return make_rmsnorm_gqa_rope_transformer_params(
            key=jax.random.PRNGKey(2), num_blocks=1
        )
    raise AssertionError(family)


@pytest.mark.parametrize(
    ("family", "parameter_root"),
    [
        ("mlp", "params.fcn"),
        ("convnet", "core"),
        ("residual_convnet", "core"),
        ("layernorm_mha_transformer", ""),
        ("rmsnorm_gqa_rope_transformer", ""),
    ],
)
def test_cli_end_to_end_covers_every_architecture_family(
    tmp_path, monkeypatch, family, parameter_root
):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    import jax

    params = _minimal_params_for_family(family)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    exp_root = tmp_path / family
    sample_path = exp_root / "samples" / "0" / "sample_0.npz"
    sample_path.parent.mkdir(parents=True)
    np.savez(
        sample_path,
        **{f"arr_{index}": np.asarray(leaf) for index, leaf in enumerate(leaves)},
    )
    with (exp_root / "tree").open("wb") as handle:
        pickle.dump(treedef, handle)

    config_path = tmp_path / f"{family}.yaml"
    config_path.write_text(
        f"""
paths:
  experiment_root: {exp_root}
architecture:
  family: {family}
  parameter_root: {json.dumps(parameter_root)}
  {"linear_residual_free: true" if family == "residual_convnet" else ""}
pipeline: [match]
selection:
  reference_chain: 0
  reference_sample: 0
match:
  barycenter_passes: 1
  solvers:
    - solver: lap
      max_sweeps: 1
runtime:
  parallelism: 1
  force_cpu: true
"""
    )
    output_dir = exp_root / "align_run"

    align_main([str(config_path), "--output-dir", str(output_dir)])

    assert (output_dir / "aligned_samples" / "0" / "sample_0.npz").exists()
    assert (output_dir / "summary.json").exists()
    run_state = json.loads((output_dir / "state" / "run_state.json").read_text())
    assert run_state["metadata"]["architecture"] == family
