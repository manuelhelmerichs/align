"""Tests for posterior-level benchmarks: R-hat, synthetic chains, real loaders."""

import pickle
from pathlib import Path

import jax
import numpy as np
import pytest

from align.matching import TransformState
from benchmarks.posterior import (
    evaluate_function_drift,
    evaluate_posterior_metrics,
    load_experiment_posterior_case,
    make_synthetic_layernorm_mha_transformer_posterior_case,
    make_synthetic_mlp_posterior_case,
    rank_normalized_split_rhat,
    run_posterior_benchmark,
    split_rhat,
)
from benchmarks.synthetic import make_mlp_orbit_case, mlp_apply, permutation_matrix


def test_split_rhat_near_one_for_stationary_chains():
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((4, 40, 3))

    rhat = split_rhat(draws)

    assert rhat.shape == (3,)
    np.testing.assert_allclose(rhat, 1.0, atol=0.1)


def test_split_rhat_detects_separated_chains_and_ignores_constants():
    rng = np.random.default_rng(1)
    draws = rng.standard_normal((4, 40, 2))
    draws[2:] += 50.0

    assert np.all(split_rhat(draws) > 5.0)
    np.testing.assert_array_equal(split_rhat(np.zeros((2, 4, 2))), 1.0)


def test_split_rhat_validates_input_shape():
    with pytest.raises(ValueError):
        split_rhat(np.zeros((4, 8)))
    with pytest.raises(ValueError):
        split_rhat(np.zeros((1, 8, 2)))
    with pytest.raises(ValueError):
        split_rhat(np.zeros((2, 3, 2)))


def test_rank_normalized_split_rhat_detects_location_and_scale_failures():
    rng = np.random.default_rng(7)
    stationary = rng.standard_normal((4, 200, 2))
    location = stationary.copy()
    location[2:, :, 0] += 3.0
    scale = stationary.copy()
    scale[2:, :, 1] *= 4.0

    stationary_rhat = rank_normalized_split_rhat(stationary)
    location_rhat = rank_normalized_split_rhat(location)
    scale_rhat = rank_normalized_split_rhat(scale)

    assert set(stationary_rhat) == {"bulk", "folded", "max"}
    np.testing.assert_allclose(stationary_rhat["max"], 1.0, atol=0.05)
    assert location_rhat["bulk"][0] > 1.5
    assert scale_rhat["folded"][1] > 1.2
    assert scale_rhat["folded"][1] > scale_rhat["bulk"][1]
    np.testing.assert_array_equal(
        scale_rhat["max"],
        np.maximum(scale_rhat["bulk"], scale_rhat["folded"]),
    )


def test_rank_normalized_split_rhat_validates_input_shape():
    with pytest.raises(ValueError, match="Expected"):
        rank_normalized_split_rhat(np.zeros((4, 8)))


def test_synthetic_posterior_case_validates_chain_and_sample_counts():
    with pytest.raises(ValueError):
        make_synthetic_mlp_posterior_case(n_chains=1)
    with pytest.raises(ValueError):
        make_synthetic_mlp_posterior_case(n_samples=3)


def test_synthetic_posterior_alignment_collapses_symmetry_copies():
    case = make_synthetic_mlp_posterior_case(seed=0, n_chains=2, n_samples=4)

    result = run_posterior_benchmark(
        case,
        schedule=[{"solver": "lap", "max_sweeps": 25, "tolerance": 0.0}],
        canonicalize=True,
    )

    before, after = result.metrics_before, result.metrics_after
    assert before.split_rhat_max > 20.0
    assert after.split_rhat_max < 5.0
    assert before.weight_averaging_gap > 0.5
    assert after.weight_averaging_gap < 0.05
    assert after.cross_chain_distance_mean < 0.1 * before.cross_chain_distance_mean
    assert result.function_drift_max < 1e-5
    assert result.samples_per_s > 0.0
    assert len(result.aligned_chains) == len(case.chains)
    assert all(len(chain) == 4 for chain in result.aligned_chains)


def test_synthetic_layernorm_mha_transformer_posterior_needs_circuit_canonicalization():
    """Chains in permutation + qk/vo-scale copies collapse (mean split R-hat
    to ~1) only when balancing canonicalization precedes matching; matching alone
    leaves the circuit-scale offsets in weight space."""

    case = make_synthetic_layernorm_mha_transformer_posterior_case(
        seed=0, n_chains=2, n_samples=6
    )
    schedule = [{"solver": "lap", "max_sweeps": 25, "tolerance": 0.0}]

    with_canonicalization = run_posterior_benchmark(
        case, schedule=schedule, canonicalize=True
    )
    without = run_posterior_benchmark(case, schedule=schedule, canonicalize=False)

    before = with_canonicalization.metrics_before
    assert before.split_rhat_mean > 10.0
    assert with_canonicalization.metrics_after.split_rhat_mean < 1.2
    assert without.metrics_after.split_rhat_mean > 2.0
    assert (
        with_canonicalization.metrics_after.cross_chain_distance_mean
        < 0.2 * without.metrics_after.cross_chain_distance_mean
    )
    assert with_canonicalization.function_drift_max is not None
    assert with_canonicalization.function_drift_max < 1e-4
    assert with_canonicalization.head_assignment_accuracy == 1.0
    assert with_canonicalization.head_assignment_exact_rate == 1.0


def _save_pytree_npz(path: Path, pytree) -> None:
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    payload = {f"arr_{idx}": np.asarray(leaf) for idx, leaf in enumerate(leaves)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _write_experiment(tmp_path: Path) -> Path:
    """Fabricate a SMILE/MILE-style directory: two chains in symmetry copies."""

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


def test_load_experiment_posterior_case_and_align_real_samples(tmp_path):
    root = _write_experiment(tmp_path)

    case = load_experiment_posterior_case(root, architecture="mlp")

    assert case.name == "experiment_run42"
    assert len(case.chains) == 2
    assert all(len(chain) == 4 for chain in case.chains)
    assert case.reference is case.chains[0][0]
    assert case.apply_fn is None
    assert case.metadata["chain_ids"] == [0, 1]
    assert case.metadata["total_samples"] == 8

    trimmed = load_experiment_posterior_case(
        root, architecture="mlp", samples_per_chain=2
    )
    assert all(len(chain) == 2 for chain in trimmed.chains)

    result = run_posterior_benchmark(
        case, schedule=[{"solver": "lap", "max_sweeps": 25, "tolerance": 0.0}]
    )

    before, after = result.metrics_before, result.metrics_after
    assert before.split_rhat_max > 20.0
    assert after.split_rhat_max < 5.0
    assert after.cross_chain_distance_mean < 0.1 * before.cross_chain_distance_mean
    # Real experiments carry no apply_fn, so function-space metrics are absent.
    assert result.function_drift_max is None
    assert before.weight_averaging_gap is None


def test_real_case_reports_invariant_functional_diagnostics_when_wired(tmp_path):
    root = _write_experiment(tmp_path)
    inputs = jax.random.normal(jax.random.key(4), (9, 3))
    case = load_experiment_posterior_case(
        root,
        architecture="mlp",
        apply_fn=mlp_apply,
        inputs=inputs,
        functional_batch_size=4,
    )

    before = evaluate_posterior_metrics(case, case.chains)
    result = run_posterior_benchmark(
        case, schedule=[{"solver": "lap", "max_sweeps": 25, "tolerance": 0.0}]
    )
    after = result.metrics_after

    assert before.weight_averaging_gap is not None
    assert before.function_variance_total is not None
    assert before.function_within_chain_variance is not None
    assert before.function_between_chain_mean_variance is not None
    assert before.function_between_variance_fraction is not None
    assert before.function_within_rms_distance is not None
    assert before.function_cross_rms_distance is not None
    assert before.function_cross_within_distance_ratio is not None
    assert before.chain_mean_prediction_rmse_mean > 0.0
    assert (
        before.chain_mean_prediction_rmse_max >= before.chain_mean_prediction_rmse_mean
    )
    np.testing.assert_allclose(
        after.chain_mean_prediction_rmse_mean,
        before.chain_mean_prediction_rmse_mean,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        after.function_cross_rms_distance,
        before.function_cross_rms_distance,
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.function_drift_max < 1e-5
    assert result.function_drift_rmse < 1e-6
    assert result.function_drift_relative_rmse < 1e-6
    drift = evaluate_function_drift(case, case.chains, result.aligned_chains)
    assert drift is not None
    assert drift["max"] == pytest.approx(result.function_drift_max)


def test_real_case_functional_wiring_is_explicit_and_validated(tmp_path):
    root = _write_experiment(tmp_path)
    inputs = jax.random.normal(jax.random.key(5), (4, 3))

    with pytest.raises(ValueError, match="provided together"):
        load_experiment_posterior_case(root, architecture="mlp", apply_fn=mlp_apply)
    with pytest.raises(ValueError, match="provided together"):
        load_experiment_posterior_case(root, architecture="mlp", inputs=inputs)
    with pytest.raises(ValueError, match="positive"):
        load_experiment_posterior_case(
            root,
            architecture="mlp",
            apply_fn=mlp_apply,
            inputs=inputs,
            functional_batch_size=0,
        )


def test_inverse_fisher_noise_mode_shapes_within_chain_noise():
    """Anisotropic noise concentrates energy in low-Fisher coordinates but
    keeps total energy comparable to the isotropic case."""

    iso = make_synthetic_mlp_posterior_case(
        seed=0, n_chains=2, n_samples=4, noise_mode="isotropic", noise_scale=0.1
    )
    anis = make_synthetic_mlp_posterior_case(
        seed=0, n_chains=2, n_samples=4, noise_mode="inverse_fisher", noise_scale=0.1
    )
    assert anis.metadata["noise_mode"] == "inverse_fisher"
    assert "inverse_fisher" in anis.name

    def _chain_residual_std(case):
        draws = np.stack(
            [
                np.concatenate(
                    [
                        np.ravel(np.asarray(leaf))
                        for leaf in jax.tree_util.tree_leaves(sample)
                    ]
                )
                for sample in case.chains[0]
            ]
        )
        return draws.std(axis=0)

    iso_std = _chain_residual_std(iso)
    anis_std = _chain_residual_std(anis)
    # Comparable total energy, but the anisotropic profile is much more spread.
    assert 0.5 < float(np.sum(anis_std**2) / np.sum(iso_std**2)) < 2.0
    assert anis_std.max() / anis_std.min() > 5.0 * iso_std.max() / iso_std.min()

    with pytest.raises(ValueError, match="noise_mode"):
        make_synthetic_mlp_posterior_case(noise_mode="bogus")


def test_layernorm_mha_transformer_posterior_supports_inverse_fisher_noise():
    case = make_synthetic_layernorm_mha_transformer_posterior_case(
        seed=0, n_chains=2, n_samples=4, noise_mode="inverse_fisher"
    )
    assert case.metadata["noise_mode"] == "inverse_fisher"
    assert "inverse_fisher" in case.name
    assert len(case.chains) == 2


def test_barycenter_refinement_high_noise_quality_and_validity():
    """Two-pass refinement stays valid and competitive at high noise.

    Under the balanced canonicalization default, single-pass gaps on this
    anisotropic case are already strong (0.24; unit-norm needed the second
    pass to reach that), so refinement's gap effect here is roughly neutral —
    its primary value is reference independence (see the dedicated test
    below). Pin: exact function preservation, a collapsed posterior, and no
    substantial gap regression from the second pass.
    """

    case = make_synthetic_mlp_posterior_case(
        seed=0, noise_mode="inverse_fisher", noise_scale=0.4
    )
    kwargs = {"calibration": "fisher"}
    schedule = [{"solver": "lap", "max_sweeps": 25, "tolerance": 0.0}]
    one = run_posterior_benchmark(
        case,
        objective="diagonal_fisher",
        objective_kwargs=kwargs,
        schedule=schedule,
        canonicalize=True,
        barycenter_passes=1,
    )
    two = run_posterior_benchmark(
        case,
        objective="diagonal_fisher",
        objective_kwargs=kwargs,
        schedule=schedule,
        canonicalize=True,
        barycenter_passes=2,
    )
    assert two.function_drift_max < 1e-4
    assert two.metrics_after.weight_averaging_gap < 0.35
    assert two.metrics_after.weight_averaging_gap < (
        1.3 * one.metrics_after.weight_averaging_gap
    )
    assert two.metrics_after.split_rhat_max < 2.0

    with pytest.raises(ValueError, match="barycenter_passes"):
        run_posterior_benchmark(case, barycenter_passes=0)


def test_barycenter_refinement_restores_reference_independence():
    """The canonicalized cloud must not depend on which sample was reference.

    Align the same posterior with two different reference samples, remove the
    single global frame permutation (barycenter-to-barycenter matching), and
    measure the mean per-sample mismatch relative to the cloud spread. At
    high noise, single-pass canonicalization is strongly reference-dependent
    (seed-0 mismatch 0.47 of spread); one barycenter refinement pass cuts it
    by an order of magnitude (0.04).
    """

    import jax

    from align.matching import TransformState, match_sample
    from benchmarks.posterior import tree_mean

    case = make_synthetic_mlp_posterior_case(
        seed=0, noise_mode="isotropic", noise_scale=0.4
    )
    schedule = [{"solver": "lap", "max_sweeps": 25, "tolerance": 0.0}]

    def flat(tree):
        return np.concatenate(
            [
                np.ravel(np.asarray(leaf, dtype=np.float64))
                for leaf in jax.tree_util.tree_leaves(tree)
            ]
        )

    def mismatch(passes):
        clouds = []
        for reference_chain in (0, 1):
            case.reference_chain = reference_chain
            case.reference_sample = 0
            result = run_posterior_benchmark(
                case,
                objective="euclidean",
                schedule=schedule,
                canonicalize=True,
                barycenter_passes=passes,
            )
            clouds.append([s for chain in result.aligned_chains for s in chain])
        base, other = clouds
        base_bary = tree_mean(base)
        spread = float(
            np.mean([np.linalg.norm(flat(s) - flat(base_bary)) for s in base])
        )
        _, frame, _ = match_sample(
            case.graph,
            base_bary,
            tree_mean(other),
            objective="euclidean",
            schedule=schedule,
        )
        state = TransformState.from_transforms(case.graph, frame)
        moved = [case.graph.apply_transforms(s, state) for s in other]
        return (
            float(
                np.mean(
                    [
                        np.linalg.norm(flat(a) - flat(b))
                        for a, b in zip(base, moved, strict=True)
                    ]
                )
            )
            / spread
        )

    one_pass = mismatch(1)
    two_pass = mismatch(2)
    assert one_pass > 0.2  # the reference-dependence problem is real here
    assert two_pass < 0.5 * one_pass
