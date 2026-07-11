"""Tests for Sinkhorn utilities."""

import jax.numpy as jnp
import numpy as np

from align.matching import sinkhorn_operator, solve_lap_maximize


def _perm_matrix(indices) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    matrix = np.zeros((idx.size, idx.size), dtype=np.float32)
    matrix[np.arange(idx.size), idx] = 1.0
    return matrix


def test_sinkhorn_operator_produces_doubly_stochastic_matrix():
    logits = jnp.array(
        [[0.2, -0.3, 0.1], [1.0, 0.0, -0.5], [0.3, 0.7, -0.2]], dtype=jnp.float32
    )
    matrix = sinkhorn_operator(logits, tau=0.5, n_iters=200)

    row_sums = np.asarray(matrix.sum(axis=1))
    col_sums = np.asarray(matrix.sum(axis=0))

    # At convergence the projection is doubly stochastic and non-negative.
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-4)
    np.testing.assert_allclose(col_sums, np.ones_like(col_sums), atol=1e-4)
    assert np.all(np.asarray(matrix) >= 0.0)


def test_sinkhorn_operator_normalizes_each_matrix_in_a_batch():
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.standard_normal((4, 5, 5)), dtype=jnp.float32)

    matrices = np.asarray(sinkhorn_operator(logits, tau=0.5, n_iters=200))

    assert matrices.shape == (4, 5, 5)
    for matrix in matrices:
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(5), atol=1e-4)
        np.testing.assert_allclose(matrix.sum(axis=0), np.ones(5), atol=1e-4)
        assert np.all(matrix >= 0.0)


def test_sinkhorn_operator_sharpens_toward_target_permutation():
    """Logits peaked on a permutation project (then harden) back to it."""

    target = [2, 0, 3, 1]
    logits = jnp.asarray(8.0 * _perm_matrix(target))

    matrix = sinkhorn_operator(logits, tau=0.1, n_iters=200)

    # The mass concentrates on the target assignment, and hardening recovers it.
    assert np.all(np.asarray(matrix)[np.arange(4), target] > 0.9)
    np.testing.assert_array_equal(solve_lap_maximize(np.asarray(matrix)), target)
