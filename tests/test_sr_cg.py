from __future__ import annotations

import numpy as np

from pcnqs.optim.cg import solve_cg
from pcnqs.optim.sr import build_sr_matvec, explicit_sr_matrix


def test_cg_solves_spd_system() -> None:
    rng = np.random.default_rng(9)
    m = rng.normal(size=(8, 8))
    A = (m.T @ m) + 0.2 * np.eye(8)
    b = rng.normal(size=8)

    result = solve_cg(
        matvec=lambda x: A @ x,
        rhs=b,
        rtol=1.0e-10,
        max_iterations=200,
    )

    x_exact = np.linalg.solve(A, b)
    np.testing.assert_allclose(result.solution, x_exact, rtol=1.0e-8, atol=1.0e-8)
    assert result.converged


def test_matrix_free_sr_matches_explicit_covariance_action() -> None:
    rng = np.random.default_rng(21)
    operators = rng.normal(size=(128, 11)).astype(np.float64)
    centered = operators - np.mean(operators, axis=0, keepdims=True)
    vector = rng.normal(size=11).astype(np.float64)

    diagonal_shift = 0.07
    explicit = explicit_sr_matrix(centered)
    expected = explicit @ vector + diagonal_shift * vector

    matvec = build_sr_matvec(centered, diagonal_shift=diagonal_shift, damping=0.0)
    actual = matvec(vector)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10)
