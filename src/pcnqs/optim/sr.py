from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pcnqs.types import FloatArray


@dataclass(frozen=True)
class SrStatistics:
    """Sample-based SR quantities in flat-parameter basis."""

    mean_operators: FloatArray
    centered_operators: FloatArray
    force: FloatArray
    mean_energy: float


def compute_sr_statistics(operators: FloatArray, local_energies: FloatArray) -> SrStatistics:
    """Compute SR force and centered operators (guide Section 7.1)."""

    if operators.ndim != 2:
        raise ValueError("operators must have shape (n_samples, n_params)")
    if local_energies.ndim != 1:
        raise ValueError("local_energies must be rank-1")
    if operators.shape[0] != local_energies.shape[0]:
        raise ValueError("sample axis mismatch between operators and local_energies")

    mean_o = np.asarray(np.mean(operators, axis=0, dtype=np.float64), dtype=np.float64)
    centered = operators - mean_o[None, :]

    mean_energy = float(np.mean(local_energies, dtype=np.float64))
    cross = np.asarray(
        np.mean(operators * local_energies[:, None], axis=0, dtype=np.float64),
        dtype=np.float64,
    )
    force = np.asarray(cross - mean_o * mean_energy, dtype=np.float64)

    return SrStatistics(
        mean_operators=mean_o,
        centered_operators=centered,
        force=force,
        mean_energy=mean_energy,
    )


def build_sr_matvec(
    centered_operators: FloatArray,
    diagonal_shift: float,
    damping: float = 0.0,
) -> Callable[[FloatArray], FloatArray]:
    """Matrix-free action of ``(S + lambda I)`` for SR-CG."""

    if centered_operators.ndim != 2:
        raise ValueError("centered_operators must have shape (n_samples, n_params)")
    if diagonal_shift <= 0.0:
        raise ValueError("diagonal_shift must be positive")

    n_samples = float(centered_operators.shape[0])

    def matvec(vector: FloatArray) -> FloatArray:
        if vector.ndim != 1:
            raise ValueError("vector must be rank-1")
        projected = centered_operators @ vector
        cov_action = (centered_operators.T @ projected) / n_samples
        reg_action = diagonal_shift * vector

        if damping > 0.0:
            reg_action = reg_action + damping * vector
        return cov_action + reg_action

    return matvec


def explicit_sr_matrix(centered_operators: FloatArray) -> FloatArray:
    """Materialize SR covariance for tests/debugging only."""

    n_samples = float(centered_operators.shape[0])
    return (centered_operators.T @ centered_operators) / n_samples
