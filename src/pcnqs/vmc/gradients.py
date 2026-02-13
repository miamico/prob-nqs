from __future__ import annotations

import numpy as np

from pcnqs.optim.sr import SrStatistics, compute_sr_statistics
from pcnqs.types import FloatArray


def stack_operator_samples(operator_rows: list[FloatArray]) -> FloatArray:
    """Stack flat ``O_theta`` rows into SR design matrix."""

    if len(operator_rows) == 0:
        raise ValueError("operator_rows cannot be empty")
    return np.stack(operator_rows, axis=0).astype(np.float64)


def compute_sr_rhs(operators: FloatArray, local_energies: FloatArray) -> SrStatistics:
    """Compute SR force/covariance statistics for VMC updates."""

    return compute_sr_statistics(operators=operators, local_energies=local_energies)
