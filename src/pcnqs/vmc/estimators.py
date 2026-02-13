from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcnqs.physics.tfim import local_energy_from_ratios
from pcnqs.types import FloatArray, IntArray, SpinArray, SpinBatch


@dataclass(frozen=True)
class MeanWithError:
    """Mean estimate with standard error from block statistics."""

    mean: float
    stderr: float


def blocking_error_bars(values: FloatArray, n_bins: int) -> MeanWithError:
    """Blocking analysis used in the paper (50 bins in full mode)."""

    if values.ndim != 1:
        raise ValueError("values must be rank-1")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if values.shape[0] < n_bins:
        raise ValueError("need at least n_bins samples for blocking")

    trimmed = values[: (values.shape[0] // n_bins) * n_bins]
    block_size = trimmed.shape[0] // n_bins

    blocks = trimmed.reshape(n_bins, block_size)
    block_means = np.mean(blocks, axis=1, dtype=np.float64)

    mean = float(np.mean(block_means, dtype=np.float64))
    if n_bins == 1:
        return MeanWithError(mean=mean, stderr=0.0)

    stderr = float(np.std(block_means, ddof=1, dtype=np.float64) / np.sqrt(n_bins))
    return MeanWithError(mean=mean, stderr=stderr)


def local_energies_from_ratio_batch(
    spins: SpinBatch,
    bonds: IntArray,
    J: float,
    gamma_x: float,
    ratios: FloatArray,
) -> FloatArray:
    """Compute TFIM local energies from precomputed single-flip ratios."""

    if spins.ndim != 2:
        raise ValueError("spins must be rank-2")
    if ratios.shape != spins.shape:
        raise ValueError("ratios must match spins shape (n_samples, n_visible)")

    out = np.zeros(spins.shape[0], dtype=np.float64)
    for i in range(spins.shape[0]):
        out[i] = local_energy_from_ratios(
            spins=spins[i],
            bonds=bonds,
            J=J,
            gamma_x=gamma_x,
            psi_flip_ratios=ratios[i],
        )
    return out


def local_energy_single(
    spins: SpinArray,
    bonds: IntArray,
    J: float,
    gamma_x: float,
    ratios: FloatArray,
) -> float:
    """Single-sample TFIM local energy helper."""

    return local_energy_from_ratios(
        spins=spins,
        bonds=bonds,
        J=J,
        gamma_x=gamma_x,
        psi_flip_ratios=ratios,
    )
