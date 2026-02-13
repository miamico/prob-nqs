from __future__ import annotations

import numpy as np

from pcnqs.types import FloatArray, IntArray, SpinArray, SpinBatch


def magnetization(spins: SpinArray) -> float:
    """Mean magnetization per spin for one configuration."""

    if spins.ndim != 1:
        raise ValueError("spins must be rank-1")
    return float(np.mean(spins, dtype=np.float64))


def magnetization_batch(spins: SpinBatch) -> FloatArray:
    """Batch magnetization per sample."""

    if spins.ndim != 2:
        raise ValueError("spins must be rank-2")
    return np.asarray(np.mean(spins, axis=1, dtype=np.float64), dtype=np.float64)


def nearest_neighbor_correlator(spins: SpinArray, bonds: IntArray) -> float:
    """Average nearest-neighbor correlator ``<s_i s_j>`` over bond list."""

    if spins.ndim != 1:
        raise ValueError("spins must be rank-1")
    pair_products = spins[bonds[:, 0]] * spins[bonds[:, 1]]
    return float(np.mean(pair_products, dtype=np.float64))
