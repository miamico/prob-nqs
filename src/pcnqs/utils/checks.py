from __future__ import annotations

import numpy as np


def require_shape(name: str, array: np.ndarray, expected: tuple[int, ...]) -> None:
    """Raise a clear error if an array shape differs from expectations."""

    if array.shape != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, received {array.shape}")


def require_finite(name: str, array: np.ndarray) -> None:
    """Guard against NaN/Inf propagation during optimization."""

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf values")


def require_spin_values(name: str, array: np.ndarray) -> None:
    """Ensure spins are represented with {-1, +1}."""

    if not np.all(np.isin(array, (-1, 1))):
        raise ValueError(f"{name} contains values outside {-1, +1}")
