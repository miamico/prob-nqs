from __future__ import annotations

import math


def cosine_decay(step: int, total_steps: int, eta_max: float, eta_min: float) -> float:
    """Cosine schedule used by the paper's Table S1 training setup."""

    if total_steps <= 1:
        return eta_min

    clipped = min(max(step, 0), total_steps - 1)
    frac = clipped / float(total_steps - 1)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * frac))
    return eta_min + (eta_max - eta_min) * cos_term


def geometric_decay(step: int, initial_value: float, factor: float) -> float:
    """Geometric decay schedule for SR diagonal shift."""

    if step < 0:
        raise ValueError("step must be non-negative")
    if factor <= 0.0:
        raise ValueError("factor must be > 0")
    return initial_value * (factor ** step)
