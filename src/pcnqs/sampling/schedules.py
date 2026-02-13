from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class McmcSchedule:
    """Generic MCMC schedule parameters."""

    burn_in: int
    n_samples: int
    thin: int

    def __post_init__(self) -> None:
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.thin < 1:
            raise ValueError("thin must be >= 1")
