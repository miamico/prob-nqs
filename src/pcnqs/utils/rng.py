from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import numpy as np
from jax import Array


@dataclass
class RngStreams:
    """Deterministic random streams for numpy and JAX/THRML callers."""

    seed: int

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self._np_rng = np.random.default_rng(self.seed)
        self._jax_key = jax.random.PRNGKey(self.seed)

    @property
    def numpy(self) -> np.random.Generator:
        return self._np_rng

    def split_jax(self) -> Array:
        self._jax_key, subkey = jax.random.split(self._jax_key)
        return cast(Array, subkey)

    def next_int(self, low: int = 0, high: int = 2**31 - 1) -> int:
        return int(self._np_rng.integers(low=low, high=high))
