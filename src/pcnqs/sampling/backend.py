from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pcnqs.types import FloatArray, SpinBatch


@dataclass(frozen=True)
class FrbmSamplingModel:
    """Ising specification for sparse FRBM joint sampling over (v, h)."""

    visible_biases: FloatArray
    hidden_biases: FloatArray
    weights_vh: FloatArray
    beta: float = 1.0


@dataclass(frozen=True)
class DbmSamplingModel:
    """Ising specification for sparse DBM joint sampling over (v, h, d)."""

    visible_biases: FloatArray
    hidden_biases: FloatArray
    deep_biases: FloatArray
    weights_vh: FloatArray
    weights_hd: FloatArray
    beta: float = 1.0


@dataclass(frozen=True)
class JointSamples:
    """Free-sampling trajectories for visible/auxiliary layers."""

    visible: SpinBatch
    hidden: SpinBatch
    deep: SpinBatch | None = None


@dataclass(frozen=True)
class ClampedSamples:
    """Conditional samples for hidden/deep layers given fixed visibles."""

    hidden: SpinBatch
    deep: SpinBatch | None = None


class SamplingBackend(Protocol):
    """Sampler backend API used by VMC loops.

    Sampling must be reproducible given explicit seeds.
    """

    def sample_free(
        self,
        model: FrbmSamplingModel | DbmSamplingModel,
        n_samples: int,
        burn_in: int,
        thin: int,
        seed: int,
    ) -> JointSamples:
        """Sample from the model joint distribution and return visible states."""

    def sample_clamped(
        self,
        model: DbmSamplingModel,
        clamp_v: SpinBatch,
        n_steps: int,
        seed: int,
    ) -> list[ClampedSamples]:
        """Run clamped conditional sampling for each visible configuration."""
