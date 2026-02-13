from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LatticeConfig(BaseModel):
    """Square lattice configuration."""

    model_config = ConfigDict(extra="forbid")

    L: int = Field(ge=2)

    @property
    def n_spins(self) -> int:
        return self.L * self.L


class TfimConfig(BaseModel):
    """TFIM couplings in the sigma^z basis (paper guide Section 1/2)."""

    model_config = ConfigDict(extra="forbid")

    J: float = 1.0
    gamma_x: float = Field(gt=0.0)


class SamplingConfig(BaseModel):
    """Outer-loop MCMC schedule for visible-state sampling."""

    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(ge=1)
    burn_in: int = Field(ge=0)
    thin: int = Field(ge=1)


class ClampedSamplingConfig(BaseModel):
    """Inner-loop conditional sampling schedule for DBM dual sampling."""

    model_config = ConfigDict(extra="forbid")

    n_steps: int = Field(ge=1)


class SrConfig(BaseModel):
    """Stochastic Reconfiguration + CG hyperparameters (guide Section 7)."""

    model_config = ConfigDict(extra="forbid")

    diagonal_shift_init: float = Field(gt=0.0)
    diagonal_shift_decay: float = Field(gt=0.0, le=1.0)
    cg_tolerance: float = Field(gt=0.0)
    cg_max_iterations: int = Field(ge=1)
    damping: float = Field(ge=0.0)


class LearningRateConfig(BaseModel):
    """Cosine-decayed learning-rate schedule."""

    model_config = ConfigDict(extra="forbid")

    eta_max: float = Field(gt=0.0)
    eta_min: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_eta_bounds(self) -> LearningRateConfig:
        if self.eta_min > self.eta_max:
            raise ValueError("eta_min must be <= eta_max")
        return self


class FrbmModelConfig(BaseModel):
    """Sparse FRBM model structure (guide Section 3 and Section 6)."""

    model_config = ConfigDict(extra="forbid")

    hidden_multiplier: float = Field(default=1.0, gt=0.0)
    connectivity_radius: int = Field(default=2, ge=1)
    init_std: float = Field(default=0.01, gt=0.0)
    beta: float = Field(default=1.0, gt=0.0)


class DbmModelConfig(BaseModel):
    """Sparse DBM model structure (guide Section 3 and Section 8)."""

    model_config = ConfigDict(extra="forbid")

    hidden_multiplier: float = Field(default=1.0, gt=0.0)
    deep_multiplier: float = Field(default=1.0, gt=0.0)
    connectivity_radius_vh: int = Field(default=2, ge=1)
    connectivity_radius_hd: int = Field(default=2, ge=1)
    init_std: float = Field(default=0.01, gt=0.0)
    beta: float = Field(default=1.0, gt=0.0)


class FrbmTrainingConfig(BaseModel):
    """Complete FRBM experiment/training specification."""

    model_config = ConfigDict(extra="forbid")

    lattice: LatticeConfig
    tfim: TfimConfig
    model: FrbmModelConfig
    sampling: SamplingConfig
    sr: SrConfig
    learning_rate: LearningRateConfig
    n_iterations: int = Field(ge=1)
    eval_samples: int = Field(default=10_000, ge=1)
    blocking_bins: int = Field(default=50, ge=1)
    seed: int = Field(default=0, ge=0)


class DbmTrainingConfig(BaseModel):
    """Complete DBM dual-sampling experiment/training specification."""

    model_config = ConfigDict(extra="forbid")

    lattice: LatticeConfig
    tfim: TfimConfig
    model: DbmModelConfig
    sampling: SamplingConfig
    clamped_sampling: ClampedSamplingConfig
    sr: SrConfig
    learning_rate: LearningRateConfig
    n_iterations: int = Field(ge=1)
    eval_samples: int = Field(default=10_000, ge=1)
    blocking_bins: int = Field(default=50, ge=1)
    seed: int = Field(default=0, ge=0)
