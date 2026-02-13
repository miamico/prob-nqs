from __future__ import annotations

from pcnqs.config.schemas import (
    ClampedSamplingConfig,
    DbmModelConfig,
    DbmTrainingConfig,
    FrbmModelConfig,
    FrbmTrainingConfig,
    LatticeConfig,
    LearningRateConfig,
    SamplingConfig,
    SrConfig,
    TfimConfig,
)


def frbm_small_config(seed: int = 7) -> FrbmTrainingConfig:
    """Small CI/laptop FRBM configuration."""

    return FrbmTrainingConfig(
        lattice=LatticeConfig(L=4),
        tfim=TfimConfig(J=1.0, gamma_x=3.044),
        model=FrbmModelConfig(
            hidden_multiplier=1.0,
            connectivity_radius=2,
            init_std=0.02,
            beta=1.0,
        ),
        sampling=SamplingConfig(n_samples=64, burn_in=20, thin=1),
        sr=SrConfig(
            diagonal_shift_init=0.1,
            diagonal_shift_decay=0.9,
            cg_tolerance=1.0e-4,
            cg_max_iterations=200,
            damping=0.0,
        ),
        learning_rate=LearningRateConfig(eta_max=0.05, eta_min=1.0e-3),
        n_iterations=12,
        eval_samples=400,
        blocking_bins=10,
        seed=seed,
    )


def frbm_paper_config(seed: int = 11) -> FrbmTrainingConfig:
    """Paper-style FRBM configuration (guide Sections 12/16)."""

    return FrbmTrainingConfig(
        lattice=LatticeConfig(L=35),
        tfim=TfimConfig(J=1.0, gamma_x=3.044),
        model=FrbmModelConfig(
            hidden_multiplier=1.0,
            connectivity_radius=2,
            init_std=0.01,
            beta=1.0,
        ),
        sampling=SamplingConfig(n_samples=10_000, burn_in=1_000, thin=1),
        sr=SrConfig(
            diagonal_shift_init=0.1,
            diagonal_shift_decay=0.9,
            cg_tolerance=1.0e-4,
            cg_max_iterations=500,
            damping=0.0,
        ),
        learning_rate=LearningRateConfig(eta_max=0.1, eta_min=1.0e-5),
        n_iterations=1_000,
        eval_samples=1_000_000,
        blocking_bins=50,
        seed=seed,
    )


def dbm_small_config(seed: int = 13) -> DbmTrainingConfig:
    """Small CI/laptop DBM dual-sampling configuration."""

    return DbmTrainingConfig(
        lattice=LatticeConfig(L=4),
        tfim=TfimConfig(J=1.0, gamma_x=3.044),
        model=DbmModelConfig(
            hidden_multiplier=1.0,
            deep_multiplier=1.0,
            connectivity_radius_vh=2,
            connectivity_radius_hd=2,
            init_std=0.02,
            beta=1.0,
        ),
        sampling=SamplingConfig(n_samples=32, burn_in=20, thin=1),
        clamped_sampling=ClampedSamplingConfig(n_steps=40),
        sr=SrConfig(
            diagonal_shift_init=0.1,
            diagonal_shift_decay=0.9,
            cg_tolerance=1.0e-4,
            cg_max_iterations=150,
            damping=0.0,
        ),
        learning_rate=LearningRateConfig(eta_max=0.04, eta_min=1.0e-3),
        n_iterations=8,
        eval_samples=200,
        blocking_bins=10,
        seed=seed,
    )


def dbm_paper_config(seed: int = 19) -> DbmTrainingConfig:
    """Table S1-style DBM configuration from the supplemental PDF."""

    return DbmTrainingConfig(
        lattice=LatticeConfig(L=10),
        tfim=TfimConfig(J=1.0, gamma_x=3.044),
        model=DbmModelConfig(
            hidden_multiplier=1.0,
            deep_multiplier=1.0,
            connectivity_radius_vh=2,
            connectivity_radius_hd=2,
            init_std=0.01,
            beta=1.0,
        ),
        sampling=SamplingConfig(n_samples=10_000, burn_in=1_000, thin=1),
        clamped_sampling=ClampedSamplingConfig(n_steps=1_000),
        sr=SrConfig(
            diagonal_shift_init=0.1,
            diagonal_shift_decay=0.9,
            cg_tolerance=1.0e-4,
            cg_max_iterations=500,
            damping=0.0,
        ),
        learning_rate=LearningRateConfig(eta_max=0.1, eta_min=1.0e-5),
        n_iterations=1_000,
        eval_samples=1_000_000,
        blocking_bins=50,
        seed=seed,
    )
