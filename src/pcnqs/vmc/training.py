from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcnqs.config.schemas import DbmTrainingConfig, FrbmTrainingConfig
from pcnqs.nqs.dbm import (
    DbmParams,
    build_local_mask_between_layers,
    conditional_log_derivatives,
    dual_sampling_ratios,
    init_dbm_params,
)
from pcnqs.nqs.dbm import (
    flatten_operators as flatten_dbm_operators,
)
from pcnqs.nqs.dbm import (
    parameter_layout as dbm_parameter_layout,
)
from pcnqs.nqs.frbm import (
    FrbmParams,
    build_local_mask,
    init_frbm_params,
    log_derivatives,
    single_flip_ratios,
)
from pcnqs.nqs.frbm import (
    flatten_operators as flatten_frbm_operators,
)
from pcnqs.nqs.frbm import (
    parameter_layout as frbm_parameter_layout,
)
from pcnqs.nqs.parameterization import flatten_with_layout, unflatten_with_layout
from pcnqs.optim.cg import CgResult, solve_cg
from pcnqs.optim.lr_schedules import cosine_decay, geometric_decay
from pcnqs.optim.sr import build_sr_matvec
from pcnqs.physics.tfim import SquareLattice, build_nearest_neighbor_bonds
from pcnqs.sampling.backend import DbmSamplingModel, FrbmSamplingModel, SamplingBackend
from pcnqs.sampling.thrml_backend import ThrmlSamplingBackend
from pcnqs.types import FloatArray
from pcnqs.utils.rng import RngStreams
from pcnqs.vmc.estimators import MeanWithError, blocking_error_bars, local_energy_single
from pcnqs.vmc.gradients import compute_sr_rhs, stack_operator_samples


@dataclass(frozen=True)
class IterationMetrics:
    """Per-iteration diagnostics for SR-CG VMC optimization."""

    iteration: int
    energy_mean: float
    energy_stderr: float
    learning_rate: float
    diagonal_shift: float
    cg_iterations: int
    cg_residual_norm: float


@dataclass(frozen=True)
class FrbmTrainingResult:
    """FRBM training output bundle."""

    params: FrbmParams
    history: list[IterationMetrics]
    final_eval: MeanWithError


@dataclass(frozen=True)
class DbmTrainingResult:
    """DBM training output bundle."""

    params: DbmParams
    history: list[IterationMetrics]
    final_eval: MeanWithError


def _frbm_sampling_model(params: FrbmParams, beta: float) -> FrbmSamplingModel:
    return FrbmSamplingModel(
        visible_biases=params.a,
        hidden_biases=params.b,
        weights_vh=params.w,
        beta=beta,
    )


def _dbm_sampling_model(params: DbmParams, beta: float) -> DbmSamplingModel:
    return DbmSamplingModel(
        visible_biases=params.a,
        hidden_biases=params.b,
        deep_biases=params.c,
        weights_vh=params.w_vh,
        weights_hd=params.w_hd,
        beta=beta,
    )


def _apply_frbm_vector_update(
    params: FrbmParams,
    update_vector: FloatArray,
    step_size: float,
) -> FrbmParams:
    layout = frbm_parameter_layout(params)
    flat = flatten_with_layout(params.named_arrays(), layout)
    next_flat = flat - step_size * update_vector
    unpacked = unflatten_with_layout(next_flat, layout)
    w = unpacked["w"] * params.mask
    return FrbmParams(a=unpacked["a"], b=unpacked["b"], w=w, mask=params.mask)


def _apply_dbm_vector_update(
    params: DbmParams,
    update_vector: FloatArray,
    step_size: float,
) -> DbmParams:
    layout = dbm_parameter_layout(params)
    flat = flatten_with_layout(params.named_arrays(), layout)
    next_flat = flat - step_size * update_vector
    unpacked = unflatten_with_layout(next_flat, layout)

    w_vh = unpacked["w_vh"] * params.mask_vh
    w_hd = unpacked["w_hd"] * params.mask_hd

    return DbmParams(
        a=unpacked["a"],
        b=unpacked["b"],
        c=unpacked["c"],
        w_vh=w_vh,
        w_hd=w_hd,
        mask_vh=params.mask_vh,
        mask_hd=params.mask_hd,
    )


def _energy_error(values: FloatArray, requested_bins: int) -> MeanWithError:
    n_bins = min(requested_bins, values.shape[0])
    return blocking_error_bars(values=values, n_bins=max(1, n_bins))


def _solve_sr_cg(
    operators: FloatArray,
    local_energies: FloatArray,
    diagonal_shift: float,
    tolerance: float,
    max_iterations: int,
    damping: float,
) -> tuple[FloatArray, CgResult]:
    sr_stats = compute_sr_rhs(operators=operators, local_energies=local_energies)
    matvec = build_sr_matvec(
        centered_operators=sr_stats.centered_operators,
        diagonal_shift=diagonal_shift,
        damping=damping,
    )
    cg = solve_cg(
        matvec=matvec,
        rhs=sr_stats.force,
        rtol=tolerance,
        max_iterations=max_iterations,
    )
    return cg.solution, cg


def train_frbm(
    config: FrbmTrainingConfig,
    backend: SamplingBackend | None = None,
) -> FrbmTrainingResult:
    """Train sparse FRBM using TFIM VMC + SR-CG (guide Sections 6 and 7)."""

    sampler = backend if backend is not None else ThrmlSamplingBackend()
    lattice = SquareLattice(config.lattice.L)
    bonds = build_nearest_neighbor_bonds(lattice.L)

    n_visible = lattice.n_spins
    n_hidden = max(1, int(round(config.model.hidden_multiplier * n_visible)))

    rngs = RngStreams(seed=config.seed)
    mask = build_local_mask(
        lattice=lattice,
        n_hidden=n_hidden,
        radius=config.model.connectivity_radius,
    )
    params = init_frbm_params(
        rng=rngs.numpy,
        n_visible=n_visible,
        n_hidden=n_hidden,
        mask=mask,
        init_std=config.model.init_std,
    )

    layout = frbm_parameter_layout(params)
    history: list[IterationMetrics] = []

    for step in range(config.n_iterations):
        model = _frbm_sampling_model(params, beta=config.model.beta)
        sampled = sampler.sample_free(
            model=model,
            n_samples=config.sampling.n_samples,
            burn_in=config.sampling.burn_in,
            thin=config.sampling.thin,
            seed=rngs.next_int(),
        )

        spins = sampled.visible
        n_samples = spins.shape[0]

        ratios = np.zeros_like(spins, dtype=np.float64)
        local_energies = np.zeros(n_samples, dtype=np.float64)
        operator_rows: list[FloatArray] = []

        for i in range(n_samples):
            v = spins[i]
            ratio_i = single_flip_ratios(v, params)
            ratios[i] = ratio_i
            local_energies[i] = local_energy_single(
                spins=v,
                bonds=bonds,
                J=config.tfim.J,
                gamma_x=config.tfim.gamma_x,
                ratios=ratio_i,
            )
            op = flatten_frbm_operators(log_derivatives(v, params), layout)
            operator_rows.append(op)

        operators = stack_operator_samples(operator_rows)

        diagonal_shift = geometric_decay(
            step=step,
            initial_value=config.sr.diagonal_shift_init,
            factor=config.sr.diagonal_shift_decay,
        )
        delta, cg = _solve_sr_cg(
            operators=operators,
            local_energies=local_energies,
            diagonal_shift=diagonal_shift,
            tolerance=config.sr.cg_tolerance,
            max_iterations=config.sr.cg_max_iterations,
            damping=config.sr.damping,
        )

        lr = cosine_decay(
            step=step,
            total_steps=config.n_iterations,
            eta_max=config.learning_rate.eta_max,
            eta_min=config.learning_rate.eta_min,
        )
        params = _apply_frbm_vector_update(params=params, update_vector=delta, step_size=lr)

        energy_stats = _energy_error(local_energies, requested_bins=config.blocking_bins)
        history.append(
            IterationMetrics(
                iteration=step,
                energy_mean=energy_stats.mean,
                energy_stderr=energy_stats.stderr,
                learning_rate=lr,
                diagonal_shift=diagonal_shift,
                cg_iterations=cg.iterations,
                cg_residual_norm=cg.residual_norm,
            )
        )

    final_eval = evaluate_frbm_energy(
        config=config,
        params=params,
        backend=sampler,
        seed=rngs.next_int(),
    )
    return FrbmTrainingResult(params=params, history=history, final_eval=final_eval)


def evaluate_frbm_energy(
    config: FrbmTrainingConfig,
    params: FrbmParams,
    backend: SamplingBackend,
    seed: int,
) -> MeanWithError:
    """Final FRBM energy evaluation with blocking error bars."""

    lattice = SquareLattice(config.lattice.L)
    bonds = build_nearest_neighbor_bonds(lattice.L)
    sampled = backend.sample_free(
        model=_frbm_sampling_model(params, beta=config.model.beta),
        n_samples=config.eval_samples,
        burn_in=config.sampling.burn_in,
        thin=config.sampling.thin,
        seed=seed,
    )

    local = np.zeros(sampled.visible.shape[0], dtype=np.float64)
    for i, v in enumerate(sampled.visible):
        ratios = single_flip_ratios(v, params)
        local[i] = local_energy_single(
            spins=v,
            bonds=bonds,
            J=config.tfim.J,
            gamma_x=config.tfim.gamma_x,
            ratios=ratios,
        )

    return _energy_error(local, requested_bins=config.blocking_bins)


def train_dbm(
    config: DbmTrainingConfig,
    backend: SamplingBackend | None = None,
) -> DbmTrainingResult:
    """Train sparse DBM with dual sampling + finite-``Nc`` correction (Algorithm S1)."""

    sampler = backend if backend is not None else ThrmlSamplingBackend()
    lattice = SquareLattice(config.lattice.L)
    bonds = build_nearest_neighbor_bonds(lattice.L)

    n_visible = lattice.n_spins
    n_hidden = max(1, int(round(config.model.hidden_multiplier * n_visible)))
    n_deep = max(1, int(round(config.model.deep_multiplier * n_visible)))

    rngs = RngStreams(seed=config.seed)
    mask_vh = build_local_mask_between_layers(
        lattice=lattice,
        n_source=n_visible,
        n_target=n_hidden,
        radius=config.model.connectivity_radius_vh,
    )
    mask_hd = build_local_mask_between_layers(
        lattice=lattice,
        n_source=n_hidden,
        n_target=n_deep,
        radius=config.model.connectivity_radius_hd,
    )
    params = init_dbm_params(
        rng=rngs.numpy,
        n_visible=n_visible,
        n_hidden=n_hidden,
        n_deep=n_deep,
        mask_vh=mask_vh,
        mask_hd=mask_hd,
        init_std=config.model.init_std,
    )

    layout = dbm_parameter_layout(params)
    history: list[IterationMetrics] = []

    for step in range(config.n_iterations):
        model = _dbm_sampling_model(params, beta=config.model.beta)
        outer_samples = sampler.sample_free(
            model=model,
            n_samples=config.sampling.n_samples,
            burn_in=config.sampling.burn_in,
            thin=config.sampling.thin,
            seed=rngs.next_int(),
        )
        visibles = outer_samples.visible

        clamped_samples = sampler.sample_clamped(
            model=model,
            clamp_v=visibles,
            n_steps=config.clamped_sampling.n_steps,
            seed=rngs.next_int(),
        )

        local_energies = np.zeros(visibles.shape[0], dtype=np.float64)
        operator_rows: list[FloatArray] = []

        for idx, v in enumerate(visibles):
            clamped = clamped_samples[idx]
            if clamped.deep is None:
                raise RuntimeError("DBM clamped sampling requires deep-layer samples")

            ratio_stats = dual_sampling_ratios(v=v, hidden_samples=clamped.hidden, params=params)
            local_energies[idx] = local_energy_single(
                spins=v,
                bonds=bonds,
                J=config.tfim.J,
                gamma_x=config.tfim.gamma_x,
                ratios=ratio_stats.corrected_amplitude_ratio,
            )

            ops = conditional_log_derivatives(
                v=v,
                hidden_samples=clamped.hidden,
                deep_samples=clamped.deep,
                params=params,
            )
            operator_rows.append(flatten_dbm_operators(ops, layout))

        operators = stack_operator_samples(operator_rows)

        diagonal_shift = geometric_decay(
            step=step,
            initial_value=config.sr.diagonal_shift_init,
            factor=config.sr.diagonal_shift_decay,
        )
        delta, cg = _solve_sr_cg(
            operators=operators,
            local_energies=local_energies,
            diagonal_shift=diagonal_shift,
            tolerance=config.sr.cg_tolerance,
            max_iterations=config.sr.cg_max_iterations,
            damping=config.sr.damping,
        )

        lr = cosine_decay(
            step=step,
            total_steps=config.n_iterations,
            eta_max=config.learning_rate.eta_max,
            eta_min=config.learning_rate.eta_min,
        )
        params = _apply_dbm_vector_update(params=params, update_vector=delta, step_size=lr)

        energy_stats = _energy_error(local_energies, requested_bins=config.blocking_bins)
        history.append(
            IterationMetrics(
                iteration=step,
                energy_mean=energy_stats.mean,
                energy_stderr=energy_stats.stderr,
                learning_rate=lr,
                diagonal_shift=diagonal_shift,
                cg_iterations=cg.iterations,
                cg_residual_norm=cg.residual_norm,
            )
        )

    final_eval = evaluate_dbm_energy(
        config=config,
        params=params,
        backend=sampler,
        seed=rngs.next_int(),
    )
    return DbmTrainingResult(params=params, history=history, final_eval=final_eval)


def evaluate_dbm_energy(
    config: DbmTrainingConfig,
    params: DbmParams,
    backend: SamplingBackend,
    seed: int,
) -> MeanWithError:
    """Final DBM energy evaluation with dual-sampling local energies."""

    lattice = SquareLattice(config.lattice.L)
    bonds = build_nearest_neighbor_bonds(lattice.L)

    model = _dbm_sampling_model(params, beta=config.model.beta)
    outer_samples = backend.sample_free(
        model=model,
        n_samples=config.eval_samples,
        burn_in=config.sampling.burn_in,
        thin=config.sampling.thin,
        seed=seed,
    )
    clamped = backend.sample_clamped(
        model=model,
        clamp_v=outer_samples.visible,
        n_steps=config.clamped_sampling.n_steps,
        seed=seed + 1,
    )

    local = np.zeros(outer_samples.visible.shape[0], dtype=np.float64)
    for idx, v in enumerate(outer_samples.visible):
        ratio_stats = dual_sampling_ratios(v=v, hidden_samples=clamped[idx].hidden, params=params)
        local[idx] = local_energy_single(
            spins=v,
            bonds=bonds,
            J=config.tfim.J,
            gamma_x=config.tfim.gamma_x,
            ratios=ratio_stats.corrected_amplitude_ratio,
        )

    return _energy_error(local, requested_bins=config.blocking_bins)
