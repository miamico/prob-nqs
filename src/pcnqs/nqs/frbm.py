from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcnqs.nqs.parameterization import FlatParameterLayout, build_layout, flatten_with_layout
from pcnqs.physics.tfim import SquareLattice, periodic_euclidean_distance
from pcnqs.types import FloatArray, SpinArray


@dataclass(frozen=True)
class FrbmParams:
    """Sparse FRBM parameters for ``E(v,h) = -a·v - b·h - v^T W h``."""

    a: FloatArray
    b: FloatArray
    w: FloatArray
    mask: FloatArray

    def named_arrays(self) -> dict[str, FloatArray]:
        return {"a": self.a, "b": self.b, "w": self.w}


@dataclass(frozen=True)
class FrbmOperators:
    """Log-derivative operators ``O_theta(v) = d log Psi / d theta``."""

    o_a: FloatArray
    o_b: FloatArray
    o_w: FloatArray

    def named_arrays(self) -> dict[str, FloatArray]:
        return {"a": self.o_a, "b": self.o_b, "w": self.o_w}


def build_local_mask(lattice: SquareLattice, n_hidden: int, radius: int) -> FloatArray:
    """Build periodic local visible-hidden connectivity mask (guide Eq. S.15)."""

    if radius < 1:
        raise ValueError("radius must be >= 1")

    n_visible = lattice.n_spins

    mask = np.zeros((n_visible, n_hidden), dtype=np.float64)
    for i in range(n_visible):
        for j in range(n_hidden):
            idx_h = int(j % n_visible)
            if periodic_euclidean_distance(lattice, i, idx_h) <= float(radius):
                mask[i, j] = 1.0

    return mask


def init_frbm_params(
    rng: np.random.Generator,
    n_visible: int,
    n_hidden: int,
    mask: FloatArray,
    init_std: float,
) -> FrbmParams:
    """Gaussian parameter initialization matching Algorithm S1 style."""

    if mask.shape != (n_visible, n_hidden):
        raise ValueError(
            f"mask shape {mask.shape} incompatible with {(n_visible, n_hidden)}"
        )

    a = rng.normal(loc=0.0, scale=init_std, size=n_visible).astype(np.float64)
    b = rng.normal(loc=0.0, scale=init_std, size=n_hidden).astype(np.float64)
    w = rng.normal(loc=0.0, scale=init_std, size=(n_visible, n_hidden)).astype(np.float64)
    w *= mask
    return FrbmParams(a=a, b=b, w=w, mask=mask.astype(np.float64))


def hidden_fields(spins: SpinArray, params: FrbmParams) -> FloatArray:
    """Compute hidden pre-activations ``theta_j(v) = b_j + sum_i W_ij v_i``."""

    return params.b + spins.astype(np.float64) @ params.w


def log_unnormalized_probability(spins: SpinArray, params: FrbmParams) -> float:
    """Compute ``log p~(v)`` after analytic hidden marginalization (guide Section 6.1)."""

    theta = hidden_fields(spins, params)
    linear = float(np.dot(params.a, spins.astype(np.float64)))
    return linear + float(np.sum(np.log(2.0 * np.cosh(theta)), dtype=np.float64))


def log_psi(spins: SpinArray, params: FrbmParams) -> float:
    """Unnormalized ``log Psi(v) = 0.5 * log p~(v)`` for stoquastic TFIM states."""

    return 0.5 * log_unnormalized_probability(spins, params)


def single_flip_ratios(spins: SpinArray, params: FrbmParams) -> FloatArray:
    """Analytic ``Psi(v^(i)) / Psi(v)`` for all visible spin flips.

    Implements guide Section 6.3 for sparse FRBM.
    """

    n_visible = spins.shape[0]
    theta = hidden_fields(spins, params)
    log_cosh_theta = np.log(np.cosh(theta))
    ratios = np.ones(n_visible, dtype=np.float64)

    for i in range(n_visible):
        vi = float(spins[i])
        connected = np.nonzero(params.mask[i] > 0.0)[0]
        log_ratio = -params.a[i] * vi

        if connected.size > 0:
            delta = -2.0 * params.w[i, connected] * vi
            shifted = theta[connected] + delta
            log_ratio += 0.5 * float(np.sum(np.log(np.cosh(shifted)) - log_cosh_theta[connected]))

        ratios[i] = float(np.exp(log_ratio))

    return ratios


def log_derivatives(spins: SpinArray, params: FrbmParams) -> FrbmOperators:
    """Compute FRBM log-derivative operators for SR (guide Section 6.2)."""

    theta = hidden_fields(spins, params)
    tanh_theta = np.tanh(theta)

    o_a = 0.5 * spins.astype(np.float64)
    o_b = 0.5 * tanh_theta
    o_w = 0.5 * np.outer(spins.astype(np.float64), tanh_theta) * params.mask
    return FrbmOperators(o_a=o_a, o_b=o_b, o_w=o_w)


def parameter_layout(params: FrbmParams) -> FlatParameterLayout:
    """Return deterministic parameter-vector layout for FRBM SR/CG."""

    return build_layout(params.named_arrays())


def flatten_operators(operators: FrbmOperators, layout: FlatParameterLayout) -> FloatArray:
    """Pack FRBM operators into the flat parameter basis."""

    return flatten_with_layout(operators.named_arrays(), layout)
