from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from pcnqs.nqs.parameterization import FlatParameterLayout, build_layout, flatten_with_layout
from pcnqs.physics.tfim import SquareLattice, periodic_euclidean_distance
from pcnqs.types import EPS, FloatArray, SpinArray, SpinBatch


@dataclass(frozen=True)
class DbmParams:
    """Sparse DBM parameters for ``E(v,h,d)`` from paper Eq. (6)."""

    a: FloatArray
    b: FloatArray
    c: FloatArray
    w_vh: FloatArray
    w_hd: FloatArray
    mask_vh: FloatArray
    mask_hd: FloatArray

    def named_arrays(self) -> dict[str, FloatArray]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "w_hd": self.w_hd,
            "w_vh": self.w_vh,
        }


@dataclass(frozen=True)
class DbmOperators:
    """Conditional-expectation approximation of DBM log-derivative operators."""

    o_a: FloatArray
    o_b: FloatArray
    o_c: FloatArray
    o_w_vh: FloatArray
    o_w_hd: FloatArray

    def named_arrays(self) -> dict[str, FloatArray]:
        return {
            "a": self.o_a,
            "b": self.o_b,
            "c": self.o_c,
            "w_hd": self.o_w_hd,
            "w_vh": self.o_w_vh,
        }


@dataclass(frozen=True)
class DualSamplingStats:
    """Dual-sampling ratio estimates used in TFIM off-diagonal local energy terms."""

    pflip: FloatArray
    pflip_sq: FloatArray
    var_pop: FloatArray
    delta: FloatArray
    corrected_amplitude_ratio: FloatArray


def build_local_mask_between_layers(
    lattice: SquareLattice,
    n_source: int,
    n_target: int,
    radius: int,
) -> FloatArray:
    """Periodic local sparsity mask for adjacent DBM layers (Eq. S.15)."""

    if radius < 1:
        raise ValueError("radius must be >= 1")

    mask = np.zeros((n_source, n_target), dtype=np.float64)
    n_anchor = lattice.n_spins
    for i in range(n_source):
        i_anchor = i % n_anchor
        for j in range(n_target):
            j_anchor = j % n_anchor
            if periodic_euclidean_distance(lattice, i_anchor, j_anchor) <= float(radius):
                mask[i, j] = 1.0
    return mask


def init_dbm_params(
    rng: np.random.Generator,
    n_visible: int,
    n_hidden: int,
    n_deep: int,
    mask_vh: FloatArray,
    mask_hd: FloatArray,
    init_std: float,
) -> DbmParams:
    """Gaussian parameter initialization compatible with Algorithm S1."""

    if mask_vh.shape != (n_visible, n_hidden):
        raise ValueError("mask_vh shape mismatch")
    if mask_hd.shape != (n_hidden, n_deep):
        raise ValueError("mask_hd shape mismatch")

    a = rng.normal(loc=0.0, scale=init_std, size=n_visible).astype(np.float64)
    b = rng.normal(loc=0.0, scale=init_std, size=n_hidden).astype(np.float64)
    c = rng.normal(loc=0.0, scale=init_std, size=n_deep).astype(np.float64)
    w_vh = (
        rng.normal(loc=0.0, scale=init_std, size=(n_visible, n_hidden)).astype(np.float64) * mask_vh
    )
    w_hd = (
        rng.normal(loc=0.0, scale=init_std, size=(n_hidden, n_deep)).astype(np.float64) * mask_hd
    )

    return DbmParams(
        a=a,
        b=b,
        c=c,
        w_vh=w_vh,
        w_hd=w_hd,
        mask_vh=mask_vh.astype(np.float64),
        mask_hd=mask_hd.astype(np.float64),
    )


def joint_energy(v: SpinArray, h: SpinArray, d: SpinArray, params: DbmParams) -> float:
    """DBM energy from paper Eq. (6)."""

    vv = v.astype(np.float64)
    hh = h.astype(np.float64)
    dd = d.astype(np.float64)

    term_v = np.dot(params.a, vv)
    term_h = np.dot(params.b, hh)
    term_d = np.dot(params.c, dd)
    term_vh = float(vv @ params.w_vh @ hh)
    term_hd = float(hh @ params.w_hd @ dd)

    return float(-(term_v + term_h + term_d + term_vh + term_hd))


def visible_local_fields(v: SpinArray, h: SpinArray, params: DbmParams) -> FloatArray:
    """Visible fields used in Eq. (S.9): ``I_i(v,h,d)``."""

    del v
    return params.a + params.w_vh @ h.astype(np.float64)


def dual_sampling_ratios(
    v: SpinArray,
    hidden_samples: SpinBatch,
    params: DbmParams,
) -> DualSamplingStats:
    """Estimate flip ratios with finite-``Nc`` correction (Algorithm S1 lines 25-37)."""

    if hidden_samples.ndim != 2:
        raise ValueError("hidden_samples must be rank-2")

    v_float = v.astype(np.float64)
    local_fields = params.a[None, :] + hidden_samples.astype(np.float64) @ params.w_vh.T
    exponents = -2.0 * local_fields * v_float[None, :]

    pflip = np.asarray(np.mean(np.exp(exponents), axis=0, dtype=np.float64), dtype=np.float64)
    pflip_sq = np.asarray(
        np.mean(np.exp(2.0 * exponents), axis=0, dtype=np.float64),
        dtype=np.float64,
    )

    n_c = hidden_samples.shape[0]
    var_pop = np.asarray((pflip_sq - np.square(pflip)) / float(n_c), dtype=np.float64)

    safe_pflip = np.maximum(pflip, EPS)
    delta = np.asarray(var_pop / (8.0 * safe_pflip * np.sqrt(safe_pflip)), dtype=np.float64)
    corrected = np.asarray(np.sqrt(safe_pflip) + delta, dtype=np.float64)

    return DualSamplingStats(
        pflip=pflip,
        pflip_sq=pflip_sq,
        var_pop=var_pop,
        delta=delta,
        corrected_amplitude_ratio=corrected,
    )


def conditional_log_derivatives(
    v: SpinArray,
    hidden_samples: SpinBatch,
    deep_samples: SpinBatch,
    params: DbmParams,
) -> DbmOperators:
    """Estimate DBM operators via conditional averages over clamped samples."""

    if hidden_samples.shape[0] != deep_samples.shape[0]:
        raise ValueError("hidden and deep samples must share sample axis")

    mean_h = np.mean(hidden_samples.astype(np.float64), axis=0)
    mean_d = np.mean(deep_samples.astype(np.float64), axis=0)

    o_a = 0.5 * v.astype(np.float64)
    o_b = 0.5 * mean_h
    o_c = 0.5 * mean_d
    o_w_vh = 0.5 * np.outer(v.astype(np.float64), mean_h) * params.mask_vh

    hd_outer = hidden_samples.astype(np.float64).T @ deep_samples.astype(np.float64)
    o_w_hd = 0.5 * (hd_outer / float(hidden_samples.shape[0])) * params.mask_hd

    return DbmOperators(o_a=o_a, o_b=o_b, o_c=o_c, o_w_vh=o_w_vh, o_w_hd=o_w_hd)


def parameter_layout(params: DbmParams) -> FlatParameterLayout:
    """Return deterministic parameter-vector layout for DBM SR/CG."""

    return build_layout(params.named_arrays())


def flatten_operators(operators: DbmOperators, layout: FlatParameterLayout) -> FloatArray:
    """Pack DBM operators into the flat parameter basis."""

    return flatten_with_layout(operators.named_arrays(), layout)


def exact_visible_marginal_probability(v: SpinArray, params: DbmParams) -> float:
    """Brute-force tiny-model ``p~(v)`` reference for tests and validation only."""

    total = 0.0
    hidden_states = list(product((-1, 1), repeat=params.b.shape[0]))
    deep_states = list(product((-1, 1), repeat=params.c.shape[0]))

    for h_tuple in hidden_states:
        h = np.asarray(h_tuple, dtype=np.int8)
        for d_tuple in deep_states:
            d = np.asarray(d_tuple, dtype=np.int8)
            total += float(np.exp(-joint_energy(v, h, d, params)))

    return total


def exact_flip_ratio(v: SpinArray, flip_index: int, params: DbmParams) -> float:
    """Brute-force tiny-model ``p(v^(i)) / p(v)`` reference."""

    v_flip = np.array(v, copy=True)
    v_flip[flip_index] = np.int8(-v_flip[flip_index])
    p_base = exact_visible_marginal_probability(v, params)
    p_flip = exact_visible_marginal_probability(v_flip, params)
    return float(p_flip / max(p_base, EPS))
