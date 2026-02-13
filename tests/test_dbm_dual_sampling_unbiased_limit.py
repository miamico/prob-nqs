from __future__ import annotations

from itertools import product

import numpy as np

from pcnqs.nqs.dbm import (
    DbmParams,
    dual_sampling_ratios,
    exact_flip_ratio,
    joint_energy,
)


def _tiny_dbm_params() -> DbmParams:
    a = np.array([0.12, -0.07], dtype=np.float64)
    b = np.array([0.09, -0.11], dtype=np.float64)
    c = np.array([0.05], dtype=np.float64)

    w_vh = np.array([[0.21, -0.17], [0.08, 0.13]], dtype=np.float64)
    w_hd = np.array([[0.15], [-0.19]], dtype=np.float64)

    return DbmParams(
        a=a,
        b=b,
        c=c,
        w_vh=w_vh,
        w_hd=w_hd,
        mask_vh=np.ones_like(w_vh),
        mask_hd=np.ones_like(w_hd),
    )


def _sample_conditional(
    v: np.ndarray,
    params: DbmParams,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    h_states = [np.asarray(t, dtype=np.int8) for t in product((-1, 1), repeat=params.b.shape[0])]
    d_states = [np.asarray(t, dtype=np.int8) for t in product((-1, 1), repeat=params.c.shape[0])]

    joint_states: list[tuple[np.ndarray, np.ndarray]] = []
    logw: list[float] = []
    for h in h_states:
        for d in d_states:
            joint_states.append((h, d))
            logw.append(-joint_energy(v, h, d, params))

    lw = np.asarray(logw, dtype=np.float64)
    lw -= np.max(lw)
    probs = np.exp(lw)
    probs /= np.sum(probs)

    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(joint_states)), size=n_samples, replace=True, p=probs)

    hs = np.stack([joint_states[int(i)][0] for i in idx], axis=0)
    ds = np.stack([joint_states[int(i)][1] for i in idx], axis=0)
    return hs.astype(np.int8), ds.astype(np.int8)


def test_dual_sampling_ratio_converges_to_exact_limit() -> None:
    params = _tiny_dbm_params()
    v = np.array([1, -1], dtype=np.int8)

    exact = exact_flip_ratio(v, flip_index=0, params=params)

    hs, _ = _sample_conditional(v=v, params=params, n_samples=25_000, seed=3)
    stats = dual_sampling_ratios(v=v, hidden_samples=hs, params=params)

    assert abs(stats.pflip[0] - exact) < 2.0e-2


def test_finite_nc_taylor_correction_reduces_bias_on_average() -> None:
    params = _tiny_dbm_params()
    v = np.array([-1, 1], dtype=np.int8)
    exact_amp = float(np.sqrt(exact_flip_ratio(v, flip_index=1, params=params)))

    n_trials = 300
    n_c = 24

    naive_errors: list[float] = []
    corrected_errors: list[float] = []

    for trial in range(n_trials):
        hs, _ = _sample_conditional(v=v, params=params, n_samples=n_c, seed=100 + trial)
        stats = dual_sampling_ratios(v=v, hidden_samples=hs, params=params)

        naive = float(np.sqrt(stats.pflip[1]))
        corrected = float(stats.corrected_amplitude_ratio[1])

        naive_errors.append(abs(naive - exact_amp))
        corrected_errors.append(abs(corrected - exact_amp))

    assert float(np.mean(corrected_errors)) <= float(np.mean(naive_errors))
