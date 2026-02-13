from __future__ import annotations

import numpy as np

from pcnqs.nqs.frbm import (
    build_local_mask,
    init_frbm_params,
    log_psi,
    single_flip_ratios,
)
from pcnqs.physics.tfim import SquareLattice


def test_frbm_analytic_ratios_match_logpsi_differences() -> None:
    rng = np.random.default_rng(123)
    lattice = SquareLattice(L=3)

    n_visible = lattice.n_spins
    n_hidden = n_visible
    mask = build_local_mask(lattice=lattice, n_hidden=n_hidden, radius=2)
    params = init_frbm_params(
        rng=rng,
        n_visible=n_visible,
        n_hidden=n_hidden,
        mask=mask,
        init_std=0.2,
    )

    spins = rng.choice([-1, 1], size=n_visible).astype(np.int8)

    analytic = single_flip_ratios(spins, params)

    brute = np.zeros(n_visible, dtype=np.float64)
    base = log_psi(spins, params)
    for i in range(n_visible):
        flipped = np.array(spins, copy=True)
        flipped[i] = np.int8(-flipped[i])
        brute[i] = float(np.exp(log_psi(flipped, params) - base))

    np.testing.assert_allclose(analytic, brute, rtol=1.0e-10, atol=1.0e-10)
