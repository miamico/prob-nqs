from __future__ import annotations

import numpy as np

from pcnqs.physics.tfim import (
    SquareLattice,
    build_nearest_neighbor_bonds,
    diagonal_energy,
    local_energy_from_ratios,
)


def test_nearest_neighbor_bonds_count() -> None:
    lattice = SquareLattice(L=4)
    bonds = build_nearest_neighbor_bonds(lattice.L)
    assert bonds.shape == (2 * lattice.n_spins, 2)


def test_tfim_diagonal_energy_known_case() -> None:
    bonds = build_nearest_neighbor_bonds(2)
    spins = np.ones(4, dtype=np.int8)
    energy = diagonal_energy(spins=spins, bonds=bonds, J=1.0)
    assert energy == -8.0


def test_local_energy_decomposition() -> None:
    bonds = build_nearest_neighbor_bonds(2)
    spins = np.ones(4, dtype=np.int8)
    ratios = np.ones(4, dtype=np.float64)

    eloc = local_energy_from_ratios(
        spins=spins,
        bonds=bonds,
        J=1.0,
        gamma_x=3.0,
        psi_flip_ratios=ratios,
    )
    assert eloc == -20.0
