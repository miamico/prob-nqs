from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcnqs.types import FloatArray, IntArray, SpinArray, SpinBatch


@dataclass(frozen=True)
class SquareLattice:
    """Periodic LxL square lattice in row-major index order."""

    L: int

    def __post_init__(self) -> None:
        if self.L < 2:
            raise ValueError("L must be >= 2 for periodic TFIM lattices")

    @property
    def n_spins(self) -> int:
        return self.L * self.L

    def index(self, row: int, col: int) -> int:
        return (row % self.L) * self.L + (col % self.L)

    def coordinate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.n_spins:
            raise ValueError(f"index out of bounds: {idx}")
        return divmod(idx, self.L)


@dataclass(frozen=True)
class TfimHamiltonian:
    """2D TFIM Hamiltonian in the sigma^z basis (paper Eq. (4))."""

    lattice: SquareLattice
    J: float
    gamma_x: float

    def __post_init__(self) -> None:
        if self.gamma_x <= 0.0:
            raise ValueError("gamma_x must be > 0")

    @property
    def bonds(self) -> IntArray:
        return build_nearest_neighbor_bonds(self.lattice.L)


def build_nearest_neighbor_bonds(L: int) -> IntArray:
    """Build unique nearest-neighbor bonds for periodic square lattice.

    Returns an array of shape ``(2 * L * L, 2)`` containing right and down
    neighbors for each site, which is sufficient to cover each undirected bond
    exactly once.
    """

    lattice = SquareLattice(L)
    bonds: list[tuple[int, int]] = []
    for r in range(L):
        for c in range(L):
            i = lattice.index(r, c)
            bonds.append((i, lattice.index(r, c + 1)))
            bonds.append((i, lattice.index(r + 1, c)))
    return np.asarray(bonds, dtype=np.int64)


def periodic_euclidean_distance(
    lattice: SquareLattice,
    idx_a: int,
    idx_b: int,
) -> float:
    """Distance definition from Eq. (S.15) in the supplement."""

    ar, ac = lattice.coordinate(idx_a)
    br, bc = lattice.coordinate(idx_b)

    dr = abs(ar - br)
    dc = abs(ac - bc)
    dr = min(dr, lattice.L - dr)
    dc = min(dc, lattice.L - dc)
    return float(np.sqrt(dr * dr + dc * dc))


def diagonal_energy(spins: SpinArray, bonds: IntArray, J: float) -> float:
    """Compute ``-J * sum_<i,j> s_i s_j`` for one visible configuration."""

    if spins.ndim != 1:
        raise ValueError(f"spins must be rank-1, received shape {spins.shape}")

    pair_products = spins[bonds[:, 0]] * spins[bonds[:, 1]]
    return float(-J * np.sum(pair_products, dtype=np.float64))


def diagonal_energy_batch(spins: SpinBatch, bonds: IntArray, J: float) -> FloatArray:
    """Vectorized diagonal TFIM contribution for a batch of spin states."""

    if spins.ndim != 2:
        raise ValueError(f"spins must be rank-2, received shape {spins.shape}")
    pair_products = spins[:, bonds[:, 0]] * spins[:, bonds[:, 1]]
    energies = -J * np.sum(pair_products, axis=1, dtype=np.float64)
    return np.asarray(energies, dtype=np.float64)


def flip_spin(spins: SpinArray, index: int) -> SpinArray:
    """Return a copy of ``spins`` with one spin flipped."""

    if index < 0 or index >= spins.shape[0]:
        raise ValueError(f"index out of bounds: {index}")
    flipped = np.array(spins, copy=True)
    flipped[index] = np.int8(-flipped[index])
    return flipped


def local_energy_from_ratios(
    spins: SpinArray,
    bonds: IntArray,
    J: float,
    gamma_x: float,
    psi_flip_ratios: FloatArray,
) -> float:
    """Assemble TFIM local energy from diagonal term and flip ratios.

    Uses the decomposition in guide Section 2.2.
    """

    if psi_flip_ratios.ndim != 1:
        raise ValueError("psi_flip_ratios must be rank-1")
    if psi_flip_ratios.shape[0] != spins.shape[0]:
        raise ValueError(
            "psi_flip_ratios length must equal number of visible spins: "
            f"{psi_flip_ratios.shape[0]} != {spins.shape[0]}"
        )

    diag = diagonal_energy(spins=spins, bonds=bonds, J=J)
    offdiag = -gamma_x * float(np.sum(psi_flip_ratios, dtype=np.float64))
    return diag + offdiag
