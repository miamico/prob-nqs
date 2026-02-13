from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pcnqs.types import FloatArray


@dataclass(frozen=True)
class CgResult:
    """Conjugate-gradient solution diagnostics."""

    solution: FloatArray
    converged: bool
    iterations: int
    residual_norm: float
    residual_history: FloatArray


def solve_cg(
    matvec: Callable[[FloatArray], FloatArray],
    rhs: FloatArray,
    rtol: float,
    max_iterations: int,
    x0: FloatArray | None = None,
) -> CgResult:
    """Matrix-free conjugate gradient solver for SPD systems."""

    if rhs.ndim != 1:
        raise ValueError("rhs must be rank-1")
    if rtol <= 0.0:
        raise ValueError("rtol must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    x = np.zeros_like(rhs) if x0 is None else np.array(x0, copy=True)
    r = rhs - matvec(x)
    p = np.array(r, copy=True)

    rhs_norm = float(np.linalg.norm(rhs))
    target = rtol * max(rhs_norm, 1.0)

    residual_sq = float(np.dot(r, r))
    history: list[float] = [float(np.sqrt(residual_sq))]

    if history[-1] <= target:
        return CgResult(
            solution=x,
            converged=True,
            iterations=0,
            residual_norm=history[-1],
            residual_history=np.asarray(history, dtype=np.float64),
        )

    converged = False
    iters = 0

    for iteration in range(1, max_iterations + 1):
        iters = iteration
        ap = matvec(p)
        denom = float(np.dot(p, ap))
        if abs(denom) < 1.0e-20:
            break

        alpha = residual_sq / denom
        x = x + alpha * p
        r = r - alpha * ap

        new_residual_sq = float(np.dot(r, r))
        residual_norm = float(np.sqrt(new_residual_sq))
        history.append(residual_norm)

        if residual_norm <= target:
            converged = True
            residual_sq = new_residual_sq
            break

        beta = new_residual_sq / residual_sq
        p = r + beta * p
        residual_sq = new_residual_sq

    return CgResult(
        solution=x,
        converged=converged,
        iterations=iters,
        residual_norm=float(np.sqrt(residual_sq)),
        residual_history=np.asarray(history, dtype=np.float64),
    )
