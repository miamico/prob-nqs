from __future__ import annotations

from itertools import product

import numpy as np

from pcnqs.sampling.backend import FrbmSamplingModel
from pcnqs.sampling.thrml_backend import ThrmlSamplingBackend


def _exact_joint_probs(a: float, b: float, w: float, beta: float) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = {}
    for v, h in product((-1, 1), repeat=2):
        exponent = beta * (a * v + b * h + w * v * h)
        weights[(v, h)] = float(np.exp(exponent))

    z = sum(weights.values())
    return {k: v / z for k, v in weights.items()}


def test_thrml_backend_joint_matches_small_ising_distribution() -> None:
    backend = ThrmlSamplingBackend()

    a = 0.30
    b = -0.15
    w = 0.40
    beta = 1.0

    model = FrbmSamplingModel(
        visible_biases=np.array([a], dtype=np.float64),
        hidden_biases=np.array([b], dtype=np.float64),
        weights_vh=np.array([[w]], dtype=np.float64),
        beta=beta,
    )

    samples = backend.sample_free(
        model=model,
        n_samples=8_000,
        burn_in=200,
        thin=1,
        seed=1234,
    )

    empirical: dict[tuple[int, int], float] = {}
    for v, h in product((-1, 1), repeat=2):
        mask = (samples.visible[:, 0] == v) & (samples.hidden[:, 0] == h)
        empirical[(v, h)] = float(np.mean(mask))

    expected = _exact_joint_probs(a=a, b=b, w=w, beta=beta)
    for state, prob in expected.items():
        assert abs(empirical[state] - prob) < 0.05
