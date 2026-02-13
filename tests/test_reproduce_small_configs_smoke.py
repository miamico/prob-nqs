from __future__ import annotations

import numpy as np

from pcnqs.config.presets import dbm_small_config, frbm_small_config
from pcnqs.sampling.thrml_backend import ThrmlSamplingBackend
from pcnqs.vmc.training import train_dbm, train_frbm


def test_frbm_small_smoke_runs() -> None:
    base = frbm_small_config(seed=5)
    cfg = base.model_copy(
        update={
            "n_iterations": 2,
            "eval_samples": 24,
            "blocking_bins": 2,
            "sampling": base.sampling.model_copy(update={"n_samples": 10, "burn_in": 6, "thin": 1}),
        }
    )

    out = train_frbm(config=cfg, backend=ThrmlSamplingBackend())

    assert len(out.history) == cfg.n_iterations
    assert np.isfinite(out.final_eval.mean)
    assert np.isfinite(out.final_eval.stderr)


def test_dbm_small_smoke_runs() -> None:
    base = dbm_small_config(seed=6)
    cfg = base.model_copy(
        update={
            "n_iterations": 2,
            "eval_samples": 12,
            "blocking_bins": 2,
            "sampling": base.sampling.model_copy(update={"n_samples": 6, "burn_in": 5, "thin": 1}),
            "clamped_sampling": base.clamped_sampling.model_copy(update={"n_steps": 8}),
        }
    )

    out = train_dbm(config=cfg, backend=ThrmlSamplingBackend())

    assert len(out.history) == cfg.n_iterations
    assert np.isfinite(out.final_eval.mean)
    assert np.isfinite(out.final_eval.stderr)
