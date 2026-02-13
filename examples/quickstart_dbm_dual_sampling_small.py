from __future__ import annotations

from pcnqs.config.presets import dbm_small_config
from pcnqs.sampling.thrml_backend import ThrmlSamplingBackend
from pcnqs.vmc.training import train_dbm

if __name__ == "__main__":
    config = dbm_small_config(seed=0)
    result = train_dbm(config=config, backend=ThrmlSamplingBackend())

    print("DBM dual-sampling small run complete")
    print(f"Iterations: {len(result.history)}")
    print(f"Final energy: {result.final_eval.mean:.6f} ± {result.final_eval.stderr:.6f}")
