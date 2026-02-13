from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt

from pcnqs.config.presets import dbm_small_config, frbm_small_config
from pcnqs.config.schemas import LatticeConfig
from pcnqs.sampling.thrml_backend import ThrmlSamplingBackend
from pcnqs.utils.io import save_json
from pcnqs.vmc.training import train_dbm, train_frbm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small scaling benchmarks")
    parser.add_argument("--mode", choices=("small",), default="small")
    parser.add_argument("--output-dir", type=Path, default=Path("results/scaling"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 6])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = ThrmlSamplingBackend()
    rows: list[dict[str, float | int | str]] = []

    for L in args.sizes:
        frbm_cfg = frbm_small_config(seed=101 + L).model_copy(deep=True)
        frbm_cfg.lattice = LatticeConfig(L=L)

        t0 = time.perf_counter()
        train_frbm(frbm_cfg, backend=backend)
        frbm_dt = time.perf_counter() - t0
        rows.append({"model": "frbm", "L": L, "seconds": frbm_dt})

        dbm_cfg = dbm_small_config(seed=201 + L).model_copy(deep=True)
        dbm_cfg.lattice = LatticeConfig(L=L)

        t1 = time.perf_counter()
        train_dbm(dbm_cfg, backend=backend)
        dbm_dt = time.perf_counter() - t1
        rows.append({"model": "dbm", "L": L, "seconds": dbm_dt})

    save_json(output_dir / "scaling_metrics.json", {"rows": rows})

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for model in ("frbm", "dbm"):
        model_rows = [r for r in rows if r["model"] == model]
        xs = [int(r["L"]) for r in model_rows]
        ys = [float(r["seconds"]) for r in model_rows]
        ax.plot(xs, ys, marker="o", label=model.upper())

    ax.set_xlabel("L")
    ax.set_ylabel("Wall-clock seconds")
    ax.set_title("Small scaling benchmark")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_plot.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
