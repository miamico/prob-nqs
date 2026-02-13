from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from pcnqs.config.presets import frbm_paper_config, frbm_small_config
from pcnqs.sampling.thrml_backend import ThrmlSamplingBackend
from pcnqs.utils.io import save_json
from pcnqs.vmc.training import train_frbm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce FRBM convergence (Fig. 2 style)")
    parser.add_argument("--mode", choices=("small", "paper"), default="small")
    parser.add_argument("--output-dir", type=Path, default=Path("results/fig2_frbm"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "small":
        config = frbm_small_config(seed=7 if args.seed is None else args.seed)
    else:
        config = frbm_paper_config(seed=11 if args.seed is None else args.seed)

    result = train_frbm(config=config, backend=ThrmlSamplingBackend())

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    energies = [m.energy_mean for m in result.history]
    errors = [m.energy_stderr for m in result.history]
    iterations = [m.iteration for m in result.history]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.errorbar(iterations, energies, yerr=errors, fmt="o-", ms=3, lw=1.0, capsize=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Local energy")
    ax.set_title(f"FRBM convergence ({args.mode} mode)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "frbm_convergence.png", dpi=150)
    plt.close(fig)

    payload = {
        "mode": args.mode,
        "config": config.model_dump(),
        "history": [m.__dict__ for m in result.history],
        "final_eval": {
            "mean": result.final_eval.mean,
            "stderr": result.final_eval.stderr,
        },
    }
    save_json(output_dir / "frbm_metrics.json", payload)


if __name__ == "__main__":
    main()
