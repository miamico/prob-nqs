# pcnqs

`pcnqs` is a Python library for reproducing the workflow in *Probabilistic Computers for Neural Quantum States* (arXiv:2512.24558v1) using THRML as the production sampler backend.

## Scope

Implemented modules cover:

- 2D TFIM in the `sigma^z` basis with local-energy decomposition.
- Sparse FRBM (`k=2` local connectivity) with analytic single-spin-flip ratios.
- Sparse DBM with dual sampling and finite-`Nc` correction from Algorithm S1.
- Variational Monte Carlo with Stochastic Reconfiguration and matrix-free CG.
- Reproducible experiment entry points with small (CI/laptop) and paper-style presets.

Primary references used:

- `docs/paper_guide.md`
- `docs/references/2512.24558v1.pdf`
- THRML docs and APIs (`thrml>=0.1.3`)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quickstart

```bash
python examples/quickstart_frbm_small.py
python examples/quickstart_dbm_dual_sampling_small.py
```

## Reproduce experiments

```bash
python -m pcnqs.experiments.reproduce_fig2_frbm --mode small --output-dir results/fig2_frbm_small
python -m pcnqs.experiments.reproduce_fig4_fig5_dbm --mode small --output-dir results/fig4_fig5_dbm_small
python -m pcnqs.experiments.scaling_benchmarks --sizes 4 6
```

Paper-scale runs are available with `--mode paper` but require significantly more compute.

## Development checks

```bash
ruff check .
mypy
pytest
```
