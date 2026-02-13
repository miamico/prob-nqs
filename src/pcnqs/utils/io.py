from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> None:
    """Create output directory tree when needed."""

    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON with stable key ordering."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Save dense arrays for checkpoints and post-analysis."""

    ensure_dir(path.parent)
    kwargs: dict[str, object] = {name: np.asarray(value) for name, value in arrays.items()}
    np.savez(path, **kwargs)  # type: ignore[arg-type]
