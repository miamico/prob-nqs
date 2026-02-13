from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pcnqs.types import FloatArray


@dataclass(frozen=True)
class ParameterSlice:
    """Slice metadata for flatten/unflatten operations."""

    name: str
    start: int
    stop: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class FlatParameterLayout:
    """Deterministic layout mapping between tensors and flat SR vectors."""

    slices: tuple[ParameterSlice, ...]

    @property
    def size(self) -> int:
        return int(sum(s.stop - s.start for s in self.slices))


def build_layout(named_arrays: dict[str, FloatArray]) -> FlatParameterLayout:
    """Create a deterministic flat-vector layout sorted by key."""

    slices: list[ParameterSlice] = []
    cursor = 0
    for name in sorted(named_arrays.keys()):
        arr = named_arrays[name]
        length = int(np.prod(arr.shape))
        slices.append(
            ParameterSlice(
                name=name,
                start=cursor,
                stop=cursor + length,
                shape=arr.shape,
            )
        )
        cursor += length
    return FlatParameterLayout(tuple(slices))


def flatten_with_layout(
    named_arrays: dict[str, FloatArray], layout: FlatParameterLayout
) -> FloatArray:
    """Pack named parameter tensors into a single contiguous vector."""

    flat = np.zeros(layout.size, dtype=np.float64)
    for sl in layout.slices:
        data = named_arrays[sl.name].reshape(-1)
        flat[sl.start : sl.stop] = data
    return flat


def unflatten_with_layout(vector: FloatArray, layout: FlatParameterLayout) -> dict[str, FloatArray]:
    """Unpack a flat vector back into tensor dictionary format."""

    if vector.ndim != 1:
        raise ValueError("vector must be rank-1")
    if vector.shape[0] != layout.size:
        raise ValueError(
            f"vector length {vector.shape[0]} does not match layout size {layout.size}"
        )

    out: dict[str, FloatArray] = {}
    for sl in layout.slices:
        out[sl.name] = np.asarray(vector[sl.start : sl.stop].reshape(sl.shape), dtype=np.float64)
    return out
