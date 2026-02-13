from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

SpinArray: TypeAlias = npt.NDArray[np.int8]
SpinBatch: TypeAlias = npt.NDArray[np.int8]
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
ParamsDict: TypeAlias = Mapping[str, FloatArray]

EPS: float = 1.0e-12
