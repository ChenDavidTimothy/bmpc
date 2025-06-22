from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


# NUMERICAL SAFETY TYPES
FloatArray: TypeAlias = NDArray[np.float64]  # for numerical precision
NumericArrayLike: TypeAlias = (
    NDArray[np.floating[Any]]
    | NDArray[np.integer[Any]]
    | Sequence[float]
    | Sequence[int]
    | list[float]
    | list[int]
    | Sequence[Sequence[float]]
    | Sequence[Sequence[int]]
    | list[list[float]]
    | list[list[int]]
)
