from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from xmdpy.backends.xyz import OnDiskXYZTrajectory
from xmdpy.parsers.trajectory_formats import TrajectoryFormat
from xmdpy.types import IntArray, PathLike, SingleDType

type TrajectoryParserFn = (
    Callable[[IntArray], np.ndarray]
    | Callable[[IntArray, IntArray], np.ndarray]
    | Callable[[IntArray, IntArray, IntArray], np.ndarray]
)
type OuterIndex = (
    int | np.integer | slice | np.ndarray[tuple[int], np.dtype[np.integer]]
)

ON_DISK_TRAJECTORY: dict[TrajectoryFormat, type[OnDiskTrajectory]] = {
    TrajectoryFormat.XYZ: OnDiskXYZTrajectory,
}


def normalize_index_to_array(
    index: OuterIndex | None, upper_bound: int
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    if isinstance(index, np.ndarray):
        if index.ndim > 1:
            raise IndexError("index cannot be multidimensional")
        if np.any(index >= upper_bound):
            raise IndexError("index extends beyond upper bound")
        return np.asarray(index, dtype=np.int64)

    if isinstance(index, (int, np.integer)):
        if index >= upper_bound:
            raise IndexError("index extends beyond upper bound")
        return np.array([index])

    if index is None:
        start, stop, step = 0, upper_bound, 1

    elif isinstance(index, slice):
        start = index.start or 0
        stop = index.stop or upper_bound
        step = index.step or 1

    else:
        raise IndexError(f"invalid index type: {type(index)}")

    if start < 0:
        start = upper_bound + start
    if stop < 0:
        stop = upper_bound + stop

    if start >= upper_bound or stop > upper_bound:
        raise IndexError("index extends beyond upper bound")

    return np.arange(start, stop, step, dtype=np.int64)


@dataclass
class OnDiskArray:
    parser_fn: TrajectoryParserFn
    shape: tuple[int, ...]
    dtype: SingleDType = "float64"

    def __getitem__(self, key: tuple[OuterIndex, ...]) -> np.ndarray:
        indices = tuple(
            normalize_index_to_array(index, self.shape[i])
            for i, index in enumerate(key)
        )

        arr = self.parser_fn(*indices)

        if any(isinstance(dim, int) for dim in key):
            arr = np.squeeze(arr)

        return arr


@dataclass
class OnDiskTrajectory(Protocol):
    """Interface for trajectories"""

    filename: PathLike
    dt: float

    def get_data_vars(self) -> dict[str, tuple[tuple[str, ...], OnDiskArray]]: ...

    def get_coords(self) -> dict[str, tuple[tuple[str, ...], np.ndarray]]: ...

    def get_attrs(self) -> dict[str, Any]: ...
