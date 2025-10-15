from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np

from xmdpy.types import PathLike, SingleDType, TrajNDArray

from .on_disk_array import OuterIndex
from .trajectory_formats import TrajectoryFormat
from .xdatcar import OnDiskXDATCARTrajectory
from .xyz import OnDiskXYZTrajectory

ON_DISK_TRAJECTORY: dict[TrajectoryFormat, type[OnDiskTrajectory]] = {
    TrajectoryFormat.XYZ: OnDiskXYZTrajectory,
    TrajectoryFormat.XDATCAR: OnDiskXDATCARTrajectory,
}


class IndexableShapedArray(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    def __getitem__(
        self: Self, key: OuterIndex | tuple[OuterIndex, ...], /
    ) -> np.ndarray[tuple[int, ...], np.dtype[Any]]: ...


@dataclass
class OnDiskTrajectory(Protocol):
    """Interface for trajectories"""

    filename: PathLike
    dt: float
    dtype: SingleDType = "float64"

    def get_data_vars(
        self,
    ) -> tuple[tuple[str, tuple[str, ...], IndexableShapedArray], ...]: ...

    def get_coords(
        self,
    ) -> tuple[tuple[str, tuple[str, ...], TrajNDArray], ...]: ...

    def get_attrs(self) -> dict[str, Any]: ...
