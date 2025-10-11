from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from xmdpy.types import PathLike, SingleDType, TrajNDArray

from .on_disk_array import OnDiskArray
from .trajectory_formats import TrajectoryFormat
from .xyz import OnDiskXYZTrajectory

ON_DISK_TRAJECTORY: dict[TrajectoryFormat, type[OnDiskTrajectory]] = {
    TrajectoryFormat.XYZ: OnDiskXYZTrajectory,
}


@dataclass
class OnDiskTrajectory(Protocol):
    """Interface for trajectories"""

    filename: PathLike
    dt: float
    dtype: SingleDType = "float64"

    def get_data_vars(self) -> dict[str, tuple[tuple[str, ...], OnDiskArray]]: ...

    def get_coords(
        self,
    ) -> dict[str, tuple[tuple[str, ...], TrajNDArray]]: ...

    def get_attrs(self) -> dict[str, Any]: ...
