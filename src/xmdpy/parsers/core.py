from collections.abc import Callable
from typing import BinaryIO

from xmdpy.types import CellArray3x3, Int1DArray, PathLike, SingleDType, TrajArray

from .trajectory_formats import TrajectoryFormat
from .xdatcar import (
    get_xdatcar_dims_and_details,
    read_xdatcar_frames,
)
from .xyz import get_xyz_dims_and_details, read_xyz_frames

type TrajectoryParser = Callable[
    [BinaryIO, tuple[Int1DArray, Int1DArray, Int1DArray], int, SingleDType],
    TrajArray,
]

type TrajectoryDetailsGetter = Callable[
    [PathLike],
    tuple[
        int,
        list[str],
        CellArray3x3 | None,
        bool,
    ],
]

TRAJECTORY_PARSERS: dict[TrajectoryFormat, TrajectoryParser] = {
    TrajectoryFormat.XYZ: read_xyz_frames,
    TrajectoryFormat.XDATCAR: read_xdatcar_frames,
}


TRAJECTORY_DETAILS_GETTERS: dict[TrajectoryFormat, TrajectoryDetailsGetter] = {
    TrajectoryFormat.XYZ: get_xyz_dims_and_details,
    TrajectoryFormat.XDATCAR: get_xdatcar_dims_and_details,
}
