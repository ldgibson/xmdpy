from collections.abc import Callable
from typing import BinaryIO

from xmdpy.types import CellArray3x3, IntArray, PathLike, SingleDType, TrajArray

from .trajectory_formats import TrajectoryFormat
from .xdatcar import (
    get_xdatcar_dims_and_details,
    read_xdatcar_frames,
    read_xdatcar_frames_with_cell,
)
from .xyz import get_xyz_dims_and_details, read_xyz_frames_atom_slice

type TrajectoryParser = Callable[
    [BinaryIO, tuple[IntArray, IntArray, IntArray], int, SingleDType],
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
    TrajectoryFormat.XYZ: read_xyz_frames_atom_slice,
    TrajectoryFormat.XDATCAR: read_xdatcar_frames,
    TrajectoryFormat.XDATCAR_NPT: read_xdatcar_frames_with_cell,
}


TRAJECTORY_DETAILS_GETTERS: dict[TrajectoryFormat, TrajectoryDetailsGetter] = {
    TrajectoryFormat.XYZ: get_xyz_dims_and_details,
    TrajectoryFormat.XDATCAR: get_xdatcar_dims_and_details,
    TrajectoryFormat.XDATCAR_NPT: get_xdatcar_dims_and_details,
}
