from collections.abc import Sequence, Callable
from typing import BinaryIO

from xmdpy.types import SingleDType, TrajNDArray

from .xyz import read_xyz_frames
from .xdatcar import read_xdatcar_frames
from .trajectory_formats import TrajectoryFormat


TrajectoryParser = Callable[
    [BinaryIO, int, Sequence[int], SingleDType],
    TrajNDArray,
]

TRAJECTORY_PARSERS: dict[TrajectoryFormat, TrajectoryParser] = {
    TrajectoryFormat.XYZ: read_xyz_frames,
    TrajectoryFormat.XDATCAR: read_xdatcar_frames,
}
