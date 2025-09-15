import os
from collections.abc import Callable
import numpy as np

from .trajectory_formats import TrajectoryFormat
from .xyz import get_num_frames_and_atoms
from .xdatcar import get_xdatcar_num_frames_atoms_cell


TrajectoryMetadata = tuple[
    int,
    list[str],
    np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]] | None,
]


METADATA_GETTERS: dict[
    TrajectoryFormat, Callable[[os.PathLike], TrajectoryMetadata]
] = {
    TrajectoryFormat.XYZ: get_num_frames_and_atoms,
    TrajectoryFormat.XDATCAR: get_xdatcar_num_frames_atoms_cell,
}
