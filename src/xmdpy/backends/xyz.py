from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from xmdpy.parsers.trajectory_formats import TrajectoryFormat
from xmdpy.parsers.xyz import get_xyz_dims_and_details, read_xyz_frames
from xmdpy.types import PathLike, SingleDType, TrajNDArray

from .names import DATA_VAR_DIMS, DEFAULT_COORDS, Coord, DataVar
from .on_disk_trajectory import OnDiskArray


@dataclass
class OnDiskXYZTrajectory:
    filename: PathLike
    dt: float = 1
    dtype: SingleDType = "float64"

    n_frames: int = field(init=False)
    n_atoms: int = field(init=False)
    atoms: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.n_frames, self.atoms = get_xyz_dims_and_details(self.filename)
        self.n_atoms = len(self.atoms)

    @property
    def positions(self) -> OnDiskArray:
        parser_fn = partial(
            read_xyz_frames,
            filename=self.filename,
            total_atoms=self.n_atoms,
            dtype=self.dtype,
        )
        return OnDiskArray(parser_fn, (self.n_frames, self.n_atoms, 3))

    def get_data_vars(self) -> dict[str, tuple[tuple[str, ...], OnDiskArray]]:
        return {DataVar.POSITIONS: (DATA_VAR_DIMS[DataVar.POSITIONS], self.positions)}

    def get_coords(self) -> dict[str, tuple[tuple[str, ...], TrajNDArray]]:
        return {
            Coord.TIME: (DATA_VAR_DIMS[Coord.TIME], np.arange(self.n_frames) * self.dt),
            Coord.ATOMID: (DATA_VAR_DIMS[Coord.ATOMID], np.arange(self.n_atoms)),
            Coord.ATOM: (DATA_VAR_DIMS[Coord.ATOM], np.asarray(self.atoms)),
            Coord.SPACE: (DATA_VAR_DIMS[Coord.SPACE], DEFAULT_COORDS[Coord.SPACE]),
        }

    def get_attrs(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_format": TrajectoryFormat.XYZ,
        }
