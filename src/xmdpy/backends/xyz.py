from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from xmdpy.types import Int1DArray, PathLike, SingleDType, TrajArray, TrajNDArray

from .names import DATA_VAR_DIMS, DEFAULT_COORDS, Coord, DataVar
from .on_disk_array import OnDiskArray
from .parsing_utils import count_lines, frame_generator
from .trajectory_formats import TrajectoryFormat


def get_xyz_dims_and_details(
    filename: PathLike,
) -> tuple[int, list[str]]:
    n_lines = count_lines(filename)

    atoms = []

    with open(filename, "rb") as f:
        n_atoms = int(f.readline().strip())

        _ = f.readline()

        for _ in range(n_atoms):
            line = f.readline()
            fields = line.split()
            atoms.append(fields[0].decode())

    n_frames = int(n_lines / (n_atoms + 2))

    return n_frames, atoms


def read_xyz_frames(
    frames: Int1DArray,
    atoms: Int1DArray,
    xyz_dim: Int1DArray,
    filename: PathLike,
    total_atoms: int,
    dtype: SingleDType = "float64",
) -> TrajArray:
    offset = 2
    lines_per_frame = total_atoms + offset

    for dim in (frames, atoms, xyz_dim):
        if not isinstance(dim, np.ndarray):
            raise TypeError(f"invalid index type: {type(dim)}")

    skipped_lines = set(range(offset)).union(
        {atom_id + offset for atom_id in range(total_atoms) if atom_id not in atoms}
    )

    positions = np.zeros((len(frames), len(atoms), 3), dtype=dtype)

    with open(filename, "rb") as file_handle:
        for i, coords in enumerate(
            frame_generator(
                file_handle,
                frames,
                lines_per_frame,
                skip_lines_in_frame=skipped_lines,
                usecol=slice(1, 4),
            )
        ):
            positions[i] = coords

    return positions[:, :, xyz_dim]


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

    def get_data_vars(self) -> tuple[tuple[str, tuple[str, ...], OnDiskArray], ...]:
        return ((DataVar.POSITIONS, DATA_VAR_DIMS[DataVar.POSITIONS], self.positions),)

    def get_coords(self) -> tuple[tuple[str, tuple[str, ...], TrajNDArray], ...]:
        return (
            (Coord.TIME, DATA_VAR_DIMS[Coord.TIME], np.arange(self.n_frames) * self.dt),
            (Coord.ATOMID, DATA_VAR_DIMS[Coord.ATOMID], np.arange(self.n_atoms)),
            (Coord.ATOM, DATA_VAR_DIMS[Coord.ATOM], np.asarray(self.atoms)),
            (Coord.SPACE, DATA_VAR_DIMS[Coord.SPACE], DEFAULT_COORDS[Coord.SPACE]),
        )

    def get_attrs(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_format": TrajectoryFormat.XYZ,
        }
