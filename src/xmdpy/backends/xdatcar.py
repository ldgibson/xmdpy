from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from xmdpy.types import (
    CellArray,
    CellArray3x3,
    Int1DArray,
    PathLike,
    SingleDType,
    TrajArray,
    TrajNDArray,
)

from .names import DATA_VAR_DIMS, DEFAULT_COORDS, Coord, DataVar
from .on_disk_array import OnDiskArray
from .parsing_utils import count_lines, frame_generator
from .trajectory_formats import TrajectoryFormat


def get_xdatcar_dims_and_details(
    filename: PathLike,
) -> tuple[int, list[str], CellArray3x3, bool]:
    n_lines = count_lines(filename)

    with open(filename, "r") as traj_file:
        next(traj_file)
        scaling_factor = float(traj_file.readline().strip())
        cell = scaling_factor * np.array(
            [traj_file.readline().split() for _ in range(3)], dtype=np.float64
        )

        atom_types = traj_file.readline().split()
        atom_counts = list(map(int, traj_file.readline().split()))

        for _ in range(sum(atom_counts) + 1):
            next(traj_file)

        # read first line of next frame to see if cell information appears
        variable_cell = "Direct" not in traj_file.readline()

    atoms = list(
        Counter(
            **{atom: count for atom, count in zip(atom_types, atom_counts)}
        ).elements()
    )
    lines_per_frame = len(atoms) + 1
    offset = 7

    if variable_cell:
        lines_per_frame += 7
        offset = 0

    n_frames = int((n_lines - offset) / lines_per_frame)
    return n_frames, atoms, cell, variable_cell


def read_xdatcar_frames(
    frames: Int1DArray,
    atoms: Int1DArray,
    xyz_dim: Int1DArray,
    filename: PathLike,
    total_atoms: int,
    cell: CellArray3x3,
    dtype: SingleDType = "float64",
    *,
    direct: bool = True,
    selective_dynamics: bool = False,
) -> TrajArray:
    for dim in (frames, atoms, xyz_dim):
        if not isinstance(dim, np.ndarray):
            raise TypeError(f"invalid index type: {type(dim)}")

    header_length = 7
    offset = 1

    if selective_dynamics:
        offset += 1

    lines_per_frame = total_atoms + offset

    skipped_lines = set(range(offset)).union(
        {atom_id + offset for atom_id in range(total_atoms) if atom_id not in atoms}
    )

    positions = np.zeros((len(frames), len(atoms), 3), dtype=dtype)

    with open(filename, "rb") as file_handle:
        for _ in range(header_length):
            next(file_handle)

        for i, coords in enumerate(
            frame_generator(
                file_handle,
                frames,
                lines_per_frame,
                skip_lines_in_frame=skipped_lines,
                usecol=slice(3),
            )
        ):
            positions[i] = coords

    if not direct:
        return positions[:, :, xyz_dim]

    return (positions @ cell)[:, :, xyz_dim]


def reshape_indices(*keys: Int1DArray) -> tuple[Int1DArray, ...]:
    nkeys = len(keys)
    reshaped_keys: list[Int1DArray] = []
    for axis_index, key in enumerate(keys):
        axis = tuple(n for n in range(nkeys) if n != axis_index)
        reshaped_keys.append(np.expand_dims(key, axis=axis))
    return tuple(reshaped_keys)


@dataclass
class OnDiskXDATCARTrajectory:
    # TODO: add checking for Direct/Cartesian coordinates

    filename: PathLike
    dt: float = 1
    dtype: SingleDType = "float64"

    n_frames: int = field(init=False)
    n_atoms: int = field(init=False)
    atoms: list[str] = field(init=False)
    _cell: CellArray3x3 = field(init=False)
    _broadcasted_cell: CellArray = field(init=False)

    def __post_init__(self) -> None:
        self.n_frames, self.atoms, self._cell, variable_cell = (
            get_xdatcar_dims_and_details(self.filename)
        )
        self.n_atoms = len(self.atoms)
        self._broadcasted_cell = np.broadcast_to(
            self._cell[np.newaxis], (self.n_frames, 3, 3)
        )

        if variable_cell:
            raise NotImplementedError(
                "XDATCAR trajectories without fixed volume not implemented"
            )

    @property
    def positions(self) -> OnDiskArray:
        parser_fn = partial(
            read_xdatcar_frames,
            filename=self.filename,
            total_atoms=self.n_atoms,
            cell=self._cell,
            dtype=self.dtype,
        )
        return OnDiskArray(parser_fn, (self.n_frames, self.n_atoms, 3))

    @property
    def cell(self) -> OnDiskArray:
        def cell_slicer(frames, cell_vector, xyz_dim) -> TrajNDArray:
            frames, cell_vector, xyz_dim = reshape_indices(frames, cell_vector, xyz_dim)
            return self._broadcasted_cell[frames, cell_vector, xyz_dim]

        cell_shape = (self.n_frames, 3, 3)

        return OnDiskArray(
            parser_fn=cell_slicer,
            shape=cell_shape,
        )

    def get_data_vars(
        self,
    ) -> tuple[tuple[str, tuple[str, ...], OnDiskArray | CellArray], ...]:
        return (
            (DataVar.POSITIONS, DATA_VAR_DIMS[DataVar.POSITIONS], self.positions),
            (DataVar.CELL, DATA_VAR_DIMS[DataVar.CELL], self._broadcasted_cell),
        )

    def get_coords(self) -> tuple[tuple[str, tuple[str, ...], TrajNDArray], ...]:
        return (
            (Coord.TIME, DATA_VAR_DIMS[Coord.TIME], np.arange(self.n_frames) * self.dt),
            (Coord.ATOMID, DATA_VAR_DIMS[Coord.ATOMID], np.arange(self.n_atoms)),
            (Coord.ATOM, DATA_VAR_DIMS[Coord.ATOM], np.asarray(self.atoms)),
            (Coord.SPACE, DATA_VAR_DIMS[Coord.SPACE], DEFAULT_COORDS[Coord.SPACE]),
            (Coord.CELL, DATA_VAR_DIMS[Coord.CELL], DEFAULT_COORDS[Coord.CELL]),
        )

    def get_attrs(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_format": TrajectoryFormat.XDATCAR,
        }
