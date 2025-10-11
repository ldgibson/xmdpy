from collections import Counter
from typing import BinaryIO

import numpy as np

from xmdpy.types import CellArray3x3, Int1DArray, PathLike, SingleDType, TrajArray

from .base_parser import count_lines, frame_generator


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
    file_handle: BinaryIO,
    indexes: tuple[Int1DArray, Int1DArray, Int1DArray],
    total_atoms: int,
    dtype: SingleDType = np.float64,
    *,
    direct: bool = True,
    selective_dynamics: bool = False,
) -> TrajArray:
    for dim in indexes:
        if not isinstance(dim, np.ndarray):
            raise TypeError(f"invalid index type: {type(dim)}")

    offset = 1

    if selective_dynamics:
        offset += 1

    lines_per_frame = total_atoms + offset

    # If direct=True, cell is used to convert to cartesian coordinates
    next(file_handle)
    scaling_factor = float(file_handle.readline().strip())
    cell = scaling_factor * np.array(
        [file_handle.readline().split() for _ in range(3)], dtype=np.float64
    )

    # skip atom names and counts
    next(file_handle)
    next(file_handle)

    frames, atoms, xyz_dim = indexes

    skipped_lines = set(range(offset)).union(
        {atom_id + offset for atom_id in range(total_atoms) if atom_id not in atoms}
    )

    positions = np.zeros((len(frames), len(atoms), 3), dtype=dtype)

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
