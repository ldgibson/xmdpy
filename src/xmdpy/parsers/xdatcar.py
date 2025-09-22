from collections import Counter
from collections.abc import Sequence
from typing import BinaryIO

import numpy as np

from xmdpy.types import CellArray3x3, IntArray, PathLike, SingleDType, TrajArray

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
    # frames: Sequence[int] | IntArray,
    indexes: tuple[IntArray, IntArray, IntArray],
    total_atoms: int,
    dtype: SingleDType = np.float64,
    *,
    direct: bool = True,
    selective_dynamics: bool = False,
) -> TrajArray:
    offset = 1

    # skip_lines_in_frame = 1

    if selective_dynamics:
        offset += 1
        # lines_per_frame += 1
        # skip_lines_in_frame += 1

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

    # positions: TrajArray = np.zeros((len(frames), n_atoms, 3), dtype=dtype)
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

    return np.dot(positions, cell)[:, :, xyz_dim]


def read_xdatcar_frames_with_cell(
    file_handle: BinaryIO,
    n_atoms: int,
    frames: Sequence[int] | IntArray,
    dtype: SingleDType = np.float64,
    *,
    direct: bool = True,
    selective_dynamics: bool = False,
) -> TrajArray:
    # 1 comment, 4 lines of cell info, 2 lines of atoms info
    lines_per_frame = n_atoms + 8
    skip_lines_in_frame = {0, 5, 6, 7}

    if selective_dynamics:
        lines_per_frame += 1
        skip_lines_in_frame.add(8)

    scaling_factor = np.zeros(len(frames), dtype=dtype)
    cell = np.zeros((len(frames), 3, 3), dtype=dtype)
    positions = np.zeros((len(frames), n_atoms, 3), dtype=dtype)
    # positions_and_cell = np.zeros((len(frames), n_atoms + 3, 3), dtype=dtype)

    # comment info is always skipped, but scaling factor and cell info are
    # always read each iteration alongside the coordinates data

    for i, coords in enumerate(
        frame_generator(
            file_handle,
            frames,
            lines_per_frame,
            skip_lines_in_frame=skip_lines_in_frame,
            usecol=slice(3),
        )
    ):
        scaling_factor[i] = coords[0][0]
        # positions_and_cell[i, :, :] = coords[1:]
        cell[i] = coords[1:4]
        positions[i] = coords[4:]

    cell *= scaling_factor[:, None, None]
    # positions_and_cell[:, :3] = (
    #     scaling_factor[:, None, None] * positions_and_cell[:, :3]
    # )

    if not direct:
        return np.concatenate([cell, positions], axis=1)

    # ! This needs to broadcasting the multiplcation for each frame
    # positions = np.dot(positions, cell)
    positions = positions * np.diagonal(cell, axis1=1, axis2=2)[:, None, :]
    return np.concatenate([cell, positions], axis=1)
