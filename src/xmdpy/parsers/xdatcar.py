import os
from collections import Counter
from collections.abc import Sequence
from typing import BinaryIO

import numpy as np

from xmdpy.types import CellArray3x3, SingleDType, TrajArray

from .base_parser import count_lines, frame_generator


def get_xdatcar_dims_and_details(
    filename: os.PathLike,
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

        # read first line of next frame to see if cell information appears
        variable_cell = "Direct" not in traj_file.readline()

    atoms = list(
        Counter(
            **{atom: count for atom, count in zip(atom_types, atom_counts)}
        ).elements()
    )

    n_frames = int((n_lines - 7) / (len(atoms) + 1))
    return n_frames, atoms, cell, variable_cell


def read_xdatcar_frames(
    file_handle: BinaryIO,
    n_atoms: int,
    frames: Sequence[int],
    dtype: SingleDType = np.float64,
    *,
    direct: bool = True,
    selective_dynamics: bool = False,
) -> TrajArray:
    lines_per_frame = n_atoms + 1
    skip_lines_in_frame = 1

    if selective_dynamics:
        lines_per_frame += 1
        skip_lines_in_frame += 1

    # If direct=True, cell is used to convert to cartesian coordinates
    next(file_handle)
    scaling_factor = float(file_handle.readline().strip())
    cell = scaling_factor * np.array(
        [file_handle.readline().split() for _ in range(3)], dtype=np.float64
    )

    # skip atom names and counts
    next(file_handle)
    next(file_handle)

    positions: TrajArray = np.zeros((len(frames), n_atoms, 3), dtype=dtype)

    for i, coords in enumerate(
        frame_generator(
            file_handle,
            frames,
            lines_per_frame,
            skip_lines_in_frame=skip_lines_in_frame,
            usecol=slice(3),
        )
    ):
        positions[i] = coords

    if not direct:
        return positions

    return np.dot(positions, cell)


def read_xdatcar_frames_with_cell(
    file_handle: BinaryIO,
    n_atoms: int,
    frames: Sequence[int],
    dtype: SingleDType = np.float64,
    *,
    direct: bool = True,
    selective_dynamics: bool = False,
) -> TrajArray:
    lines_per_frame = (
        n_atoms + 7
    )  # 1 comment, 4 lines of cell info, 2 lines of atoms info
    skip_lines_in_frame = {0, 5, 6, 7}

    if selective_dynamics:
        lines_per_frame += 1
        skip_lines_in_frame.add(8)

    scaling_factor = np.zeros(len(frames), dtype=dtype)
    cell = np.zeros((len(frames), 3, 3), dtype=dtype)
    positions = np.zeros((len(frames), n_atoms, 3), dtype=dtype)

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
        scaling_factor[i] = coords[0]
        cell[i] = coords[:3]
        positions[i] = coords[3:]

    if not direct:
        return positions

    return np.dot(positions, cell)
