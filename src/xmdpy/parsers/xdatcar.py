from typing import BinaryIO
from collections.abc import Sequence
from collections import Counter
import os
import numpy as np

from xmdpy.types import SingleDType, TrajNDArray

from .core import count_lines, frame_generator


def get_xdatcar_num_frames_atoms_cell(
    filename: os.PathLike,
) -> tuple[
    int, list[str], np.ndarray[tuple[int, int], np.dtype[np.floating | np.integer]]
]:
    n_lines = count_lines(filename)

    with open(filename, "r") as traj_file:
        next(traj_file)
        scaling_factor = float(traj_file.readline().strip())
        cell = scaling_factor * np.array(
            [traj_file.readline().split() for _ in range(3)], dtype=np.float64
        )

        atom_types = traj_file.readline().split()
        atom_counts = list(map(int, traj_file.readline().split()))

    atoms = list(
        Counter(
            **{atom: count for atom, count in zip(atom_types, atom_counts)}
        ).elements()
    )

    n_frames = int((n_lines - 7) / (len(atoms) + 1))
    return n_frames, atoms, cell


def read_xdatcar_frames(
    file_handle: BinaryIO,
    n_atoms: int,
    frames: Sequence[int],
    dtype: SingleDType = np.float64,
    *,
    direct: bool = True,
    selective_dynamics: bool = False,
) -> TrajNDArray:
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

    positions = np.zeros((len(frames), n_atoms, 3), dtype=dtype)

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
