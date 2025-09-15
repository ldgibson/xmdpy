import os

from typing import BinaryIO
from collections.abc import Sequence
import numpy as np

from xmdpy.types import SingleDType, TrajNDArray

from .core import frame_generator, count_lines


def get_num_frames_and_atoms(filename: os.PathLike) -> tuple[int, list[str], None]:
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

    return n_frames, atoms, None


def read_xyz_frames(
    file_handle: BinaryIO,
    n_atoms: int,
    frames: Sequence[int],
    dtype: SingleDType = np.float64,
) -> TrajNDArray:
    lines_per_frame = n_atoms + 2

    positions = np.zeros((len(frames), n_atoms, 3), dtype=dtype)

    for i, coords in enumerate(
        frame_generator(
            file_handle,
            frames,
            lines_per_frame,
            skip_lines_in_frame=2,
            usecol=slice(1, 4),
        )
    ):
        positions[i] = coords

    return positions
