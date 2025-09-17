import os
from collections.abc import Sequence
from typing import BinaryIO, Literal

import numpy as np

from xmdpy.types import SingleDType, TrajArray

from .base_parser import count_lines, frame_generator


def get_xyz_dims_and_details(
    filename: os.PathLike,
) -> tuple[int, list[str], None, Literal[False]]:
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

    # xyz format does not read cell information
    return n_frames, atoms, None, False


def read_xyz_frames(
    file_handle: BinaryIO,
    n_atoms: int,
    frames: Sequence[int],
    dtype: SingleDType = np.float64,
) -> TrajArray:
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
