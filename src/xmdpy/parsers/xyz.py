from typing import BinaryIO, Literal

import numpy as np

from xmdpy.types import IntArray, PathLike, SingleDType, TrajArray

from .base_parser import count_lines, frame_generator


def get_xyz_dims_and_details(
    filename: PathLike,
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
    indexes: tuple[IntArray, IntArray, IntArray],
    total_atoms: int,
    dtype: SingleDType = "float64",
) -> TrajArray:
    offset = 2
    lines_per_frame = total_atoms + offset

    for dim in indexes:
        if not isinstance(dim, np.ndarray):
            raise TypeError(f"invalid index type: {type(dim)}")

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
            usecol=slice(1, 4),
        )
    ):
        positions[i] = coords

    return positions[:, :, xyz_dim]
