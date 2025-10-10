import numpy as np

from xmdpy.types import IntArray, PathLike, SingleDType, TrajArray

from .base_parser import count_lines, frame_generator


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

    # xyz format does not read cell information
    return n_frames, atoms


def read_xyz_frames(
    filename: PathLike,
    frames: IntArray,
    atoms: IntArray,
    xyz_dim: IntArray,
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
