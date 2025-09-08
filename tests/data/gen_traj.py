from typing import TextIO

import numpy as np
import numpy.typing as npt


def write_xyz_frame(
    file_handle: TextIO, atoms: list[str], positions: npt.ArrayLike
) -> None:
    n_atoms = len(atoms)

    file_handle.write(f" {n_atoms}\n")
    file_handle.write("\n")

    for atom, (x, y, z) in zip(atoms, positions):
        file_handle.write(f" {atom}    {x:16.10f} {y:16.10f} {z:16.10f}\n")


def write_xyz(filename: str, atoms: list[str], positions: npt.ArrayLike) -> None:
    with open(filename, "w") as traj_file:
        for frame in positions:
            write_xyz_frame(traj_file, atoms, frame)


def main() -> None:
    atoms = [
        "Li",
    ] * 100 + [
        "Cl",
    ] * 100
    n_atoms = len(atoms)
    cell_length = 20.123
    n_frames = 1000
    positions = np.random.random((n_frames, n_atoms, 3)) * cell_length

    write_xyz("./test_traj.xyz", atoms, positions)


if __name__ == "__main__":
    main()
