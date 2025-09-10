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
    n_atoms = 300
    n_frames = 100000

    atoms = [
        "Li",
    ] * (n_atoms // 2) + [
        "Cl",
    ] * (n_atoms // 2)

    n_atoms = len(atoms)
    cell_length = 20.123

    positions = np.random.random((n_frames, n_atoms, 3)) * cell_length

    write_xyz("./tests/data/test_traj_100ps.xyz", atoms, positions)


if __name__ == "__main__":
    main()
