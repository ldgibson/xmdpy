from collections import Counter
from typing import TextIO

import numpy as np


def write_xyz_frame(
    file_handle: TextIO,
    atoms: list[str],
    positions: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
) -> None:
    n_atoms = len(atoms)

    file_handle.write(f" {n_atoms}\n")
    file_handle.write("\n")

    for atom, (x, y, z) in zip(atoms, positions):
        file_handle.write(f" {atom}    {x:16.10f} {y:16.10f} {z:16.10f}\n")


def write_xyz(
    filename: str,
    atoms: list[str],
    positions: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
) -> None:
    with open(filename, "w") as traj_file:
        for frame in positions:
            write_xyz_frame(traj_file, atoms, frame)


def write_xdatcar_coords(
    file_handle: TextIO,
    positions: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
) -> None:
    file_handle.write(" Direct\n")

    for x, y, z in positions:
        file_handle.write(f" {x:16.10f} {y:16.10f} {z:16.10f}\n")


def write_xdatcar_header(
    traj_file: TextIO,
    atoms: list[str],
    cell: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    scaling_factor: float = 1.000,
) -> None:
    atom_counts = Counter(atoms)
    traj_file.write("Comment\n")
    traj_file.write(f" {scaling_factor}\n")
    for A, B, C in cell:
        traj_file.write(f"  {A:16.10f} {B:16.10f} {C:16.10f}\n")
    traj_file.write("".join([f" {atom}" for atom in atom_counts.keys()]) + "\n")
    traj_file.write(
        "".join([f" {atom_counts[atom]}" for atom in atom_counts.keys()]) + "\n"
    )


def write_xdatcar(
    filename: str,
    atoms: list[str],
    positions: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
    cell: np.ndarray[tuple[int, int], np.dtype[np.floating]],
) -> None:
    with open(filename, "w") as traj_file:
        write_xdatcar_header(traj_file, atoms, cell)

        for frame in positions:
            write_xdatcar_coords(traj_file, frame)


def write_npt_xdatcar(
    filename: str,
    atoms: list[str],
    positions: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
    cell: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
) -> None:
    with open(filename, "w") as traj_file:
        for cell_i, frame in zip(cell, positions):
            write_xdatcar_header(traj_file, atoms, cell_i)
            write_xdatcar_coords(traj_file, frame)


def main() -> None:
    n_atoms = 300
    n_frames = 1000

    atoms = [
        "Li",
    ] * (n_atoms // 2) + [
        "Cl",
    ] * (n_atoms // 2)

    n_atoms = len(atoms)
    cell_length = 20.123
    cell = np.stack(
        [np.diag([cell_length, cell_length, cell_length]) for _ in range(n_frames)]
    )
    cell *= 0.5 + np.random.random(n_frames)[:, None, None]
    positions = np.random.random((n_frames, n_atoms, 3))

    write_npt_xdatcar("./tests/data/test_npt_1000.XDATCAR", atoms, positions, cell)


if __name__ == "__main__":
    main()
