from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Container, Generator, Sequence
from typing import BinaryIO

import numpy as np

from xmdpy.types import SingleDType, TrajNDArray


TrajectoryParsingFn = Callable[
    [BinaryIO, int, Sequence[int], SingleDType],
    TrajNDArray,
]


def frame_generator(
    f: BinaryIO,
    frames: Sequence[int],
    lines_per_frame: int,
    skip_lines_in_frame: int | Container[int] = 0,
    usecol: slice[int] | None = None,
) -> Generator[list[list[str]]]:
    if isinstance(skip_lines_in_frame, int):
        skip_lines_in_frame = set(range(skip_lines_in_frame))

    if usecol is None:
        usecol = slice(None)

    # seek to the first line in `frames`
    if frames[0] > 0:
        for _ in range(frames[0] * lines_per_frame):
            next(f)

    prev = frames[0] - 1
    for frame in frames:
        if frame - prev > 1:
            for _ in range((frame - prev - 1) * lines_per_frame):
                next(f)

        yield _parse_frame(
            frame_bytes=[f.readline() for _ in range(lines_per_frame)],
            skip_lines_in_frame=skip_lines_in_frame,
            usecol=usecol,
        )
        prev = frame


def _parse_frame(
    frame_bytes: list[bytes], skip_lines_in_frame: Container[int], usecol: slice[int]
) -> list[list[str]]:
    buffer = []

    for i, line in enumerate(frame_bytes):
        if i not in skip_lines_in_frame:
            buffer.append(line.split()[usecol])

    return buffer


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


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def count_lines(filename):
    f = open(filename, "rb", buffering=0)
    f_gen = _make_gen(f.read)
    return sum(buffer.count(b"\n") for buffer in f_gen)


def get_xdatcar_num_frames_atoms_cell(
    filename: os.PathLike,
) -> tuple[int, list[str], np.ndarray[tuple[int, ...]]]:
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


def get_num_frames_and_atoms(filename: os.PathLike) -> tuple[int, list[str]]:
    total_size = os.path.getsize(filename)
    atoms = []
    frame_size = 0

    with open(filename, "rb") as f:
        line = f.readline()
        frame_size += len(line)
        n_atoms = int(line.strip())

        line = f.readline()
        frame_size += len(line)

        for _ in range(n_atoms):
            line = f.readline()
            frame_size += len(line)
            fields = line.split()
            atoms.append(fields[0].decode())

    if total_size % frame_size == 0:
        n_frames = int(total_size / frame_size)
    else:
        n_frames = int((total_size + 1) / frame_size)

    return n_frames, atoms
