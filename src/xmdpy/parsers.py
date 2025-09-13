from __future__ import annotations

import os
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
