from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Container, Sequence
from enum import StrEnum
from typing import BinaryIO

import numpy as np
import pandas as pd

from xmdpy.types import SingleDType, TrajNDArray


class TrajectoryFormat(StrEnum):
    XYZ = "xyz"


Array1D_uint64 = np.ndarray[tuple[int], np.dtype[np.uint64]]

TrajectoryParsingFn = Callable[
    [BinaryIO, int, Sequence[int], SingleDType],
    TrajNDArray,
]


def get_trajectory_parsing_fn(file_format: str) -> TrajectoryParsingFn:
    if file_format not in TrajectoryFormat:
        raise AttributeError(f"Invalid file format: {file_format}.")
    return TRAJECTORY_PARSING_FNS[TrajectoryFormat(file_format)]


def skip_lines(
    frames: Sequence[int],
    lines_per_frame: int,
    lines_of_each_frame: Sequence[int],
    offset: int,
) -> list[int]:
    # ! This does not account for any frames skipped in the bounds
    return np.concatenate(
        [
            np.arange(offset),
            np.ravel(
                np.asarray(lines_of_each_frame)[None, :]
                + np.asarray(frames)[:, None] * lines_per_frame
            ),
        ]
    ).tolist()


def skip_row(
    row_id: int,
    frames: Container[int],
    lines_per_frame: int,
    lines_of_each_frame: Iterable[int],
    offset: int = 0,
) -> bool:
    if row_id < offset:
        return True

    if int(row_id / lines_per_frame) in frames:
        return (row_id % lines_per_frame == 0) or ((row_id - 1) % lines_per_frame == 0)
    return False


def parse_xyz_frames(
    file_handle: BinaryIO,
    n_atoms: int,
    frames: Sequence[int],
    dtype: SingleDType = np.float64,
) -> TrajNDArray:
    lines_per_frame = n_atoms + 2
    offset = frames[0] * lines_per_frame
    n_rows = len(frames) * n_atoms  # only count rows with XYZ information
    lines_to_skip = skip_lines(
        frames,
        lines_per_frame,
        lines_of_each_frame=[0, 1],  # skip first two lines of each XYZ frame
        offset=offset,
    )

    df = pd.read_csv(
        file_handle,
        sep=r"\s+",
        usecols=[1, 2, 3],
        nrows=n_rows,
        names=["x", "y", "z"],
        dtype={"x": dtype, "y": dtype, "z": dtype},
        skiprows=lines_to_skip,
        engine="c",
    )
    return df.to_numpy().reshape(-1, n_atoms, 3)


def read_xyz_frames(
    file_handle: BinaryIO,
    n_atoms: int,
    frames: Sequence[int],
    dtype: SingleDType = np.float64,
) -> TrajNDArray:
    lines_per_frame = n_atoms + 2

    if frames[0] > 0:
        for _ in range(frames[0] * lines_per_frame):
            _ = file_handle.readline()

    all_bounded_frames = range(frames[0], frames[-1] + 1)
    skip_frames = set(all_bounded_frames).difference(set(frames))

    positions = np.zeros((len(frames), n_atoms, 3), dtype=dtype)

    i = 0
    for frame in all_bounded_frames:
        if frame in skip_frames:
            [file_handle.readline() for _ in range(lines_per_frame)]
            continue
        positions[i] = _read_xyz_frame(file_handle, n_atoms)
        i += 1

    return positions


def _read_xyz_frame(
    file_handle: BinaryIO,
    n_atoms: int,
) -> list[list[str]]:
    buffer = []

    _ = file_handle.readline()
    _ = file_handle.readline()

    for _ in range(n_atoms):
        buffer.append(file_handle.readline().split()[1:4])

    return buffer


TRAJECTORY_PARSING_FNS: dict[TrajectoryFormat, TrajectoryParsingFn] = {
    TrajectoryFormat.XYZ: read_xyz_frames,
}


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
