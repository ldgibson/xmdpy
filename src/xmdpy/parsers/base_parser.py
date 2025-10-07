from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Container, Generator, Sequence
from typing import Any, BinaryIO

import numpy as np

from xmdpy.types import FloatLike, IntArray, PathLike


def frame_generator(
    f: BinaryIO,
    frames: Sequence[int] | IntArray,
    lines_per_frame: int,
    skip_lines_in_frame: int | Container[int] = 0,
    usecol: slice[int | None] | None = None,
) -> Generator[list[list[bytes]]]:
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
    frame_bytes: list[bytes],
    skip_lines_in_frame: Container[int],
    usecol: slice[int | None],
) -> list[list[bytes]]:
    buffer = []

    for i, line in enumerate(frame_bytes):
        if i not in skip_lines_in_frame:
            buffer.append(line.split()[usecol])

    return buffer


def _make_gen(reader: Callable[[int], bytes | None], size=1048576) -> Generator[bytes]:
    buffer = reader(size)
    while buffer:
        yield buffer
        buffer = reader(size)


def count_lines(filename: PathLike) -> int:
    f = open(filename, "rb", buffering=0)
    f_gen = _make_gen(f.read)
    return sum(buffer.count(b"\n") for buffer in f_gen)


TrajSlice = np.ndarray[tuple[int, ...], np.dtype[FloatLike]]
type TrajData = dict[str, Any]


class TrajectoryParser(ABC):
    @abstractmethod
    def get_dimensions(self) -> tuple[int, int, int]:
        """Gets the number of frames, atoms, and spatial dimensions in the trajectory"""

    @abstractmethod
    def get(
        self, properties: list[str], key: tuple[IntArray, IntArray, IntArray]
    ) -> TrajData:
        """Gets property data from a slice of the trajectory"""

    @abstractmethod
    def get_atoms(self) -> list[str]:
        """Gets the list of atoms ordered as they appear in the trajectory"""

    @abstractmethod
    def get_positions(self, key: tuple[IntArray, IntArray, IntArray]) -> TrajSlice:
        """Gets the positions for a slice of the trajectory"""

    def get_cell(self, key: tuple[IntArray, IntArray, IntArray]) -> TrajSlice | None:
        """Gets the cell parameters for a slice of the trajectory if they exist"""
        return None


# filename = ".xyz"
# traj = Trajectory(filename, dt=1)

# data_vars = traj.get_data_vars()
# coords = traj.get_coordinates()
# attrs = traj.get_attributes()

# data = data_vars["xyz"][1]
# x = data[(0, 1)]
