from __future__ import annotations

from collections.abc import Callable, Container, Sequence
from enum import StrEnum
from functools import partial
from typing import BinaryIO

import numpy as np
import pandas as pd

from xmdpy.trajectory import Trajectory
from xmdpy.types import SingleDType


class TrajectoryFormat(StrEnum):
    XYZ = "xyz"


TrajectoryParsingFn = Callable[
    [BinaryIO, int, Sequence[int] | int | None, SingleDType], Trajectory
]


def get_trajectory_parsing_fn(file_format: str) -> TrajectoryParsingFn:
    if file_format not in TrajectoryFormat:
        raise AttributeError(f"Invalid file format: {file_format}.")
    return TRAJECTORY_PARSING_FNS[TrajectoryFormat(file_format)]


def skip_row(row_id: int, lines_per_frame: int, keep_frames: Container[int]) -> bool:
    """Determines if current row should be skipped.

    Args:
        row_id (int): Raw row number in file.
        lines_per_frame (int): The constant number of lines comprising a single frame.
        keep_frames (Container[int]): Frames that the reader should not skip.

    Returns:
        bool: Skip frame if True
    """
    if int(row_id / lines_per_frame) not in keep_frames:
        return True

    if row_id % lines_per_frame == 0 or (row_id - 1) % lines_per_frame == 0:
        return True

    return False


def parse_xyz_frames(
    file_handle: BinaryIO,
    n_atoms: int,
    frames: Sequence[int] | int | None = None,
    dtype: SingleDType = np.float64,
) -> Trajectory:
    lines_per_frame = n_atoms + 2

    if isinstance(frames, int):
        frames = range(frames)

    if frames is None:

        class ContainsAllFrames(list):
            def __contains__(self, x: int) -> bool:
                return True

            def __getitem__(self, index: int) -> int:
                return 0

        frames = ContainsAllFrames()
        n_rows = None
    else:
        n_rows = len(frames) * n_atoms

    # if the first frame is 0, then do not provide any header size
    header_lines = frames[0] * lines_per_frame or None

    skip_row_fn: Callable[[int], bool] = partial(
        skip_row, lines_per_frame=lines_per_frame, keep_frames=frames
    )

    df = pd.read_csv(
        file_handle,
        sep=r"\s+",
        usecols=[0, 1, 2, 3],
        header=header_lines,
        nrows=n_rows,
        names=["symbol", "x", "y", "z"],
        dtype={"symbol": np.str_, "x": dtype, "y": dtype, "z": dtype},
        skiprows=skip_row_fn,
    )
    return Trajectory(
        symbols=df["symbol"].values.reshape(-1, n_atoms),  # type: ignore
        positions=df[list("xyz")].values.reshape(-1, n_atoms, 3),
    )


TRAJECTORY_PARSING_FNS: dict[TrajectoryFormat, TrajectoryParsingFn] = {
    TrajectoryFormat.XYZ: parse_xyz_frames,
}
