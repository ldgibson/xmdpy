from __future__ import annotations

import os
from collections.abc import Callable, Container, Generator, Sequence
from typing import BinaryIO


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


def _make_gen(reader: Callable[[int], bytes | None]) -> Generator[bytes]:
    buffer = reader(1024 * 1024)
    while buffer:
        yield buffer
        buffer = reader(1024 * 1024)


def count_lines(filename: os.PathLike) -> int:
    f = open(filename, "rb", buffering=0)
    f_gen = _make_gen(f.read)
    return sum(buffer.count(b"\n") for buffer in f_gen)
