import io
from math import ceil

import pytest

from xmdpy.parsers.base_parser import (
    _make_gen,
    _parse_frame,
    count_lines,
    frame_generator,
)

TEST_BYTES = b"a b c d\n1 2 3 4\nfoo bar baz quux\n10 20 30 40\n"


@pytest.mark.parametrize(
    "skip_lines_in_frame",
    [
        {},
        {0},
        {0, 1},
        {0, 2},
        {1, 3},
    ],
)
@pytest.mark.parametrize(
    "usecol",
    [
        slice(None),
        slice(-1),
        slice(0, -1),
        slice(1, 4),
        slice(3, 0, -2),
    ],
)
def test_parse_frame(skip_lines_in_frame, usecol) -> None:
    bytes_list = TEST_BYTES.split(b"\n")

    result = _parse_frame(
        bytes_list, skip_lines_in_frame=skip_lines_in_frame, usecol=usecol
    )

    expected = [
        line.split()[usecol]
        for i, line in enumerate(bytes_list)
        if i not in skip_lines_in_frame
    ]

    for res_line, exp_line in zip(result, expected):
        for res_field, exp_field in zip(res_line, exp_line):
            assert res_field == exp_field


@pytest.mark.parametrize(
    "frames",
    [
        [0],
        [1],
        [0, 2],
        [0, 3],
    ],
)
@pytest.mark.parametrize(
    "lines_per_frame",
    [1, 2, 3, 4],
)
def test_frame_generator(frames, lines_per_frame) -> None:
    TEST_FRAMES = b"".join(
        [TEST_BYTES for _ in range(frames[-1] * ceil(lines_per_frame / 4))]
    )

    expected = [
        TEST_FRAMES.split(b"\n")[i * lines_per_frame : (i + 1) * lines_per_frame]
        for i in frames
    ]
    expected = [[line.split() for line in frame] for frame in expected]

    f = io.BytesIO(TEST_FRAMES)
    for res_frame, exp_frame in zip(
        frame_generator(f, frames=frames, lines_per_frame=lines_per_frame), expected
    ):
        for res_line, exp_line in zip(res_frame, exp_frame):
            for res_field, exp_field in zip(res_line, exp_line):
                assert res_field == exp_field


@pytest.mark.parametrize("size", (2, 5, 256, 1024))
def test_make_gen(size) -> None:
    f = io.BytesIO(TEST_BYTES)
    buffer_gen = _make_gen(f.read, size=size)

    for i, result in enumerate(buffer_gen):
        expected = TEST_BYTES[size * i : size * (i + 1)]
        assert result == expected


def test_count_lines(fs) -> None:
    fs.create_file("tests/data/test_file.txt", contents=TEST_BYTES)

    result = count_lines("tests/data/test_file.txt")
    expected = TEST_BYTES.count(b"\n")
    assert result == expected
