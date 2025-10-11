from pathlib import Path

import pytest

from xmdpy.backends.trajectory_formats import (
    InvalidTrajectoryFormatError,
    TrajectoryFormat,
    get_valid_trajectory_format,
    guess_trajectory_format_str,
)
from xmdpy.types import PathLike


@pytest.mark.parametrize(
    ("filename", "expected"),
    (
        ["mock.xyz", "xyz"],
        ["MOCK.XYZ", "xyz"],
        ["path/to/mock.xyz", "xyz"],
        ["XDATCAR", "xdatcar"],
        ["mock_XDATCAR", "xdatcar"],
        ["path/to/mock_XDATCAR", "xdatcar"],
        [b"mock.xyz", "xyz"],
        [memoryview(b"mock.xyz"), "xyz"],
        [Path("path/to/mock.xyz"), "xyz"],
    ),
)
def test_guess_trajectory_format_str_valid(filename: PathLike, expected: str) -> None:
    result = guess_trajectory_format_str(filename)
    assert result == expected


@pytest.mark.parametrize(
    "filename",
    ["", "test", None, Path(""), "test.badext"],
)
def test_guess_trajectory_format_str_invalid(filename: PathLike) -> None:
    result = guess_trajectory_format_str(filename)
    assert result is None


@pytest.mark.parametrize(
    ("file_format", "expected"),
    (["XYZ", TrajectoryFormat.XYZ], ["XDATCAR", TrajectoryFormat.XDATCAR]),
)
def test_get_valid_trajectory_format_lower_case(
    file_format: str, expected: TrajectoryFormat
) -> None:
    result = get_valid_trajectory_format(file_format)
    assert result is expected


@pytest.mark.parametrize(
    ("file_format", "expected"),
    (["XYZ", TrajectoryFormat.XYZ], ["XDATCAR", TrajectoryFormat.XDATCAR]),
)
def test_get_valid_trajectory_format_upper_case(
    file_format: str, expected: TrajectoryFormat
) -> None:
    result = get_valid_trajectory_format(file_format)
    assert result is expected


def test_get_valid_trajectory_format_none_raises_error() -> None:
    with pytest.raises(InvalidTrajectoryFormatError):
        get_valid_trajectory_format(None)


def test_get_valid_trajectory_format_invalid_raises_error() -> None:
    with pytest.raises(InvalidTrajectoryFormatError):
        get_valid_trajectory_format("invalid")
