import pytest

from xmdpy.parsers.trajectory_formats import (
    InvalidTrajectoryFormatError,
    TrajectoryFormat,
    get_valid_trajectory_format,
)


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


def test_get_valid_trajectory_format_raise_invalid_error() -> None:
    with pytest.raises(InvalidTrajectoryFormatError):
        get_valid_trajectory_format("invalid")
