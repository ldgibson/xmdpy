import os.path
from enum import StrEnum

from xmdpy.types import PathLike


class TrajectoryFormat(StrEnum):
    XYZ = "xyz"
    XDATCAR = "xdatcar"


class InvalidTrajectoryFormatError(NotImplementedError):
    pass


def guess_trajectory_format_str(filename_or_obj: PathLike) -> str | None:
    if isinstance(filename_or_obj, (bytes, bytearray)):
        filename_or_obj = filename_or_obj.decode()

    elif isinstance(filename_or_obj, memoryview):
        filename_or_obj = filename_or_obj.tobytes().decode()

    try:
        filename = os.path.basename(filename_or_obj)
        base, ext = os.path.splitext(filename)

    except TypeError:
        return None

    if not ext:
        for f in TrajectoryFormat:
            if f.value in base.lower():
                return f.value
        return None
    else:
        ext = ext[1:].lower()

    if ext in TrajectoryFormat:
        return ext

    return None


def get_valid_trajectory_format(file_format: str | None) -> TrajectoryFormat:
    if file_format is None:
        raise InvalidTrajectoryFormatError(f"invalid file format: {file_format}")

    if file_format.lower() not in TrajectoryFormat:
        raise InvalidTrajectoryFormatError(f"invalid file format: {file_format}")

    return TrajectoryFormat(file_format.lower())
