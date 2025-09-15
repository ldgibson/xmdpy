from enum import StrEnum


class TrajectoryFormat(StrEnum):
    XYZ = "xyz"
    XDATCAR = "xdatcar"


class InvalidTrajectoryFormatError(NotImplementedError):
    pass


def get_valid_format(file_format: str) -> TrajectoryFormat:
    if file_format.lower() not in TrajectoryFormat:
        raise InvalidTrajectoryFormatError(f"invalid file format: {file_format}")
    return TrajectoryFormat(file_format.lower())
