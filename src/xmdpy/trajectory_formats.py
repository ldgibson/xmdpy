from enum import StrEnum

from xmdpy.parsers import TrajectoryParsingFn, read_xyz_frames


class TrajectoryFormat(StrEnum):
    XYZ = "xyz"


TRAJECTORY_PARSING_FNS: dict[TrajectoryFormat, TrajectoryParsingFn] = {
    TrajectoryFormat.XYZ: read_xyz_frames,
}


def get_trajectory_parsing_fn(file_format: str) -> TrajectoryParsingFn:
    if file_format.lower() not in TrajectoryFormat:
        raise AttributeError(f"Invalid file format: {file_format}.")
    return TRAJECTORY_PARSING_FNS[TrajectoryFormat(file_format.lower())]
