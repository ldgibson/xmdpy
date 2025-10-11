from xmdpy.parsers.trajectory_formats import TrajectoryFormat

from .on_disk_trajectory import OnDiskTrajectory
from .xyz import OnDiskXYZTrajectory

ON_DISK_TRAJECTORY: dict[TrajectoryFormat, type[OnDiskTrajectory]] = {
    TrajectoryFormat.XYZ: OnDiskXYZTrajectory,
}
