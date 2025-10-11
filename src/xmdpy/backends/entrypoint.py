from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
import xarray.backends
import xarray.backends.locks
from xarray.core import indexing

from xmdpy.parsers.trajectory_formats import (
    TrajectoryFormat,
    get_valid_trajectory_format,
    guess_trajectory_format_str,
)
from xmdpy.types import FloatLike, SingleDType, TrajNDArray

from .core import ON_DISK_TRAJECTORY
from .names import Coord
from .on_disk_trajectory import OnDiskArray, OuterIndex

LOCK = xarray.backends.locks.SerializableLock()


def to_lazy_variable(dims: tuple[str, ...], arr: OnDiskArray) -> xr.Variable:
    return xr.Variable(
        dims,
        indexing.LazilyIndexedArray(
            TrajectoryBackendArray(arr, arr.shape, arr.dtype, LOCK)
        ),
    )


class TrajectoryBackendArray(xarray.backends.BackendArray):
    def __init__(
        self,
        array: OnDiskArray,
        shape: tuple[int, ...],
        dtype: SingleDType,
        lock: xarray.backends.locks.SerializableLock,
    ) -> None:
        self.array = array
        self.shape = shape
        self.dtype = dtype
        self.lock = lock

    def __getitem__(self, key: Any):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple[OuterIndex, ...]) -> TrajNDArray:
        with self.lock:
            return self.array[key]


class XMDPYBackendEntrypoint(xarray.backends.BackendEntrypoint):
    description = (
        "Load molecular dynamics trajectories into Xarray. "
        f"Valid file formats are: {[f.value for f in TrajectoryFormat]}"
    )

    def guess_can_open(self, filename_or_obj) -> bool:
        file_format = guess_trajectory_format_str(filename_or_obj)  # type: ignore

        if file_format is None:
            return False

        return True

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables: Any | None = None,
        dtype: SingleDType = "float64",
        dt: int | float = 1,
        cell: npt.NDArray[FloatLike] | None = None,
        file_format: str | None = None,
    ) -> xr.Dataset:
        dtype = np.dtype(dtype)
        if file_format is None:
            file_format = guess_trajectory_format_str(filename_or_obj)

        trajectory_format = get_valid_trajectory_format(file_format)

        traj = ON_DISK_TRAJECTORY[trajectory_format](filename_or_obj, dt, dtype=dtype)

        data_vars = {
            key: to_lazy_variable(dims, on_disk_array)
            for key, (dims, on_disk_array) in traj.get_data_vars().items()
        }

        coords = {
            key: xr.Variable(dims, coord)
            for key, (dims, coord) in traj.get_coords().items()
        }

        attrs = traj.get_attrs()

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs,
        ).set_xindex(Coord.ATOM)

        if cell is not None and "cell" not in ds:
            ds = ds.xmd.set_cell(cell)

        return ds
