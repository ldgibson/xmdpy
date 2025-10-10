from __future__ import annotations

import os.path
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
)
from xmdpy.types import FloatLike, SingleDType

from .core import ON_DISK_TRAJECTORY, OnDiskArray, OuterIndex
from .names import Coord

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

    def _raw_indexing_method(
        self, key: tuple[OuterIndex, OuterIndex, OuterIndex]
    ) -> np.ndarray[tuple[int, ...], np.dtype[FloatLike]]:
        with self.lock:
            return self.array[key]


class XMDPYBackendEntrypoint(xarray.backends.BackendEntrypoint):
    description = (
        "Load molecular dynamics trajectories into Xarray. "
        f"Valid file formats are: {[f.value for f in TrajectoryFormat]}"
    )

    def guess_can_open(self, filename_or_obj) -> bool:
        try:
            filename = os.path.basename(filename_or_obj)  # type: ignore
            base, ext = os.path.splitext(filename)
        except TypeError:
            return False

        if ext is None:
            return any(base.lower() in f.value for f in TrajectoryFormat)

        return ext[1:] in TrajectoryFormat

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables: Any | None = None,
        dtype: SingleDType = np.float64,
        dt: int | float = 1,
        cell: npt.NDArray[FloatLike] | None = None,
        file_format: str = "guess",
    ) -> xr.Dataset:
        dtype = np.dtype(dtype)

        trajectory_format = get_valid_trajectory_format(file_format)

        traj = ON_DISK_TRAJECTORY[trajectory_format](filename_or_obj, dt)

        data_vars = {
            key: to_lazy_variable(*data_var)
            for key, data_var in traj.get_data_vars().items()
        }

        coords = {key: xr.Variable(*coord) for key, coord in traj.get_coords().items()}

        attrs = traj.get_attrs()

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs,
        ).set_xindex(Coord.ATOM)

        if cell is not None and "cell" not in ds:
            ds = ds.xmd.set_cell(cell)

        return ds
