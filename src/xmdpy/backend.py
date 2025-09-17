from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
import xarray.backends
import xarray.backends.locks
import xarray.core.indexing

from xmdpy.parsers.core import (
    TRAJECTORY_DETAILS_GETTERS,
    TRAJECTORY_PARSERS,
    TrajectoryParser,
)
from xmdpy.parsers.trajectory_formats import get_valid_format
from xmdpy.types import FloatLike, SingleDType

type OuterIndex = (
    int | np.integer | slice | np.ndarray[tuple[int], np.dtype[np.integer]]
)


class TrajectoryBackendArray(xarray.backends.BackendArray):
    def __init__(
        self,
        filename_or_obj: os.PathLike,
        shape: tuple[int, ...],
        dtype: SingleDType,
        lock: xarray.backends.locks.SerializableLock,
        parser: TrajectoryParser,
    ) -> None:
        self.filename_or_obj = filename_or_obj
        self.shape = shape
        self.dtype = dtype
        self.lock = lock
        self.parser = parser

    def __getitem__(self, key: Any):
        return xarray.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xarray.core.indexing.IndexingSupport.OUTER,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(
        self, key: tuple[OuterIndex, OuterIndex, OuterIndex]
    ) -> np.ndarray[tuple[int, ...], np.dtype[FloatLike]]:
        frame_id, atom_id, xyz_dim_id = key

        if isinstance(frame_id, np.ndarray):
            frames = frame_id

        else:
            if isinstance(frame_id, slice):
                start = frame_id.start or 0
                stop = frame_id.stop or self.shape[0]
                step = frame_id.step or 1

            elif isinstance(frame_id, int):
                start = frame_id or 0
                stop = frame_id + 1
                step = 1
            else:
                start, stop, step = 0, self.shape[0], 1

            frames = range(start, stop, step)

        if np.max(frames) >= self.shape[0]:
            raise IndexError("time index out of range")

        with self.lock, open(self.filename_or_obj, "rb") as f:
            positions = self.parser(
                file_handle=f,  # type: ignore
                n_atoms=self.shape[1],
                frames=frames,
                dtype=self.dtype,
            )

        # Get slices along other axes
        arr = positions[:, atom_id, xyz_dim_id]

        if any(isinstance(dim, int) for dim in key):
            arr = arr.squeeze()

        return arr


class XMDPYBackendEntrypoint(xarray.backends.BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables: Any | None = None,
        dtype: SingleDType = np.float64,
        dt: int | float = 1,
        cell: npt.NDArray[FloatLike] | None = None,
        file_format: str = "xyz",
    ) -> xr.Dataset:
        dtype = np.dtype(dtype)

        valid_format = get_valid_format(file_format)

        n_frames, atoms, cell_from_file, variable_cell = TRAJECTORY_DETAILS_GETTERS[
            valid_format
        ](filename_or_obj)

        if variable_cell:
            valid_format = get_valid_format(file_format + "npt")

        traj_parser = TRAJECTORY_PARSERS[valid_format]

        if cell is not None and cell_from_file is not None:
            raise ValueError(
                "cell information provided as argument and found in trajectory file"
            )
        elif cell is None:
            cell = cell_from_file

        time_var = xr.Variable(
            dims=("time",), data=np.arange(0, n_frames * dt, dt, dtype=type(dt))
        )
        atoms_var = xr.Variable(dims=("atom_id",), data=atoms)
        atom_id_var = xr.Variable(dims=("atom_id",), data=range(len(atoms)))

        backend_array = TrajectoryBackendArray(
            filename_or_obj=filename_or_obj,
            shape=(n_frames, len(atoms), 3),
            dtype=dtype,
            lock=xarray.backends.locks.SerializableLock(),
            parser=traj_parser,
        )

        xyz_data = xarray.core.indexing.LazilyIndexedArray(backend_array)
        xyz_var = xr.Variable(dims=("time", "atom_id", "xyz_dim"), data=xyz_data)

        ds = xr.Dataset(
            data_vars={"xyz": xyz_var},
            coords={
                "time": time_var,
                "atom_id": atom_id_var,
                "atoms": atoms_var,
            },
        ).set_xindex("atoms")

        if cell is not None:
            ds = ds.xmd.set_cell(cell)

        return ds
