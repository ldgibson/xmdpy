from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
import xarray.backends
import xarray.backends.locks
from xarray.core import indexing

from xmdpy.parsers import (
    get_trajectory_parsing_fn,
    get_xyz_dimensions,
    TrajectoryParsingFn,
)
from xmdpy.trajectory import Trajectory

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
        parsing_fn: TrajectoryParsingFn,
    ) -> None:
        self.filename_or_obj = filename_or_obj
        self.shape = shape
        self.dtype = dtype
        self.lock = lock
        self.parsing_fn = parsing_fn

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
        frame_id, atom_id, xyz_dim_id = key

        if isinstance(frame_id, np.ndarray):
            # TODO: add validation
            frames = frame_id
        else:
            if isinstance(frame_id, slice):
                start = frame_id.start or 0
                stop = frame_id.stop or self.shape[0]
                step = frame_id.step or 1
            else:
                start = frame_id or 0
                stop = frame_id + 1
                step = 1

            frames = range(start, stop, step)

        with self.lock, open(self.filename_or_obj, "rb") as f:
            trajectory: Trajectory = self.parsing_fn(
                file_handle=f,  # type: ignore
                n_atoms=self.shape[1],
                frames=frames,
                dtype=self.dtype,
            )

        # Get slices along other axes
        arr = trajectory.positions[:, atom_id, xyz_dim_id]

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
        time: npt.NDArray[FloatLike] | None = None,
        cell: npt.NDArray[FloatLike] | None = None,
        file_format: str = "xyz",
    ) -> xr.Dataset:
        traj_parsing_fn = get_trajectory_parsing_fn(file_format)

        dtype = np.dtype(dtype)
        frames, atoms = get_xyz_dimensions(filename_or_obj)

        frame_var = xr.Variable(dims=("frame",), data=frames)
        atoms_var = xr.Variable(dims=("atom_id",), data=atoms)
        atom_id_var = xr.Variable(
            dims=("atom_id",), data=np.arange(len(atoms), dtype=int)
        )

        backend_array = TrajectoryBackendArray(
            filename_or_obj=filename_or_obj,
            shape=(len(frames), len(atoms), 3),
            dtype=dtype,
            lock=xarray.backends.locks.SerializableLock(),
            parsing_fn=traj_parsing_fn,
        )

        xyz_data = indexing.LazilyIndexedArray(backend_array)
        xyz_var = xr.Variable(dims=("frame", "atom_id", "xyz_dim"), data=xyz_data)

        ds = xr.Dataset(
            data_vars={"xyz": xyz_var},
            coords={
                "frame": frame_var,
                "atom_id": atom_id_var,
                "atoms": atoms_var,
            },
        ).set_xindex("atoms")

        if time is not None:
            ds = ds.xmd.set_time_index(time)

        if cell is not None:
            ds = ds.xmd.set_cell(cell, n_frames=len(frames))

        return ds
