import os
from typing import Any, BinaryIO

import dask
import numpy as np
import numpy.typing as npt
import xarray as xr

from .core import SingleDType


def read_and_parse_frame_bytes(
    file_handle: BinaryIO,
    n_frames: int,
    n_atoms: int,
    n_dim: int,
    dtype: SingleDType = np.float64,
) -> npt.ArrayLike:
    positions = np.zeros((n_frames, n_atoms, n_dim), dtype=dtype)

    for i in range(n_frames):
        # skip atom count and comment lines
        line = file_handle.readline()
        line = file_handle.readline()

        for j in range(n_atoms):
            line = file_handle.readline()
            fields = line.split()
            positions[i, j] = fields[1:]
    return positions


def get_xyz_metadata(filename: BinaryIO):
    total_size = os.path.getsize(filename)
    atoms = []
    cell = None
    frame_size = 0
    with open(filename, "rb") as f:
        line = f.readline()
        frame_size += len(line)
        n_atoms = int(line.strip())

        line = f.readline()
        frame_size += len(line)

        for _ in range(n_atoms):
            line = f.readline()
            frame_size += len(line)
            fields = line.split()
            atoms.append(fields[0].decode())

    if total_size % frame_size == 0:
        n_frames = int(total_size / frame_size)
    else:
        n_frames = int((total_size + 1) / frame_size)

    return total_size, frame_size, n_frames, atoms, cell


class XYZBackendArray(xr.backends.BackendArray):
    def __init__(
        self,
        filename_or_obj: Any,
        shape: tuple,
        dtype,
        lock,
        frame_size: int,
    ):
        self.filename_or_obj = filename_or_obj
        self.shape = shape
        self.dtype = dtype
        self.lock = lock
        self.frame_size = frame_size

    def __getitem__(self, key: tuple):
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.OUTER,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        frame_id, atom_id, xyz_dim_id = key

        if isinstance(frame_id, slice):
            start = frame_id.start or 0
            stop = frame_id.stop or self.shape[0]
            step = frame_id.step or 1
        else:
            start = frame_id or 0
            stop = frame_id + 1
            step = 1

        offset = self.frame_size * start
        n_frames = stop - start

        with self.lock, open(self.filename_or_obj, "rb") as f:
            f.seek(offset)
            all_atom_positions = read_and_parse_frame_bytes(
                file_handle=f,
                n_frames=n_frames,
                n_atoms=self.shape[1],
                n_dim=self.shape[2],
                dtype=self.dtype,
            )

        sliced_frame_id = slice(None, None, step)
        arr = all_atom_positions[sliced_frame_id, atom_id, xyz_dim_id]

        if any([isinstance(dim, int) for dim in key]):
            arr = arr.squeeze()

        return arr


class XYZBackendEntrypoint(xr.backends.BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables: Any | None = None,
        dtype: SingleDType = np.float64,
        time: npt.ArrayLike | None = None,
        cell: npt.ArrayLike | None = None,
    ) -> xr.Dataset:
        dtype = np.dtype(dtype)
        _, frame_size, n_frames, atoms, _ = get_xyz_metadata(filename_or_obj)

        frame_var = xr.Variable(dims=("frame",), data=np.arange(n_frames, dtype=int))
        atoms_var = xr.Variable(dims=("atom_id",), data=atoms)
        atom_id_var = xr.Variable(
            dims=("atom_id",), data=np.arange(len(atoms), dtype=int)
        )

        backend_array = XYZBackendArray(
            filename_or_obj=filename_or_obj,
            shape=(n_frames, len(atoms), 3),
            dtype=dtype,
            lock=dask.utils.SerializableLock(),
            frame_size=frame_size,
        )

        xyz_data = xr.core.indexing.LazilyIndexedArray(backend_array)
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
            ds = ds.xmd.set_cell(cell, n_frames=n_frames)

        return ds
