from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Any, BinaryIO

import dask
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from xmdpy.core import (
    CellNDArray,
    FloatLike,
    NumAtoms,
    NumFrames,
    SingleDType,
    TrajNDArray,
)


@dataclass
class Trajectory:
    positions: TrajNDArray
    cell: CellNDArray | None = None
    atoms: np.ndarray[tuple[NumFrames, NumAtoms], np.dtype[str]] | None = None


TrajectoryParsingFn = Callable[
    [BinaryIO, NumAtoms, Iterable[int] | NumFrames | None, SingleDType], Trajectory
]


def get_trajectory_parsing_fn(file_format: str) -> TrajectoryParsingFn:
    if file_format not in TrajectoryFormat:
        raise AttributeError(
            f"Invalid file format: {file_format}. "
            f"Available formats: {TrajectoryFormat.available_formats}"
        )
    return TrajectoryFormat[file_format.upper()].parsing_fn


def parse_xyz_frames(
    file_handle: BinaryIO,
    n_atoms: NumAtoms,
    frames: Iterable[int] | NumFrames | None = None,
    dtype: SingleDType = np.float64,
) -> Trajectory:
    lines_per_frame = n_atoms + 2

    if isinstance(frames, int):
        frames = range(frames)

    if frames is None:

        class AllFrames:
            def __contains__(self, x: int) -> bool:
                return True

            def __len__(self) -> None:
                return None

        frames = AllFrames()
        n_rows = None
    else:
        n_rows = len(frames) * n_atoms

    def skiprow(row_id: int) -> bool:
        if int(row_id / lines_per_frame) not in frames:
            return True

        if row_id % lines_per_frame == 0 or (row_id - 1) % lines_per_frame == 0:
            return True

        return False

    df = pd.read_csv(
        file_handle,
        sep=r"\s+",
        usecols=[0, 1, 2, 3],
        nrows=n_rows,
        names=["symbol", "x", "y", "z"],
        dtype={"symbol": str, "x": dtype, "y": dtype, "z": dtype},
        skiprows=skiprow,
    )
    return Trajectory(
        atoms=df["symbol"].values.reshape(-1, n_atoms),
        positions=df[list("xyz")].values.reshape(-1, n_atoms, 3),
    )


def read_and_parse_xyz_frame_bytes(
    file_handle: BinaryIO,
    n_frames: int,
    n_atoms: int,
    dtype: SingleDType = np.float64,
) -> TrajNDArray:
    positions: TrajNDArray = np.zeros((n_frames, n_atoms, 3), dtype=dtype)

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
            all_atom_positions = read_and_parse_xyz_frame_bytes(
                file_handle=f,
                n_frames=n_frames,
                n_atoms=self.shape[1],
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


class TrajFormatEnumMeta(EnumMeta):
    def __contains__(cls, value: str) -> bool:
        return value.upper() in cls.__members__.keys()

    @property
    def available_formats(cls) -> tuple[str, ...]:
        return tuple(cls.__members__.keys())


class TrajectoryFormat(Enum, metaclass=TrajFormatEnumMeta):
    XYZ = ("xyz", parse_xyz_frames)

    def __new__(cls, base_file_type: str, parsing_fn: TrajectoryParsingFn):
        obj = object.__new__(cls)
        obj._value_ = base_file_type
        obj.parsing_fn = parsing_fn
        return obj

    def __repr__(self):
        return (
            f"<TrajectoryFormat.{self.name}: {self.value}>"
            f".parsing_fn={self.parsing_fn.__name__}"
        )


class TrajectoryBackendArray(xr.backends.BackendArray):
    def __init__(
        self,
        filename_or_obj: Any,
        shape: tuple,
        dtype,
        lock,
        parsing_fn: TrajectoryParsingFn,
    ):
        self.filename_or_obj = filename_or_obj
        self.shape = shape
        self.dtype = dtype
        self.lock = lock
        self.parsing_fn = parsing_fn

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

        frames = range(start, stop, step)

        with self.lock, open(self.filename_or_obj, "rb") as f:
            trajectory = self.parsing_fn(
                file_handle=f,
                n_atoms=self.shape[1],
                frames=frames,
                dtype=self.dtype,
            )

        # Get slices along other axes
        arr = trajectory.positions[:, atom_id, xyz_dim_id]

        if any([isinstance(dim, int) for dim in key]):
            arr = arr.squeeze()

        return arr


class XMDPYBackendEntrypoint(xr.backends.BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables: Any | None = None,
        dtype: SingleDType = np.float64,
        time: npt.ArrayLike[FloatLike] | None = None,
        cell: npt.ArrayLike[FloatLike] | None = None,
        file_format: str = "xyz",
    ) -> xr.Dataset:
        traj_parsing_fn = get_trajectory_parsing_fn(file_format)

        dtype = np.dtype(dtype)
        _, _, n_frames, atoms, _ = get_xyz_metadata(filename_or_obj)

        frame_var = xr.Variable(dims=("frame",), data=np.arange(n_frames, dtype=int))
        atoms_var = xr.Variable(dims=("atom_id",), data=atoms)
        atom_id_var = xr.Variable(
            dims=("atom_id",), data=np.arange(len(atoms), dtype=int)
        )

        backend_array = TrajectoryBackendArray(
            filename_or_obj=filename_or_obj,
            shape=(n_frames, len(atoms), 3),
            dtype=dtype,
            lock=dask.utils.SerializableLock(),
            parsing_fn=traj_parsing_fn,
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
