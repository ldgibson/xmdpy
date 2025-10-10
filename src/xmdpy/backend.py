from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

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
from xmdpy.parsers.trajectory_formats import (
    TrajectoryFormat,
    get_valid_trajectory_format,
)
from xmdpy.types import (
    CellArray,
    CellArray3x3,
    FloatLike,
    IntArray,
    PathLike,
    SingleDType,
    TrajArray,
)

type OuterIndex = (
    int | np.integer | slice | np.ndarray[tuple[int], np.dtype[np.integer]]
)


def normalize_index_to_array(
    index: OuterIndex | None, upper_bound: int
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    if isinstance(index, np.ndarray):
        if index.ndim > 1:
            raise IndexError("index cannot be multidimensional")
        if np.any(index >= upper_bound):
            raise IndexError("index extends beyond upper bound")
        return np.asarray(index, dtype=np.int64)

    if isinstance(index, (int, np.integer)):
        if index >= upper_bound:
            raise IndexError("index extends beyond upper bound")
        return np.array([index])

    if index is None:
        start, stop, step = 0, upper_bound, 1

    elif isinstance(index, slice):
        start = index.start or 0
        stop = index.stop or upper_bound
        step = index.step or 1

    else:
        raise IndexError(f"invalid index type: {type(index)}")

    if start < 0:
        start = upper_bound + start
    if stop < 0:
        stop = upper_bound + stop

    if start >= upper_bound or stop > upper_bound:
        raise IndexError("index extends beyond upper bound")

    return np.arange(start, stop, step, dtype=np.int64)


@dataclass
class OnDiskTraj:
    filename: PathLike
    trajectory_format: TrajectoryFormat
    dtype: SingleDType = "float64"

    atoms: list[str] = field(init=False)
    n_atoms: int = field(init=False)
    n_frames: int = field(init=False)
    cell: CellArray | CellArray3x3 | None = field(init=False)
    parser_fn: TrajectoryParser = field(init=False)

    _variable_cell: bool = field(init=False)

    def __post_init__(self) -> None:
        self.n_frames, atoms, cell_from_file, variable_cell = (
            TRAJECTORY_DETAILS_GETTERS[self.trajectory_format](self.filename)
        )
        self.atoms = atoms
        self.n_atoms = len(self.atoms)
        self.cell = cell_from_file
        self._variable_cell = variable_cell

        self.parser_fn = TRAJECTORY_PARSERS[self.trajectory_format]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_frames, self.n_atoms, 3)

    def get_coordinate_vars(self) -> list[tuple[str, Any]]:
        return [
            ("atom_id", np.arange(self.n_atoms)),
            ("atom_id", self.atoms),
            ("time", np.arange(self.n_frames)),
        ]

    def __getitem__(self, key: tuple[OuterIndex, OuterIndex, OuterIndex]) -> TrajArray:
        indices: tuple[IntArray, IntArray, IntArray] = tuple(
            normalize_index_to_array(index, self.shape[i])
            for i, index in enumerate(key)
        )  # type: ignore

        with open(self.filename, "rb") as f:
            arr = self.parser_fn(f, indices, self.n_atoms, self.dtype)

        if any(isinstance(dim, int) for dim in key):
            arr = np.squeeze(arr)

        return arr


# class OnDiskCell:
#     filename: PathLike
#     trajectory_format: TrajectoryFormat
#     n_atoms: int
#     n_frames: int
#     dtype: SingleDType = "float64"
#     shape: tuple[int, Literal[3], Literal[3]] = field(init=False)
#     parser_fn: TrajectoryParser = field(init=False)

#     def __post_init__(self) -> None:
#         self.shape = (self.n_frames, 3, 3)
#         self.ndim = 3
#         self.parser_fn = TRAJECTORY_PARSERS[self.trajectory_format]

#     def __getitem__(self, key: tuple[OuterIndex, OuterIndex, OuterIndex]) -> CellArray:
#         # frame_index, atom_index, xyz_dim_index = key
#         # frame_index = normalize_index_to_array(frame_index, upper_bound=self.n_frames)
#         indices = tuple(
#             normalize_index_to_array(index, self.shape[i])
#             for i, index in enumerate(key)
#         )  # type: ignore

#         with open(self.filename, "rb") as f:
#             return self.parser_fn(f, indices, self.n_atoms, self.dtype)


@dataclass
class OnDiskCoordinates:
    filename: PathLike
    trajectory_format: TrajectoryFormat
    n_frames: int
    n_atoms: int
    dtype: SingleDType = "float64"
    parser_fn: TrajectoryParser = field(init=False)
    shape: tuple[int, int, Literal[3]] = field(init=False)

    def __post_init__(self) -> None:
        self.shape = (self.n_frames, self.n_atoms, 3)
        self.parser_fn = TRAJECTORY_PARSERS[self.trajectory_format]

    def __getitem__(self, key: tuple[OuterIndex, OuterIndex, OuterIndex]) -> TrajArray:
        indices: tuple[IntArray, IntArray, IntArray] = tuple(
            normalize_index_to_array(index, self.shape[i])
            for i, index in enumerate(key)
        )  # type: ignore

        with open(self.filename, "rb") as f:
            arr = self.parser_fn(f, indices, self.n_atoms, self.dtype)

        if any(isinstance(dim, int) for dim in key):
            arr = np.squeeze(arr)

        return arr


class TrajectoryBackendArray(xarray.backends.BackendArray):
    def __init__(
        self,
        array: OnDiskCoordinates,
        shape: tuple[int, ...],
        dtype: SingleDType,
        lock: xarray.backends.locks.SerializableLock,
    ) -> None:
        self.array = array
        self.shape = shape
        self.dtype = dtype
        self.lock = lock

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

        n_frames, atoms, cell_from_file, variable_cell = TRAJECTORY_DETAILS_GETTERS[
            trajectory_format
        ](filename_or_obj)

        # traj_parser = TRAJECTORY_PARSERS[valid_format]

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

        coords = OnDiskCoordinates(
            filename_or_obj, trajectory_format, n_frames, len(atoms)
        )

        backend_array = TrajectoryBackendArray(
            array=coords,
            shape=(n_frames, len(atoms), 3),
            dtype=dtype,
            lock=xarray.backends.locks.SerializableLock(),
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

        if cell is not None and "cell" not in ds:
            ds = ds.xmd.set_cell(cell)

        return ds
