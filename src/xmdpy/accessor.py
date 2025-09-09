from typing import Sequence

import dask
import dask.array as da
import numpy.typing as npt
import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

from xmdpy.analysis import compute_pairwise_distances
from xmdpy.cell import Cell


class CellNotDefinedError(Exception):
    def __init__(self, message: str | None = None) -> None:
        self.message = message or (
            "Variable `'cell'` is not defined. "
            "Use the `set_cell(...)` method in the "
            "accessor to define cell parameters."
        )
        super().__init__(self.message)


def _has_trajectory_like_objects(obj: Dataset | DataArray) -> bool:
    if "xyz" in obj:
        if obj["xyz"].ndim == 3 and obj["xyz"].shape[-1] == 3:
            return True
    return False


@xr.register_dataset_accessor("xmd")
class TrajectoryAccessor:
    # def __new__(cls, obj) -> Self:
    #     if not _has_trajectory_like_objects(obj):
    #         raise AttributeError(
    #             "TrajectoryAccessor only usable for Datasets with trajectory-like variables."
    #         )
    #     return super().__new__(cls)

    def __init__(self, xarray_obj: Dataset) -> None:
        self._obj = xarray_obj

        if "cell" in self._obj:
            self.cell = Cell(self._obj["cell"])

    def set_cell(self, cell: npt.ArrayLike, n_frames: int | None = None) -> Dataset:
        """Adds or updates the cell variable in the Dataset.

        If the trajectory has constant volume (i.e., only one set of cell
        vectors), then the cell variable will be broadcasted across all
        frames in the trajectory.

        Args:
            cell (npt.ArrayLike): Cell parameters.
            n_frames (int | None, optional): Option to explicitly set the
            number of frames that are associated with the `cell` argument.
            Defaults to None.

        Returns:
            xr.Dataset: New Dataset with the cell variable added.
        """
        return self._obj.assign({"cell": Cell(cell, n_frames=n_frames).to_xarray()})

    def set_time_index(self, time: npt.ArrayLike, indexable: bool = True) -> Dataset:
        """Adds or updates the time index in the Dataset.

        Args:
            time (npt.ArrayLike): Values of the time index
            indexable (bool, optional): Allows the new coordinate to be
            indexable. Defaults to True.

        Returns:
            xr.Dataset: New Dataset with the time index added.
        """
        time_var = xr.Variable(dims=("frame",), data=time)
        ds = self._obj.assign_coords(dict(time=time_var))

        if indexable:
            ds = ds.set_xindex("time")

        return ds

    def add_group(self, selection, name, indexable=True) -> Dataset:
        # TODO
        raise NotImplementedError()

    def get_distances(
        self,
        atoms1: str | int | Sequence[int],
        atoms2: str | int | Sequence[int] | None = None,
        mic: bool = True,
        lazy: bool = True,
    ) -> DataArray:
        """Compute the pairwise distances between pairs of atoms.

        Args:
            atoms1 (str | Sequence[int]): Selection of atoms - can be either a `str`,
            where all atoms matching that name are selected; or one or more `atom_id`s.
            If `atoms2=None`, then all unique pairs from `atoms1` selection are used.
            atoms2 (str | Sequence[int] | None, optional): Second selection of atoms.
            Follows the same rules as `atoms1`. Defaults to None.
            mic (bool, optional): Use minimum image convention. Requires cell parameters
            to be defined. Defaults to True.
            lazy (bool, optional): Return a lazy view of the distances using Dask.
            Defaults to True.

        Raises:
            CellNotDefinedError: If using `mic=True` without the Dataset containing
            a `cell` variable.

        Returns:
            xr.DataArray: Pairwise distances with coordinates corresponding to
            `atoms1` and `atoms2`. If `atoms2=None`, then the second selection
            will just match the selection from `atoms1`.
        """
        if mic:
            if "cell" not in self._obj:
                raise CellNotDefinedError()
            cell = self.cell
        else:
            cell = None

        if isinstance(atoms1, str):
            sel1 = self._obj.sel(atoms=atoms1)
        else:
            sel1 = self._obj.sel(atom_id=atoms1)
        xyz1 = sel1.xyz

        if atoms2:
            if isinstance(atoms2, str):
                sel2 = self._obj.sel(atoms=atoms2)
            else:
                sel2 = self._obj.sel(atom_id=atoms2)
        else:
            sel2 = sel1
        xyz2 = sel2.xyz

        if lazy:
            if dask.is_dask_collection(self._obj.xyz.data):
                frame_chunks = self._obj.xyz.chunks[0]
                chunks = (frame_chunks, (len(sel1.atoms),), (len(sel2.atoms),))
            else:
                chunks = "auto"
            distances = da.map_blocks(
                compute_pairwise_distances,
                xyz1.data,
                xyz2.data,
                cell,
                dtype=float,
                chunks=chunks,
            )
        else:
            distances = compute_pairwise_distances(xyz1.values, xyz2.values, cell)

        return xr.DataArray(
            name="distances",
            data=distances,
            coords={
                "frame": self._obj.frame.data,
                "atom_id1": ("atom_id1", sel1.atom_id.data),
                "atom_id2": ("atom_id2", sel2.atom_id.data),
                "atoms1": ("atom_id1", sel1.atoms.data),
                "atoms2": ("atom_id2", sel2.atoms.data),
            },
            dims=["frame", "atom_id1", "atom_id2"],
        )
