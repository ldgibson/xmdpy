from typing import Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

from xmdpy.analysis import compute_distance_vectors
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
            self.cell = Cell._from_normalized(self._obj["cell"].data)

    def set_cell(self, cell: npt.ArrayLike) -> Dataset:
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
        return self._obj.assign(
            {"cell": Cell(cell).to_xarray(time_index=self._obj.time)}
        )

    def add_group(self, selection, name, indexable=True) -> Dataset:
        # TODO
        raise NotImplementedError()

    def atom_sel(self, atoms: str | int | Sequence[int] | slice) -> Dataset:
        """Returns a dataset for the given atom selection.

        Parameters
        ----------
        atoms : str | int | Sequence[int] | slice
            Index for selecting atoms

        Returns
        -------
        Dataset
            Atom selection
        """
        if isinstance(atoms, str):
            return self._obj.where(self._obj.atoms == atoms, drop=True)
        elif isinstance(atoms, int):
            return self._obj.where(self._obj.atom_id == atoms, drop=True)
        else:
            return self._obj.sel(atom_id=atoms)

    def get_atom_selections(
        self, *args: str | int | Sequence[int] | slice, update_coord_names: bool = True
    ) -> tuple[Dataset, ...]:
        """Returns a dataset for each selection provided.

        Parameters
        ----------
        update_coord_names : bool, optional
            If True, appends 1, 2, ... to 'atom_id' and 'atoms' coordinates
            based on the order the selections were provided, by default True

        Returns
        -------
        tuple[Dataset, ...]
            Datasets of atom selections
        """
        atom_selections = []
        n_args = len(args)

        for i, atoms in enumerate(args, start=1):
            atom_sel_ds = self.atom_sel(atoms)

            if update_coord_names and n_args > 1:
                atom_sel_ds = atom_sel_ds.rename(
                    {"atom_id": f"atom_id{i}", "atoms": f"atoms{i}"}
                )

            atom_selections.append(atom_sel_ds)

        return tuple(atom_selections)

    def get_distances(
        self,
        atoms1: str | int | Sequence[int],
        atoms2: str | int | Sequence[int] | None = None,
        mic: bool = True,
        vector: bool = False,
    ) -> DataArray:
        """Compute the pairwise distances between selections of atoms.

        Parameters
        ----------
        atoms1 : str | Sequence[int]
            Selection of atoms - can be either a `str`, where all atoms
            matching that name are selected; or one or more `atom_id`s.
            If `atoms2=None`, then all unique pairs from `atoms1` selection
            are used.
        atoms2 : str | Sequence[int] | None, optional
            Second selection of atoms. Follows the same rules as `atoms1`,
            by default None.
        mic : bool, optional
            Use minimum image convention. Requires cell parameters to be
            defined, by default True
        vector : bool, optional
            If True, returns distances as vectors, otherwise distances
            returned as magnitudes, by default False

        Returns
        -------
        xr.DataArray
            Pairwise distances

        Raises
        ------
        CellNotDefinedError
            If using `mic=True` without the Dataset containing a cell variable

        """
        core_dims = [
            ["atom_id1", "xyz_dim"],
            ["atom_id2", "xyz_dim"],
        ]

        if mic:
            if "cell" not in self._obj:
                raise CellNotDefinedError()
            cell_lengths = self.cell.lengths
            core_dims.append(["xyz_dim"])
        else:
            cell_lengths = None
            core_dims.append([])

        if atoms2 is None:
            atoms2 = atoms1

        atom_sel1, atom_sel2 = self.get_atom_selections(
            atoms1, atoms2, update_coord_names=True
        )

        distances: DataArray = xr.apply_ufunc(
            compute_distance_vectors,
            atom_sel1.xyz,
            atom_sel2.xyz,
            cell_lengths,
            input_core_dims=core_dims,
            output_core_dims=[["atom_id1", "atom_id2", "xyz_dim"]],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs=dict(
                output_sizes={
                    "atom_id1": atom_sel1.sizes["atom_id1"],
                    "atom_id2": atom_sel2.sizes["atom_id2"],
                    "xyz_dim": 3,
                },
            ),
            output_dtypes=[
                "float64",
            ],
        ).transpose("time", "atom_id1", "atom_id2", "xyz_dim")

        if not vector:
            distances = xr.apply_ufunc(
                np.linalg.norm,
                distances,
                kwargs={"axis": -1},
                input_core_dims=[["xyz_dim"]],
                dask="parallelized",
            )

        return distances
