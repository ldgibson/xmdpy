import dask
import dask.array as da
import numpy as np
import xarray as xr

from .analysis import pairwise_distances


@xr.register_dataset_accessor("xmd")
class TrajectoryAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def get_distances(self, atoms1, atoms2=None, vector=False, lazy=True):
        if 'cell' in self._obj:
            cell = self._obj.cell.data
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
                time_chunks = self._obj.xyz.chunks[0]
                chunks = (time_chunks, (len(sel1.atoms),), (len(sel2.atoms),))
                if vector:
                    chunks += ((3,),)
            else:
                chunks = 'auto'
            distances = da.map_blocks(
                pairwise_distances,
                xyz1.data,
                xyz2.data,
                cell,
                vector,
                dtype=float,
                chunks=chunks)

        else:
            distances = pairwise_distances(xyz1.values, xyz2.values, cell, vector)

        dims = ['time', 'atom_id1', 'atom_id2']
        if vector:
            dims.append('xyz_dim')

        return xr.DataArray(
            distances,
            coords={
                'time': self._obj.time.data,
                'atom_id1': ('atom_id1', sel1.atom_id.data),
                'atom_id2': ('atom_id2', sel2.atom_id.data),
                'atoms1': ('atom_id1', sel1.atoms.data),
                'atoms2': ('atom_id2', sel2.atoms.data),
            },
            dims=dims,
            name='distances',
        )
