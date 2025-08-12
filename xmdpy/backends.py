import os

import dask
import numpy as np
import xarray as xr


def read_and_parse_frame_bytes(f, n_frames, n_atoms, n_dim):
    positions = np.zeros((n_frames, n_atoms, n_dim), dtype=float)
    
    # with io.BytesIO(frame_bytes) as f:
    for i in range(n_frames):
        # skip atom count and comment lines
        line = f.readline()
        line = f.readline()
        
        for j in range(n_atoms):
            line = f.readline()
            fields = line.split()
            positions[i, j] = fields[1:]
    return positions


def get_xyz_metadata(filename):
    total_size = os.path.getsize(filename)
    atoms = []
    cell = None
    frame_size = 0
    with open(filename, 'rb') as f:
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
        filename_or_obj,
        shape,
        dtype,
        lock,
        frame_size,
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
        size = self.frame_size
        frame_id, atom_id, xyz_dim_id = key
        if isinstance(frame_id, slice):
            start = frame_id.start or 0
            stop = frame_id.stop or self.shape[0]
            step = frame_id.step or 1
            offset = size * start
            count = stop - start
        else:
            start = frame_id
            stop = frame_id + 1
            step = 1
            offset = size * frame_id
            count = 1

        length = count * size
        with self.lock, open(self.filename_or_obj, 'rb') as f:
            f.seek(offset)
            all_atom_positions = read_and_parse_frame_bytes(f, count, self.shape[1], self.shape[2])
        sliced_frame_id = slice(None, None, step)
        arr = all_atom_positions[sliced_frame_id, atom_id, xyz_dim_id]

        if isinstance(frame_id, int) or isinstance(atom_id, int) or isinstance(xyz_dim_id, int):
            arr = arr.squeeze()

        return arr


class XYZBackendEntrypoint(xr.backends.BackendEntrypoint):
    def open_dataset(self, filename_or_obj, *, drop_variables=None, dtype=np.float64, time=None):
        dtype = np.dtype(dtype)
        total_size, frame_size, n_frames, atoms, cell = get_xyz_metadata(filename_or_obj)

        atoms_var = xr.Variable(dims=("atom_id",), data=atoms)
        atom_id_var = xr.Variable(dims=("atom_id",), data=np.arange(len(atoms)))
        if not time:
            time = np.arange(n_frames, dtype=dtype)
        time_var = xr.Variable(dims=("time",), data=time)

        backend_array = XYZBackendArray(
            filename_or_obj=filename_or_obj,
            shape=(n_frames, len(atoms), 3),
            dtype=dtype,
            lock=dask.utils.SerializableLock(),
            frame_size=frame_size,
        )

        data = xr.core.indexing.LazilyIndexedArray(backend_array)
        xyz_var = xr.Variable(dims=("time", "atom_id", "xyz_dim"), data=data)

        ds = xr.Dataset(
            data_vars={"xyz": xyz_var},
            coords={
                "time": time_var,
                "atom_id": atom_id_var,
                "atoms": atoms_var,
            },
        ).set_xindex('atoms')
        return ds
