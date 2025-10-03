# xmdpy
[![CI](https://github.com/ldgibson/xmdpy/actions/workflows/ci.yml/badge.svg)](https://github.com/ldgibson/xmdpy/actions/workflows/ci.yml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**xmdpy** is an open source package that provides an xarray interface for molecular dynamics trajectory files to facilitate fast and easy analysis of large trajectories.

### Key Features
* Lazy loading trajectories
* Common analysis tools provided via the `.xmd` accessor on xarray objects
* Parallel and memory-sensitive data manipulation with `dask`
* Powerful slicing/indexing of N-dimensional datasets 

### Available Trajectory Formats

| Format | Description |
| --- | --- |
| `xyz` | Basic XYZ files; ignores information on comment line and anything beyond the 4th column |
| `xdatcar` | VASP XDATCAR trajectory files; compatible with both Direct and Cartesian coordinates; not compatible with trajectories with variable cell parameters (e.g., NPT simulations) |

> [!NOTE]
> Other common formats to be implemented in the future include: `extxyz`, `netcdf`, `hdf5`, `lammpstrj`, `traj` (binary ASE format), `xtc`, `dcd`

## Dependencies
* [Python](https://python.org) 3.12+
* [xarray](https://xarray.dev) 2025.9+
* [NumPy](https://numpy.org) 2.3.2+
* [Dask](https://dask.org) 2025.7+

## Installation
* From source:
```bash
git clone https://github.com/ldgibson/xmdpy.git
cd xmdpy
pip install .
```

> [!WARNING]
> This package is in early stages of development and is subject to change without backward compatibility.

## Examples
### Lazy Loading XYZ Files

Standard XYZ files do not contain information about the cell parameters, so XYZ files that are loaded do not include a `cell` data variable.

```python
>>> import xarray as xr
>>> import xmdpy
>>> traj = xr.open_dataset("trajectory.xyz", engine="xmdpy")
>>> print(traj)
<xarray.Dataset> Size: 5MB
Dimensions:      (time: 1000, atom_id: 200, xyz_dim: 3)
Coordinates:
  * time         (time) int64 8kB 0 1 2 3 4 5 6 ... 993 994 995 996 997 998 999
  * atom_id      (atom_id) int64 2kB 0 1 2 3 4 5 6 ... 194 195 196 197 198 199
  * atoms        (atom_id) <U2 2kB 'Li' 'Li' 'Li' 'Li' ... 'Cl' 'Cl' 'Cl' 'Cl'
  * xyz_dim      (xyz_dim) <U1 12B 'x' 'y' 'z'
Data variables:
    xyz          (time, atom_id, xyz_dim) float64 5MB ...
```

Cell parameters can be easily specified via the `cell` keyword argument, which accepts a scalar or various array shapes and nested list structures:
* Single cell length: `cell=L`
* Separate lengths of a single cell: `cell=[Lx, Ly, Lz]`
* Cell vectors for each dimension: `cell=[[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]`
* Variable cell vectors per frame:
```python
cell=[
    [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]],
    [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]],
    ...,
    [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]],
]
```
When specified, xmdpy will attempt to reshape the input to represent cell vectors at every frame. If parameters for only a single cell are detected, the cell vectors will be broadcasted along the `time` index of the trajectory to minimize the memory footprint.

```python
>>> traj = xr.open_dataset("trajectory.xyz", engine="xmdpy", cell=20.12)
>>> print(traj)
>>> traj
<xarray.Dataset> Size: 5MB
Dimensions:      (time: 1000, atom_id: 200, xyz_dim: 3, cell_vector: 3)
Coordinates:
  * time         (time) int64 8kB 0 1 2 3 4 5 6 ... 993 994 995 996 997 998 999
  * atom_id      (atom_id) int64 2kB 0 1 2 3 4 5 6 ... 194 195 196 197 198 199
  * atoms        (atom_id) <U2 2kB 'Li' 'Li' 'Li' 'Li' ... 'Cl' 'Cl' 'Cl' 'Cl'
  * xyz_dim      (xyz_dim) <U1 12B 'x' 'y' 'z'
  * cell_vector  (cell_vector) <U1 12B 'A' 'B' 'C'
Data variables:
    xyz          (time, atom_id, xyz_dim) float64 5MB ...
    cell         (time, cell_vector, xyz_dim) float64 72kB ...
```

### Xarray Accessor: `.xmd`

**Lazily compute all pairwise distances betwen two atom types.**

First, load the trajectory and specify the chunk size along the `time` index, which loads the trajectory data as Dask arrays. 
```python
>>> traj = xr.open_dataset(
    "trajectory.xyz",
    engine="xmdpy",
    cell=20.12,
    chunks={"time": 100},
    )
>>> print(traj)
<xarray.Dataset> Size: 5MB
Dimensions:      (time: 1000, atom_id: 200, xyz_dim: 3, cell_vector: 3)
Coordinates:
  * time         (time) int64 8kB 0 1 2 3 4 5 6 ... 993 994 995 996 997 998 999
  * atom_id      (atom_id) int64 2kB 0 1 2 3 4 5 6 ... 194 195 196 197 198 199
  * atoms        (atom_id) <U2 2kB 'Li' 'Li' 'Li' 'Li' ... 'Cl' 'Cl' 'Cl' 'Cl'
  * xyz_dim      (xyz_dim) <U1 12B 'x' 'y' 'z'
  * cell_vector  (cell_vector) <U1 12B 'A' 'B' 'C'
Data variables:
    xyz          (time, atom_id, xyz_dim) float64 5MB dask.array<chunksize=(100, 200, 3), meta=np.ndarray>
    cell         (time, cell_vector, xyz_dim) float64 72kB dask.array<chunksize=(100, 3, 3), meta=np.ndarray>
```

To compute the distances lazily, simply use the `.xmd.get_distances()` accessor method. The result will immediately return an xarray object with the distances stored in a Dask array.

```python
>>> distances = traj.xmd.get_distances('Li', 'Cl')
>>> print(distances)
<xarray.DataArray (time: 1000, atom_id1: 100, atom_id2: 100)> Size: 80MB
dask.array<transpose, shape=(1000, 100, 100), dtype=float64, chunksize=(100, 100, 100), chunktype=numpy.ndarray>
Coordinates:
  * time      (time) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
  * atom_id1  (atom_id1) int64 800B 0 1 2 3 4 5 6 7 ... 92 93 94 95 96 97 98 99
  * atoms1    (atom_id1) <U2 800B 'Li' 'Li' 'Li' 'Li' ... 'Li' 'Li' 'Li' 'Li'
  * atom_id2  (atom_id2) int64 800B 100 101 102 103 104 ... 195 196 197 198 199
  * atoms2    (atom_id2) <U2 800B 'Cl' 'Cl' 'Cl' 'Cl' ... 'Cl' 'Cl' 'Cl' 'Cl'
```