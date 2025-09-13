import numpy as np
import xarray as xr
from xarray.testing import assert_equal

import pytest


@pytest.mark.parametrize("start", [None, 10, 250, -750])
@pytest.mark.parametrize("stop", [None, 990, 750, -100])
@pytest.mark.parametrize("step", [None, 2, 3])
def test_XMDPYBackendEntrypoint_time_indexing(start, stop, step):
    traj_fname = "./tests/data/test_traj.xyz"

    cell = np.diag([20.123, 20.123, 20.123])

    result = xr.load_dataset(
        traj_fname,
        engine="xmdpy",
        cell=cell,
        file_format="xyz",
    ).isel(time=slice(start, stop, step))

    expected = xr.load_dataset("./tests/data/test_traj.nc").isel(
        time=slice(start, stop, step)
    )
    assert_equal(result, expected)


@pytest.mark.parametrize("start", [None, 10, 100, -150])
@pytest.mark.parametrize("stop", [None, 180, -100])
@pytest.mark.parametrize("step", [None, 2, 3])
def test_XMDPYBackendEntrypoint_atom_id_indexing(start, stop, step):
    traj_fname = "./tests/data/test_traj.xyz"

    cell = np.diag([20.123, 20.123, 20.123])

    result = xr.load_dataset(
        traj_fname,
        engine="xmdpy",
        cell=cell,
        file_format="xyz",
    ).sel(atom_id=slice(start, stop, step))

    expected = xr.load_dataset("./tests/data/test_traj.nc").sel(
        atom_id=slice(start, stop, step)
    )
    assert_equal(result, expected)


@pytest.mark.parametrize("atom_selection", ["Li", "Cl"])
@pytest.mark.parametrize(
    "time_slices",
    [
        slice(None),
        slice(10, 990),
        slice(None, None, 2),
        slice(30, 700, 3),
    ],
)
def test_XMDPYBackendEntrypoint_atoms_indexing_time_slicing(
    atom_selection, time_slices
):
    traj_fname = "./tests/data/test_traj.xyz"

    cell = np.diag([20.123, 20.123, 20.123])

    result = xr.load_dataset(
        traj_fname,
        engine="xmdpy",
        cell=cell,
        file_format="xyz",
    )

    expected = xr.load_dataset("./tests/data/test_traj.nc").set_xindex("atoms")
    assert_equal(
        result.sel(atoms=atom_selection, time=time_slices),
        expected.sel(atoms=atom_selection, time=time_slices),
    )
