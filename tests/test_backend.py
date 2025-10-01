import os

import dask
import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose, assert_equal

from xmdpy.backend import XMDPYBackendEntrypoint


@pytest.fixture
def fake_filesystem(fs):
    dask_real_path = os.path.dirname(dask.__file__)
    fs.add_real_directory(dask_real_path)

    MOCK_XYZ_FRAME = (
        "3\ni = {0}\nO 0.0 0.0 {0:.1f}\nH 0.9 0.7 {0:.1f}\nH -0.9 0.7 {0:.1f}\n"
    )
    MOCK_XYZ_TRAJ = "".join([MOCK_XYZ_FRAME.format(i) for i in range(10)])

    MOCK_XDATCAR_HEADER = (
        "COMMENT\n1.0\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nO H\n1 2\n"
    )
    MOCK_XDATCAR_FRAME = (
        "Direct configuration num= {0}\n"
        "0.0 0.0 {1:.1f}\n"
        "0.09 0.07 {1:.1f}\n"
        "-0.09 0.07 {1:.1f}\n"
    )

    MOCK_XDATCAR_TRAJ = MOCK_XDATCAR_HEADER + "".join(
        [MOCK_XDATCAR_FRAME.format(i, i / 10.0) for i in range(10)]
    )

    fs.create_file(
        "./tests/data/mock_traj.xyz",
        contents=MOCK_XYZ_TRAJ,
    )
    fs.create_file(
        "./tests/data/mock_XDATCAR",
        contents=MOCK_XDATCAR_TRAJ,
    )
    assert os.path.exists("./tests/data/mock_traj.xyz")
    assert os.path.exists("./tests/data/mock_XDATCAR")
    yield fs


@pytest.fixture
def expected() -> xr.Dataset:
    xyz_data = np.stack(
        [np.array([[0.0, 0.0, i], [0.9, 0.7, i], [-0.9, 0.7, i]]) for i in range(10)]
    )

    cell_data = np.broadcast_to(np.eye(3, dtype="float64") * 10.0, (10, 3, 3))

    return xr.Dataset(
        data_vars={
            "xyz": (["time", "atom_id", "xyz_dim"], xyz_data),
            "cell": (["time", "cell_vector", "xyz_dim"], cell_data),
        },
        coords={
            "time": ("time", np.arange(10)),
            "atom_id": ("atom_id", np.arange(3)),
            "atoms": ("atom_id", list("OHH")),
            "xyz_dim": ("xyz_dim", list("xyz")),
            "cell_vector": ("cell_vector", list("ABC")),
        },
    ).set_xindex("atoms")


@pytest.mark.parametrize("start", [None, 1, -4])
@pytest.mark.parametrize("stop", [None, 3, -2])
@pytest.mark.parametrize("step", [None, 2, 3])
def test_XMDPYBackendEntrypoint_time_indexing_xyz(
    fake_filesystem, expected, start, stop, step
) -> None:
    traj_fname = "./tests/data/mock_traj.xyz"

    cell = np.diag([10.0, 10.0, 10.0])

    result = xr.load_dataset(
        traj_fname,
        engine=XMDPYBackendEntrypoint,
        cell=cell,
        file_format="xyz",
    ).isel(time=slice(start, stop, step))

    assert_equal(result, expected.isel(time=slice(start, stop, step)))


@pytest.mark.parametrize("start", [None, 0, 1, -2])
@pytest.mark.parametrize("stop", [None, 1, -1])
@pytest.mark.parametrize("step", [None, 2])
def test_XMDPYBackendEntrypoint_atom_id_indexing_xyz(
    fake_filesystem, expected, start, stop, step
) -> None:
    traj_fname = "./tests/data/mock_traj.xyz"

    cell = np.diag([10.0, 10.0, 10.0])

    result = xr.load_dataset(
        traj_fname,
        engine=XMDPYBackendEntrypoint,
        cell=cell,
        file_format="xyz",
    ).sel(atom_id=slice(start, stop, step))

    assert_equal(result, expected.sel(atom_id=slice(start, stop, step)))


@pytest.mark.parametrize("atom_selection", ["O", "H"])
@pytest.mark.parametrize(
    "time_slices",
    [
        slice(None),
        slice(None, None, 2),
        slice(None, None, 3),
        slice(1, 10, 3),
    ],
)
def test_XMDPYBackendEntrypoint_atoms_indexing_time_slicing_xyz(
    fake_filesystem, expected, atom_selection, time_slices
) -> None:
    traj_fname = "./tests/data/mock_traj.xyz"

    cell = np.diag([10.0, 10.0, 10.0])

    result = xr.load_dataset(
        traj_fname,
        engine=XMDPYBackendEntrypoint,
        cell=cell,
        file_format="xyz",
    ).sel(atoms=atom_selection, time=time_slices)

    assert_equal(result, expected.sel(atoms=atom_selection, time=time_slices))


@pytest.mark.parametrize("start", [None, 1, -4])
@pytest.mark.parametrize("stop", [None, 3, -2])
@pytest.mark.parametrize("step", [None, 2, 3])
def test_XMDPYBackendEntrypoint_time_indexing_XDATCAR(
    fake_filesystem, expected, start, stop, step
) -> None:
    traj_fname = "./tests/data/mock_XDATCAR"

    result = xr.load_dataset(
        traj_fname,
        engine=XMDPYBackendEntrypoint,
        file_format="xdatcar",
    ).isel(time=slice(start, stop, step))

    assert_allclose(
        result,
        expected.isel(time=slice(start, stop, step)),
    )


@pytest.mark.parametrize("start", [None, 0, 1, -8])
@pytest.mark.parametrize("stop", [None, 9, -1])
@pytest.mark.parametrize("step", [None, 2])
def test_XMDPYBackendEntrypoint_atom_id_indexing_XDATCAR(
    fake_filesystem, expected, start, stop, step
) -> None:
    traj_fname = "./tests/data/mock_XDATCAR"

    result = xr.load_dataset(
        traj_fname,
        engine=XMDPYBackendEntrypoint,
        file_format="xdatcar",
    ).sel(atom_id=slice(start, stop, step))

    assert_allclose(result, expected.sel(atom_id=slice(start, stop, step)))


@pytest.mark.parametrize("atom_selection", ["O", "H"])
@pytest.mark.parametrize(
    "time_slices",
    [
        slice(None),
        slice(None, None, 2),
        slice(None, None, 3),
        slice(1, 10, 3),
    ],
)
def test_XMDPYBackendEntrypoint_atoms_indexing_time_slicing_XDATCAR(
    fake_filesystem, expected, atom_selection, time_slices
) -> None:
    traj_fname = "./tests/data/mock_XDATCAR"

    result = xr.load_dataset(
        traj_fname,
        engine=XMDPYBackendEntrypoint,
        file_format="xdatcar",
    ).sel(atoms=atom_selection, time=time_slices)

    assert_allclose(result, expected.sel(atoms=atom_selection, time=time_slices))
