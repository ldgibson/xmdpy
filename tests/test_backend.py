import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from xmdpy.backend import XMDPYBackendEntrypoint

MOCK_XYZ_FRAME = (
    "3\ni = {0}\nO 0.0 0.0 {0:.1f}\nH 0.9 0.7 {0:.1f}\nH -0.9 0.7 {0:.1f}\n"
)
MOCK_XYZ_TRAJ = "".join([MOCK_XYZ_FRAME.format(i) for i in range(10)])

MOCK_XDATCAR_HEADER = (
    "COMMENT\n1.0\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\n1 2\nO H\n"
)
MOCK_XDATCAR_FRAME = "Direct configuration num= {0}\n0.0 0.0 {1:.1f}\n0.09 0.07 {1:.1f}\n-0.09 0.07 {1:.1f}\n"
MOCK_XDATCAR_TRAJ = MOCK_XDATCAR_HEADER + "".join(
    [MOCK_XDATCAR_FRAME.format(i, i / 10.0) for i in range(10)]
)

REAL_TRAJ_DS = xr.Dataset(
    {
        "xyz": (
            ["time", "atom_id", "xyz_dim"],
            np.stack(
                [
                    np.array([[0.0, 0.0, i], [0.9, 0.7, i], [-0.9, 0.7, i]])
                    for i in range(10)
                ]
            ),
        ),
        "cell": (
            ["time", "cell_vector", "xyz_dim"],
            np.broadcast_to(np.eye(3, dtype="float64") * 10.0, (10, 3, 3)),
        ),
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
def test_XMDPYBackendEntrypoint_time_indexing(fs, start, stop, step) -> None:
    fs.create_file(
        "tests/data/mock_traj.xyz",
        contents=MOCK_XYZ_TRAJ,
    )
    traj_fname = "tests/data/mock_traj.xyz"

    cell = np.diag([10.0, 10.0, 10.0])

    result = xr.load_dataset(
        traj_fname,
        engine=XMDPYBackendEntrypoint,
        cell=cell,
        file_format="xyz",
    ).isel(time=slice(start, stop, step))

    expected = REAL_TRAJ_DS.isel(time=slice(start, stop, step))
    assert_equal(result, expected)


@pytest.mark.parametrize("start", [None, 0, 1, -2])
@pytest.mark.parametrize("stop", [None, 1, -1])
@pytest.mark.parametrize("step", [None, 2])
def test_XMDPYBackendEntrypoint_atom_id_indexing(fs, start, stop, step) -> None:
    fs.create_file(
        "tests/data/mock_traj.xyz",
        contents=MOCK_XYZ_TRAJ,
    )
    traj_fname = "tests/data/mock_traj.xyz"

    cell = np.diag([10.0, 10.0, 10.0])

    result = xr.load_dataset(
        traj_fname,
        engine=XMDPYBackendEntrypoint,
        cell=cell,
        file_format="xyz",
    ).sel(atom_id=slice(start, stop, step))

    expected = REAL_TRAJ_DS.sel(atom_id=slice(start, stop, step))
    assert_equal(result, expected)


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
def test_XMDPYBackendEntrypoint_atoms_indexing_time_slicing(
    fs, atom_selection, time_slices
) -> None:
    fs.create_file(
        "tests/data/mock_traj.xyz",
        contents=MOCK_XYZ_TRAJ,
    )
    traj_fname = "tests/data/mock_traj.xyz"

    cell = np.diag([10.0, 10.0, 10.0])

    result = xr.load_dataset(
        traj_fname,
        engine=XMDPYBackendEntrypoint,
        cell=cell,
        file_format="xyz",
    ).sel(atoms=atom_selection, time=time_slices)

    expected = REAL_TRAJ_DS.sel(atoms=atom_selection, time=time_slices)
    assert_equal(result, expected)
