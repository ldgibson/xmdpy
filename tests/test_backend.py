import numpy as np
import xarray as xr
from xarray.testing import assert_equal


def test_XMDPYBackendEntrypoint():
    traj_fname = "./tests/data/test_traj.xyz"

    cell = np.diag([20.123, 20.123, 20.123])

    result = xr.load_dataset(
        traj_fname,
        engine="xmdpy",
        cell=cell,
        file_format="xyz",
    )

    expected = xr.load_dataset("./tests/data/test_traj.nc")
    assert_equal(result, expected)
