# import xmdpy
import time

import ase.io
from MDAnalysis.coordinates.XYZ import XYZReader
import numpy as np
import xarray as xr


def print_run_times(load_fn, filename, index=slice(None), n_iter=1):
    times = []

    for _ in range(n_iter):
        t = load_fn(filename, index=index)
        times.append(t)

    print(
        f" {load_fn.__name__} : {np.mean(times):.3f} +/ {np.std(times):.3f} seconds over {n_iter} iterations"
    )


def time_load_xarray(filename, cell=None, index=slice(None)):
    start = time.time()
    ds = (
        xr.open_dataset(
            filename,
            engine="xmdpy",
            cell=cell,
            file_format="xyz",
        )
        .isel(time=index)
        .compute()
    )
    end = time.time()
    return end - start


def time_load_ase(filename, index=None):
    start = time.time()
    ase.io.read(filename, format="xyz", index=":")
    end = time.time()
    return end - start


def time_load_mdanalysis(filename, index=slice(None)):
    start = time.time()
    [frame.positions for frame in XYZReader(filename).__getitem__(index)]
    end = time.time()
    return end - start


def main() -> None:
    filename = "./tests/data/test_traj_100ps.xyz"
    cell = np.diag([20.123, 20.123, 20.123])
    index = slice(0, None, 5)
    N = 3
    print_run_times(time_load_xarray, filename, index=index, n_iter=N)
    # print_run_times(time_load_ase, filename, n_iter=N)
    # load_mdanalysis(filename)


if __name__ == "__main__":
    main()
