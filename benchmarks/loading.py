# import xmdpy
import time

# import ase.io
from MDAnalysis.coordinates.XYZ import XYZReader
import numpy as np
import xarray as xr


def print_run_times(load_fn, filename, index=slice(None), n_iter=1, **kwargs):
    times = []

    for _ in range(n_iter):
        t, out = time_fn(load_fn, filename, index=index, **kwargs)
        times.append(t)
        output_size = len(out)

    if len(times) > 1:
        run_time = f"{np.mean(times):.3f} +/ {np.std(times):.3f}"
    else:
        run_time = f"{times[0]:.3f}"

    print(
        f" {load_fn.__name__} : {run_time} seconds over {n_iter} iterations for trajectory with {output_size} frames"
    )


def time_fn(fn, *args, **kwargs):
    start = time.time()
    output = fn(*args, **kwargs)
    end = time.time()
    return (end - start, output)


def load_xarray(filename, cell=None, index=slice(None)):
    ds = (
        xr.open_dataset(
            filename,
            engine="xmdpy",
            cell=cell,
            file_format="xyz",
        )
        .isel(time=index)
        .xyz.compute()
    )
    return ds


def time_load_ase(filename, index=None):
    return ase.io.read(filename, format="xyz", index=":")


def time_load_mdanalysis(filename, index=slice(None)):
    return [frame.positions for frame in XYZReader(filename).__getitem__(index)]


def main() -> None:
    filename = "./tests/data/test_traj.xyz"
    cell = np.diag([20.123, 20.123, 20.123])
    index = slice(None, None)
    N = 1

    print_run_times(load_xarray, filename, cell=cell, index=index, n_iter=N)
    # print_run_times(time_load_ase, filename, n_iter=N)
    # load_mdanalysis(filename)


if __name__ == "__main__":
    main()
