import sys

import xarray as xr

import xmdpy

BOLD = "\033[1m"  # ANSI escape sequence for bold text
END = "\033[0m"  # ANSI escape sequence to reset formatting

if __name__ == "__main__":
    traj = xr.open_dataset(
        "./trajectories/pbe-d3bj-22e_8750-9750ps_raman.xyz",
        engine=xmdpy.backend.XYZBackendEntrypoint,
        cell=22.235,
    )
    cell = traj.xmd.cell
    print(traj)
    print(sys.getsizeof(cell))
    cell2 = xmdpy.Cell(22.235)
    print(cell2)
    print(sys.getsizeof(cell2.array))
