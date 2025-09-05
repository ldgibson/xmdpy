# from xmdpy.backend import parse_xyz_frames
import numpy as np
import xarray as xr

import xmdpy.backend

if __name__ == "__main__":
    traj_fname = "./trajectories/pbe-d3bj-22e_8750-9750ps_raman.xyz"

    cell = np.diag([22.235, 22.235, 22.235])
    traj = xr.open_dataset(
        traj_fname,
        engine=xmdpy.backend.XMDPYBackendEntrypoint,
        cell=cell,
        file_format="xyz",
    )
    print(traj.isel(frame=slice(0, 2000, 3)).cell[-1].data)
