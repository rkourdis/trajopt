import h5py
import pickle

import numpy as np

from problem import Problem
from variables import CollocationVars

# Exports a trajectory to .hdf5 to be imported by the control code.
# Import from C++ using: https://bluebrain.github.io/HighFive/poster/
def export_hdf5(trajectory: CollocationVars[np.ndarray], filename: str):
    with h5py.File(filename, "w") as wf:
        fields = ["q", "v", "a", "τ"]

        # Save trajectory data fields:
        for field in fields:
            data = getattr(trajectory, f"{field}_k")
            mat = np.hstack(data).T                         # n_knots x dim
            wf.create_dataset(field, mat.shape, data = mat)

        # Save the number of knots and knot durations:
        wf.create_dataset(
            "n_knots",
            shape = (1,),
            dtype = 'i',
            data = np.array([trajectory.n_knots])
        )
        
        wf.create_dataset(
            "knot_durations",
            shape = (len(trajectory.knot_duration),),
            data = np.array(trajectory.knot_duration)
        )

if __name__ == "__main__":
    with open("solution.bin", "rb") as rf:
        soln = pickle.load(rf)

    trajectories = Problem.load_trajectories(soln)
    stitched = Problem.stitch_trajectories(trajectories)
    export_hdf5(stitched, "solution.hdf5")
