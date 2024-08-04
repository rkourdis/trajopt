import h5py
import numpy as np

with h5py.File("backflip_optim_footholds.hdf5", "r") as f:
    taus = f["τ"]
    maxn = 0
    
    for k in range(taus.shape[0]):
        maxn = max(maxn, np.linalg.norm(taus[k, :], 2))

    print(maxn)