import pickle
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from collocation import load_solo12
from transcription import Trajectory

def plot_traj_characteristics(traj: Trajectory, delta_t: float):
    ks = list(range(traj.num_knots))
    t = [delta_t * k for k in ks]

    plt.scatter(t, [0] * len(t), s = 6, label = "Collocation Times (sec)")
    plt.plot(t, [float(traj.q_k[k][2]) for k in ks], label = "Torso Z (m)")
    plt.plot(t, [float(traj.v_k[k][2]) / 4 for k in ks], label = "Torso Z vel (m/s x 4)")
    plt.plot(t, [np.max(np.abs(traj.tau_k[k])) for k in ks], label = "Torque max (N*m)")
    # plt.plot(t, [np.min(np.abs(traj.tau_k[k])) for k in ks], label = "Torque min (N*m)")
    plt.plot(t, [np.max(traj.λ_k[k][:, 2]) / 120 for k in ks], label = "Z-up GRF max (N x 120)")
    # plt.plot(t, [np.min(traj.λ_k[k][:, 2]) / 120 for k in ks], label = "Z-up GRF min (N x 120)")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    with open("backflip_launch_20hz_1000ms.bin", "rb") as rf:
        solution = pickle.load(rf)

    robot, _ = load_solo12(floor_z = 0.0, visualise = False)

    traj = Trajectory.load_from_vec(20, robot, solution["x"])
    plot_traj_characteristics(traj, 1/20)