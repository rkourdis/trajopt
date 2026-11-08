import time
import pickle
import numpy as np

from robot import Go2
from problem import Problem, Solution
from utilities import ca_to_np, q_mrp_to_quat

def visualise_solution(filename: str, go2: Go2):
    with open(filename, "rb") as rf:
        soln: Solution = pickle.load(rf)

    assert isinstance(soln, Solution)
    sub_trajectories = Problem.load_trajectories(soln)
    g_traj = Problem.stitch_trajectories(sub_trajectories)

    go2.robot.display(ca_to_np(q_mrp_to_quat(g_traj.q_k[0])))
    input(f"Press ENTER to start playback ({g_traj.n_knots} knots)")

    for k, (q_mrp, dt) in enumerate(zip(g_traj.q_k, g_traj.knot_duration)):
        print(f"Knot: {k}, duration: {int(dt * 1e+3)}ms")
        go2.robot.display(ca_to_np(q_mrp_to_quat(q_mrp)))
        time.sleep(dt)
        input()

        # print("norm(τ): ", np.linalg.norm(g_traj.τ_k[k], ord = 2))
        # print("torso height:", g_traj.q_k[k][2] - go2.floor_z)

        # print("RR_calf_joint", g_traj.q_k[k][go2.q_off("RR_calf_joint")])
        # print("RL_calf_joint", g_traj.q_k[k][go2.q_off("RL_calf_joint")])
        # input()
