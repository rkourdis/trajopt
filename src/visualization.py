import time
import pickle
import numpy as np

from robot import LeggedRobot
from problem import Problem, Solution
from utilities import ca_to_np, q_mrp_to_quat

def visualise_solution(filename: str, robot: LeggedRobot):
    with open(filename, "rb") as rf:
        soln: Solution = pickle.load(rf)

    assert isinstance(soln, Solution)
    sub_trajectories = Problem.load_trajectories(soln)
    g_traj = Problem.stitch_trajectories(sub_trajectories)

    robot.robot.display(ca_to_np(q_mrp_to_quat(g_traj.q_k[0])))
    input(f"Press ENTER to start playback ({g_traj.n_knots} knots)")

    for k, (q_mrp, v, dt) in enumerate(zip(g_traj.q_k, g_traj.v_k, g_traj.knot_duration)):
        print(f"Knot: {k}, duration: {int(dt * 1e+3)}ms")
        robot.robot.display(ca_to_np(q_mrp_to_quat(q_mrp)))
        
        np.set_printoptions(precision=4, suppress=True)
        # print(q_mrp[robot.q_off("FR_HFE")])
        print(q_mrp.T)
        print(v.T)
        print(list(robot.robot.model.names))
        # print()
        time.sleep(dt)
        
        input()

        # print("norm(τ): ", np.linalg.norm(g_traj.τ_k[k], ord = 2))
        # print("torso height:", g_traj.q_k[k][2] - solo.floor_z)

        # print("HR_KFE", g_traj.q_k[k][solo.q_off("HR_KFE")])
        # print("HL_KFE", g_traj.q_k[k][solo.q_off("HL_KFE")])
        # input()