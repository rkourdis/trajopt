import time
import pickle

from robot import Solo12
from problem import Problem, Solution
from utilities import ca_to_np, q_mrp_to_quat

def visualise_solution(filename: str, solo: Solo12):
    with open(filename, "rb") as rf:
        soln: Solution = pickle.load(rf)

    assert isinstance(soln, Solution)
    sub_trajectories = Problem.load_solution(soln)
    g_traj = Problem.stitch_trajectories(sub_trajectories)

    input(f"Press ENTER to start playback ({g_traj.n_knots} knots)")

    for k, (q_mrp, dt) in enumerate(zip(g_traj.q_k, g_traj.knot_duration)):
        print(f"Knot: {k}, duration: {int(dt * 1e+3)}ms")
        solo.robot.display(ca_to_np(q_mrp_to_quat(q_mrp)))
        time.sleep(dt)
