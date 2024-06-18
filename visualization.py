import pickle

from robot import Solo12
from problem import Problem
from utilities import ca_to_np, q_mrp_to_quat

def visualise_solution(filename: str, problem: Problem, solo: Solo12):
    with open(filename, "rb") as rf:
        soln = pickle.load(rf)

    vars = ca_to_np(soln["x"])
    traj = problem.load_solution(vars)

    for k, q_mrp in enumerate(traj.q_k):
        print(f"Knot: {k}")
        solo.robot.display(ca_to_np(q_mrp_to_quat(q_mrp)))

        input("Press ENTER to continue to next knot.")
