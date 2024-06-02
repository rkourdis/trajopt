import time
import pickle
import pinocchio as pin

from transcription import Trajectory
from utilities import ca_to_np

def visualise_solution(filename: str, n_knots: int, delta_t: float, robot: pin.RobotWrapper):
    with open(filename, "rb") as rf:
        soln = pickle.load(rf)
    
    traj = Trajectory.load_from_vec(n_knots, robot, soln["x"])

    input(f"Start trajectory ({n_knots * delta_t * 1e+3}ms)!")

    for q_quat in traj.q_k:
        robot.display(ca_to_np(q_quat))
        time.sleep(delta_t)
