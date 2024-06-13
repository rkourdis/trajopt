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

    for k, q in enumerate(traj.q_k):
        print(f"Knot: {k}, time: {k * delta_t}s")
        print(traj.tau_k[k])
        
        robot.display(ca_to_np(q))
        time.sleep(delta_t * 2)
        input()

if __name__ == "__main__":
    from robot import load_solo12
    
    FREQ_HZ = 60
    DURATION = 1.0
    FLOOR_Z = -0.226274
    FILENAME = "/dev/null"

    robot, _ = load_solo12(FLOOR_Z, visualise = True)
    visualise_solution(FILENAME, int(FREQ_HZ * DURATION), 1 / FREQ_HZ, robot)