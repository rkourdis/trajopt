import time
import pickle
import pinocchio as pin

from trajectory import Trajectory
from utilities import ca_to_np, q_mrp_to_quat

def visualise_solution(filename: str, n_knots: int, delta_t: float, robot: pin.RobotWrapper):
    with open(filename, "rb") as rf:
        soln = pickle.load(rf)
    
    traj = Trajectory.load_from_vec(n_knots, robot, soln["x"])

    input(f"Start trajectory ({n_knots * delta_t * 1e+3}ms)!")

    for k, q_mrp in enumerate(traj.q_k):
        print(f"Knot: {k}, time: {k * delta_t}s")
        print(traj.λ_k[k])
        robot.display(ca_to_np(q_mrp_to_quat(q_mrp)))
        time.sleep(delta_t * 2)
        input()

if __name__ == "__main__":
    from robot import load_solo12
    
    FREQ_HZ = 60
    DURATION = 1.0
    FLOOR_Z = -0.226274
    FILENAME = "jump_60hz_1000ms_so_good.bin"

    robot, _ = load_solo12(FLOOR_Z, visualise = True)
    visualise_solution(FILENAME, int(FREQ_HZ * DURATION), 1 / FREQ_HZ, robot)