import pickle
from typing import Optional

import numpy as np
import casadi as ca
import pinocchio as pin

from kinematics import ADFootholdKinematics
from transcription import Trajectory
from poses import load_robot_pose, Pose

# Creates a guess trajectory that constantly holds the desired pose.
# If pose == None, everything in the guess is zero / neutral.
def const_pose_guess(n_knots: int, fk: ADFootholdKinematics, pose: Pose = None) -> Trajectory:
    if pose is not None:
        q0, v0, tau0, λ0 = load_robot_pose(Pose.STANDING_V)
    else:
        q0 = pin.neutral(fk.cmodel.model)
        v0, tau0, λ0 = np.zeros((fk.cmodel.nv, 1)), np.zeros((12, 1)), np.zeros((4, 3))

    a0 = np.zeros((fk.cmodel.nv, 1))                      

    # Create numerical instance of FK to calculate feet kinematics:
    q_sym = ca.SX.sym("q_sym", q0.shape)
    f_pos_0 = ca.Function("num_fk_pos", [q_sym], [fk(q_sym)])(q0)

    q_s = [np.copy(q0) for _ in range(n_knots)]

    for k in range(n_knots):
        if k > n_knots // 2:
            q_s[k][3:7] *= -1

    return Trajectory(
        num_knots   = n_knots,
        q_k         = q_s,
        v_k         = [np.copy(v0) for _ in range(n_knots)],
        a_k         = [np.copy(a0) for _ in range(n_knots)],
        tau_k       = [np.copy(tau0) for _ in range(n_knots)],
        λ_k         = [np.copy(λ0) for _ in range(n_knots)],
        f_pos_k     = [np.copy(f_pos_0) for _ in range(n_knots)],
    )

# Loads a previous solution as an initial guess trajectory.
# If `interp_knots` is set, the trajectory will be interpolated to
# the target knot count.
def prev_soln_guess(
        n_knots: int,
        robot: pin.RobotWrapper,
        filename: str,
        interp_knots: Optional[int] = None
) -> Trajectory:
    
    with open(filename, "rb") as rf:
        soln = pickle.load(rf)
    
    traj = Trajectory.load_from_vec(n_knots, robot, soln["x"])

    if interp_knots is not None:
        return traj.interpolate(interp_knots)
    
    return traj
    
