import numpy as np
import casadi as ca
import pinocchio as pin

from utilities import flatten
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
                                                         
    # Create numerical instance of FK to calculate feet positions:
    q_sym = ca.SX.sym("q_sym", q0.shape)
    f_pos_0 = ca.Function("num_fk", [q_sym], [fk(q_sym)])(q0)

    return Trajectory(
        num_knots   = n_knots,
        q_k         = [np.copy(q0) for _ in range(n_knots)],
        v_k         = [np.copy(v0) for _ in range(n_knots)],
        a_k         = [np.zeros((fk.cmodel.nv, 1)) for _ in range(n_knots)],
        tau_k       = [np.copy(tau0) for _ in range(n_knots)],
        λ_k         = [np.copy(λ0) for _ in range(n_knots)],
        f_pos_k     = [np.copy(f_pos_0) for _ in range(n_knots)]
    )