import numpy as np
import casadi as ca
import pinocchio as pin
from liecasadi import SE3, SE3Tangent

# Custom state integration function. This is to avoid 
# numerical issues with pin3's integrate during Hessian calculation.
#  Please see: https://github.com/stack-of-tasks/pinocchio/issues/2050 for a similar issue
# Calculates the result of integrating v for a unit timestep starting from q.
# -----------------------------------
# TODO: Try Pinocchio 3 release integrate()
# -----------------------------------
def integrate_state(q: ca.SX, v: ca.SX):
    q_se3 = SE3(pos = q[:3], xyzw = q[3:7])
    v_se3 = SE3Tangent(v[:6])

    # Integrate the floating base using the Lie operation:
    fb_r_se3 = q_se3 * v_se3.exp()

    # Integrate revolute joints manually:
    r_r = q[7:] + v[6:]
    return ca.vertcat(fb_r_se3.pos, fb_r_se3.xyzw, r_r)

# Given a dictionary of {"JOINT_NAME": angle} pairs, return a state vector
# with all joints at the neutral configuration except those in the dictionary,
# which will be set at the provided angles.
def create_state_vector(robot: pin.RobotWrapper, joint_angles: dict[str, float]):
    q_quat = pin.neutral(robot.model)

    for j_name, angle in joint_angles.items():
        idx = robot.model.getJointId(j_name)
        q_quat[robot.model.joints[idx].idx_q] = angle

    return np.expand_dims(q_quat, axis = -1)     # 19x1

# Daft thing converting a CasADi SX to a numpy array.
# For some reason, np.array(SX) doesn't work? Neither .full()?
def ca_to_np(x: ca.SX) -> np.array:
    result = np.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i, j] = x[i, j]
    
    return result

# Flattens a list of SX matrixes into a single column vector.
# Each matrix is flattened by combining all rows into a large column vector,
# all of which are concatenated.
def flatten(mats: list[ca.SX]) -> ca.SX:
    # Flatten each matrix separately:
    flattened_mats = [ca.vertcat(*ca.horzsplit(m.T)) for m in mats]

    # Combine all column vectors:
    return ca.vertcat(*flattened_mats)

# Unflatten a column vector into matrices in the provided shape, row by row.
def unflatten(vars: ca.SX, shape: tuple[int, int]) -> list[ca.SX]:
    assert vars.shape[1] == 1

    num_per_mat = shape[0] * shape[1]
    assert vars.shape[0] % (num_per_mat) == 0

    # Split into column vectors sized (shape[0] * shape[1]) x 1
    mat_cols = ca.vertsplit(vars, num_per_mat)

    return [
        ca.horzcat(*ca.vertsplit(mc, shape[1])).T
        for mc in mat_cols
    ]

# Machine float epsilon:
ε = np.finfo(float).eps