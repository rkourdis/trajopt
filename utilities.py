import numpy as np
import casadi as ca
import pinocchio as pin
from itertools import chain
from liecasadi import SE3, SE3Tangent

# Quaternion in xyzw form to MRP:
def quat2mrp(xyzw: ca.SX) -> ca.SX:
    norm = xyzw / ca.sqrt(xyzw.T @ xyzw)
    return norm[:3] / (1 + norm[3]) 

# MRP to quaternion in xyz form:
def mrp2quat(xyz: ca.SX) -> ca.SX:
    normsq = xyz.T @ xyz
    w = (ca.SX.ones(1) - normsq) / (1 + normsq)
    return ca.vertcat(2 * xyz / (1 + normsq), w)

# State using MRP for base orientation to state with quaternion.
# We assume the floating base is at [0:6] -> x, y, z, mrp
def q_mrp_to_quat(q_mrp: ca.SX) -> ca.SX:
    return ca.vertcat(q_mrp[:3], mrp2quat(q_mrp[3:6]), q_mrp[6:])

# State using quaternion for base orientation to state with MRP.
# We assume the floating base is at [0:7] -> x, y, z, xyzw
def q_quat_to_mrp(q_quat: ca.SX) -> ca.SX:
    return ca.vertcat(q_quat[:3], quat2mrp(q_quat[3:7]), q_quat[7:])

# Given a dictionary of {"JOINT_NAME": angle} pairs, return a state vector
# with all joints at the neutral configuration except those in the dictionary,
# which will be set at the provided angles.
# The returned state expresses the floating base orientation using MRP.
def create_state_vector(robot: pin.RobotWrapper, joint_angles: dict[str, float]):
    q_quat = pin.neutral(robot.model)

    for j_name, angle in joint_angles.items():
        idx = robot.model.getJointId(j_name)
        q_quat[robot.model.joints[idx].idx_q] = angle

    q_mrp = np.concatenate((q_quat[:3], quat2mrp(q_quat[3:7]), q_quat[7:]))
    return np.expand_dims(q_mrp, axis = -1)     # 18x1

# Custom state integration function. This is to avoid 
# numerical issues with pin3's integrate during Hessian calculation.
#  Please see: https://github.com/stack-of-tasks/pinocchio/issues/2050 for a similar issue
# Calculates the result of integrating v for a unit timestep starting from q.
# We assume the input / output floating base orientation uses MRP.
def integrate_state(q_mrp: ca.SX, v: ca.SX):
    q = q_mrp_to_quat(q_mrp)
    q_se3 = SE3(pos = q[:3], xyzw = q[3:7])
    v_se3 = SE3Tangent(v[:6])

    # Integrate the floating base using the Lie operation:
    fb_r_se3 = q_se3 * v_se3.exp()

    # Integrate revolute joints manually:
    r_r = q[7:] + v[6:]
    return ca.vertcat(fb_r_se3.pos, quat2mrp(fb_r_se3.xyzw), r_r)
# =====================================================

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