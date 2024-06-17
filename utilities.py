from fractions import Fraction
from typing import Union, Iterator

import numpy as np
import casadi as ca
import pinocchio as pin
from liecasadi import SE3, SE3Tangent

MatrixLike = Union[np.ndarray, ca.SX]

# This will switch an MRP to its shadow. Typically you would
# do this if the MRP is crossing the unit norm sphere (or if
# some other switching criterion is met), to avoid the singularity
# at 2π. However, doing the switch conditionally during gradient-based
# optimization increases the difficulty of the problem as a
# discontinuity is introduced. For that reason, if a full
# flip is needed, we'll split the problem into two halves,
# with the MRP switched at the beginning of the second one.
# NOTE: I think liecasadi is switching the MRP as well, a while
#       after crossing the unit norm sphere.
def switch_mrp(mrp: ca.SX) -> ca.SX:
    return -mrp / (mrp.T @ mrp)

# Helper function to switch the MRP part of a full state vector:
def switch_mrp_in_q(q_mrp: ca.SX) -> ca.SX:
    return ca.vertcat(q_mrp[:3], switch_mrp(q_mrp[3:6]), q_mrp[6:])

# Quaternion in xyzw form to MRP:
def quat2mrp(xyzw: ca.SX) -> ca.SX:
    normalized = xyzw / ca.norm_2(xyzw)
    return normalized[:3] / (1 + normalized[3])

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

"""
# Given a dictionary of {"JOINT_NAME": angle} pairs, return a state vector
# with all joints at the neutral configuration except those in the dictionary,
# which will be set at the provided angles.
# The returned state expresses the floating base orientation using MRP.
def create_state_vector(robot: pin.RobotWrapper, joint_angles: dict[str, float]) -> np.ndarray:
    q_quat = pin.neutral(robot.model)

    for j_name, angle in joint_angles.items():
        idx = robot.model.getJointId(j_name)
        q_quat[robot.model.joints[idx].idx_q] = angle

    q_mrp = np.concatenate((q_quat[:3], quat2mrp(q_quat[3:7]), q_quat[7:]))
    return np.expand_dims(q_mrp, axis = -1)     # 18x1
"""

# Custom state integration function. This is to avoid 
# numerical issues with pin3's integrate during Hessian calculation.
# Please see: https://github.com/stack-of-tasks/pinocchio/issues/2050
# for a similar issue.
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
def ca_to_np(x: ca.SX) -> np.ndarray:
    result = np.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i, j] = x[i, j]
    
    return result

# Iterates an SX matrix row-by-row.
# This should be used with care as it's inefficient due to 
# CasADi matrices being sparse.
def ca_iter(x: ca.SX) -> Iterator[ca.SX]:
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            yield x[row, col]

# Flattens a list of matrices into a single column vector.
# For each matrix, all rows are combined into a column vector.
# All column vectors are then concatenated.
def flatten_mats(mats: list[MatrixLike]) -> MatrixLike:
    T = type(mats[0])

    if T is ca.SX:
        return ca.vertcat(*iter(ca.vertcat(*ca.horzsplit(m.T)) for m in mats))
    elif T is np.ndarray:
        return np.vstack([np.vstack(np.hsplit(m.T, m.shape[0])) for m in mats])
    else:
        raise ValueError(f"Incorrect matrix type: {T}")

# Unflatten a column vector into matrices in the provided shape, row by row.
def unflatten_mats(vars: MatrixLike, shape: tuple[int, int]) -> list[MatrixLike]:
    assert vars.shape[1] == 1

    num_per_mat = shape[0] * shape[1]
    assert vars.shape[0] % (num_per_mat) == 0

    # Split into column vectors sized (shape[0] * shape[1]) x 1.
    # Then, for each column vector, split into rows of the original matrix
    # (length shape[1]). Stack these and return.
    T = type(vars)
    
    if T is ca.SX:
        return [
            ca.horzcat(*ca.vertsplit(mc, shape[1])).T
            for mc in ca.vertsplit(vars, num_per_mat)
        ]
    
    elif T is np.ndarray:
        return [
            np.hstack(np.vsplit(mc, shape[0])).T
            for mc in np.vsplit(vars, vars.shape[0] / num_per_mat)
        ]

    else:
        raise ValueError(f"Incorrect matrix type: {T}")

# Fraction epsilon - useful for slightly extending time intervals
# as interval tree operations aren't inclusive of the final point:
frac_ε = Fraction("1e-9")