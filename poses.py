import pickle
from enum import Enum

import casadi as ca
from utilities import ca_to_np

class Pose(Enum):
    STANDING_V = 1

# MRP to quaternion in xyz form:
def mrp2quat(xyz: ca.SX) -> ca.SX:
    normsq = xyz.T @ xyz
    w = (ca.SX.ones(1) - normsq) / (1 + normsq)
    return ca.vertcat(2 * xyz / (1 + normsq), w)

# State using MRP for base orientation to state with quaternion.
# We assume the floating base is at [0:6] -> x, y, z, mrp
def q_mrp_to_quat(q_mrp: ca.SX) -> ca.SX:
    return ca.vertcat(q_mrp[:3], mrp2quat(q_mrp[3:6]), q_mrp[6:])

def load_robot_pose(pose_type: Pose):
    blobs = {
        Pose.STANDING_V: "standing_pose.bin"
    }

    with open(f"robot_poses/{blobs[pose_type]}", "rb") as rf:
        pose = pickle.load(rf)

    q_quat = ca_to_np(q_mrp_to_quat(pose["q"]))
    return q_quat, pose["v"], pose["tau"], pose["λ_local_wa"]
