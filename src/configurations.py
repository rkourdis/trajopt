import numpy as np
import pinocchio as pin

from utilities import q_quat_to_mrp, ca_to_np

# Various useful robot joint configurations.
# The order of joints does not correspond to the order of joints in the model!

# Solo12 in folded configuration, with the knees slightly open downwards
# so the feet are in contact with the ground:
FOLDED_JOINT_MAP = {
    "FR_HAA": 0,
    "FL_HAA": 0,
    "HR_HAA": 0,
    "HL_HAA": 0,
    "FR_KFE": -np.pi + np.deg2rad(10),
    "FR_HFE": np.pi / 2,
    "FL_KFE": -np.pi + np.deg2rad(10),
    "FL_HFE": np.pi / 2,
    "HR_KFE": np.pi - np.deg2rad(10),
    "HR_HFE": -np.pi / 2,
    "HL_KFE": np.pi - np.deg2rad(10),
    "HL_HFE": -np.pi / 2,
}

# Solo12 standing with legs in V configuration:
UPRIGHT_JOINT_MAP = {
    "FR_HAA": 0,
    "FL_HAA": 0,
    "HR_HAA": 0,
    "HL_HAA": 0,
    "FR_KFE": -np.pi / 2,
    "FR_HFE": np.pi / 4,
    "FL_KFE": -np.pi / 2,
    "FL_HFE": np.pi / 4,
    "HR_KFE": np.pi / 2,
    "HR_HFE": -np.pi / 4,
    "HL_KFE": np.pi / 2,
    "HL_HFE": -np.pi / 4,
}

# Given a dictionary of {"JOINT_NAME": angle} pairs, return a state vector
# with all joints in the neutral configuration except those in the dictionary.
# The returned state expresses the floating base orientation using MRP.
def create_state_vector(robot: pin.RobotWrapper, joint_angles: dict[str, float]) -> np.ndarray:
    q_quat = np.expand_dims(pin.neutral(robot.model), axis = -1)    # 19x1

    for j_name, angle in joint_angles.items():
        idx = robot.model.getJointId(j_name)
        q_quat[robot.model.joints[idx].idx_q] = angle

    return ca_to_np(q_quat_to_mrp(q_quat))
