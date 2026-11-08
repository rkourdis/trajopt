import numpy as np
import pinocchio as pin

from utilities import q_quat_to_mrp, ca_to_np

# Various useful robot joint configurations.
# The order of joints does not correspond to the order of joints in the model!

# Go2 crouch within its URDF limits. All four knees bend in the same direction.
FOLDED_JOINT_MAP = {
    f"{leg}_{joint}_joint": angle
    for leg in ("FR", "FL", "RR", "RL")
    for joint, angle in (("hip", 0.0), ("thigh", 1.3), ("calf", -2.6))
}

# Symmetric Go2 standing pose, in radians:
UPRIGHT_JOINT_MAP = {
    f"{leg}_{joint}_joint": angle
    for leg in ("FR", "FL", "RR", "RL")
    for joint, angle in (("hip", 0.0), ("thigh", 0.8), ("calf", -1.6))
}

# Given a dictionary of {"JOINT_NAME": angle} pairs, return a state vector
# with all joints in the neutral configuration except those in the dictionary.
# The returned state expresses the floating base orientation using MRP.
def create_state_vector(robot: pin.RobotWrapper, joint_angles: dict[str, float]) -> np.ndarray:
    q_quat = np.expand_dims(pin.neutral(robot.model), axis = -1)    # 19x1

    for j_name, angle in joint_angles.items():
        if not robot.model.existJointName(j_name):
            raise ValueError(f"Unknown joint: {j_name}")
        idx = robot.model.getJointId(j_name)
        q_quat[robot.model.joints[idx].idx_q] = angle

    return ca_to_np(q_quat_to_mrp(q_quat))
