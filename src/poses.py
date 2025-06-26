import pickle
from enum import Enum

class Pose(Enum):
    SOLO_STANDING_V = 1

POSE_BLOBS = {
    # To solve for standing torques and reaction forces, run
    # `experiments/stand_up_closed_form.py`:
    Pose.SOLO_STANDING_V: "solo_standing_pose.bin"
}

def load_robot_pose(pose_type: Pose):
    with open(f"robot_poses/{POSE_BLOBS[pose_type]}", "rb") as rf:
        pose = pickle.load(rf)

    return pose["q"], pose["v"], pose["tau"], pose["λ_local_wa"]