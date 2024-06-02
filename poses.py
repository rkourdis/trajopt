import pickle
from enum import Enum

class Pose(Enum):
    STANDING_V = 1

def load_robot_pose(pose_type: Pose):
    blobs = {
        Pose.STANDING_V: "standing_pose.bin"
    }

    with open(f"robot_poses/{blobs[pose_type]}", "rb") as rf:
        pose = pickle.load(rf)

    return pose["q"], pose["v"], pose["tau"], pose["λ_local_wa"]