import pickle
from enum import Enum
from functools import lru_cache
from pathlib import Path

class Pose(Enum):
    STANDING = 1

@lru_cache(maxsize=None)
def _load_pose(pose_type: Pose):
    blobs = {Pose.STANDING: "go2_standing_pose.bin"}
    path = Path(__file__).resolve().parents[1] / "robot_poses" / blobs[pose_type]
    if not path.is_file():
        raise FileNotFoundError(
            f"Go2 standing pose missing: {path}. Run python3 experiments/stand_up_closed_form.py."
        )
    with path.open("rb") as rf:
        pose = pickle.load(rf)
    if pose.get("robot") != "go2":
        raise ValueError("Standing pose must be generated for Go2.")
    return pose

# Regenerate using experiments/stand_up_closed_form.py after changing the model
# or standing joint angles. Translate the pose for a different ground height.
def load_robot_pose(pose_type: Pose, floor_z: float = 0.0):
    pose = _load_pose(pose_type)
    q = pose["q"].copy()
    q[2, 0] += floor_z - pose["floor_z"]
    return q, pose["v"].copy(), pose["tau"].copy(), pose["λ_local_wa"].copy()
