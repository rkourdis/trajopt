import numpy as np

# Various useful robot joint configurations.
# The order of joints does not correspond to the order of joints in the model!

# Solo 12 at folded configuration:
FOLDED_JOINT_MAP = {
    "FR_HAA": 0,
    "FL_HAA": 0,
    "HR_HAA": 0,
    "HL_HAA": 0,
    "FR_KFE": -np.pi,
    "FR_HFE": np.pi / 2,
    "FL_KFE": -np.pi,
    "FL_HFE": np.pi / 2,
    "HR_KFE": np.pi,
    "HR_HFE": -np.pi / 2,
    "HL_KFE": np.pi,
    "HL_HFE": -np.pi / 2,
}

# Solo 12 standing with legs at V configuration:
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