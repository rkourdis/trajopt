import casadi as ca
from pinocchio import casadi as cpin

from robot import Solo12
from utilities import q_mrp_to_quat

# Autodiff frame kinematics using CasADi.
# Calculates frame positions at the provided state:
class ADFrameKinematics():
    def __init__(self, robot: Solo12):
        self.cmodel, self.cdata = robot.cmodel, robot.cdata

        self.frame_ids = {
            group: [self.cmodel.getFrameId(fn) for fn in names]
            for group, names in robot.frames.items()
        }

    def __call__(self, q_mrp: ca.SX):
        q = q_mrp_to_quat(q_mrp)

        cpin.forwardKinematics(self.cmodel, self.cdata, q)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        # For each group of frames, get all frame positions (3x1)
        # and concatenate them into a (3xN) matrix. Return its
        # transpose (Nx3).
        return {
            group: ca.horzcat(
                *iter(self.cdata.oMf[fid].translation for fid in ids)
            ).T
            
            for group, ids in self.frame_ids.items()
        }
