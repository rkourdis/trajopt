import casadi as ca
from pinocchio import casadi as cpin

from robot import Go2
from utilities import q_mrp_to_quat

# Autodiff frame kinematics using CasADi.
# Calculates frame placements at the provided robot state:
class ADFrameKinematics():
    def __init__(self, robot: Go2):
        self.cmodel, self.cdata = robot.cmodel, robot.cdata
        self.foot_center_offset = ca.SX(robot.foot_center_offset)

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
                *(self.cdata.oMf[fid].act(self.foot_center_offset) if group == "feet"
                  else self.cdata.oMf[fid].translation for fid in ids)
            ).T
            
            for group, ids in self.frame_ids.items()
        }
