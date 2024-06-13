import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin

# Autodiff foothold kinematics using CasADi.
# Calculates foot frame positions at the provided state:
class ADFootholdKinematics():
    def __init__(self, cmodel, cdata, feet: list[str]):
        self.cmodel, self.cdata = cmodel, cdata
        self.ff_ids = [cmodel.getFrameId(f) for f in feet]

    def __call__(self, q: ca.SX):
        q = cpin.normalize(self.cmodel, q)

        cpin.forwardKinematics(self.cmodel, self.cdata, q)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        pos = [self.cdata.oMf[f].translation for f in self.ff_ids]
        
        # Convert 3x1 vectors to a 4x3 output:
        return ca.horzcat(*pos).T
