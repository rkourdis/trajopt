import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from utilities import q_mrp_to_quat

# Autodiff foothold kinematics using CasADi.
# Calculates foot frame positions, velocities and accelerations at the provided state:
class ADFootholdKinematics():
    def __init__(self, cmodel, cdata, feet: list[str]):
        self.cmodel, self.cdata = cmodel, cdata
        self.ff_ids = [cmodel.getFrameId(f) for f in feet]

    def __call__(self, q_mrp: ca.SX, v: ca.SX, a: ca.SX):
        q = q_mrp_to_quat(q_mrp)

        # Second order kinematics:
        cpin.forwardKinematics(self.cmodel, self.cdata, q, v, a)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        pos = [self.cdata.oMf[f].translation                                           for f in self.ff_ids]
        vel = [cpin.getFrameVelocity(self.cmodel, self.cdata, f, pin.WORLD).linear     for f in self.ff_ids]
        acc = [cpin.getFrameAcceleration(self.cmodel, self.cdata, f, pin.WORLD).linear for f in self.ff_ids] 
        
        # Convert 3x1 vectors to 4x3 outputs:
        return ca.horzcat(*pos).T, ca.horzcat(*vel).T, ca.horzcat(*acc).T
