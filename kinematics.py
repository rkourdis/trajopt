import casadi as ca
from pinocchio import casadi as cpin

# Autodiff foothold kinematics using CasADi.
# Calculates the position of all feet at the provided state.
class ADFootholdKinematics():
    def __init__(self, cmodel, cdata, feet: list[str]):
        self.cmodel, self.cdata = cmodel, cdata
        self.ff_ids = [cmodel.getFrameId(f) for f in feet]

    def __call__(self, q_quat: ca.SX):
        q_norm = cpin.normalize(self.cmodel, q_quat)

        cpin.forwardKinematics(self.cmodel, self.cdata, q_norm)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        positions = [self.cdata.oMf[ff_id].translation for ff_id in self.ff_ids]
        return ca.horzcat(*positions).T         # 4 x 3 output
