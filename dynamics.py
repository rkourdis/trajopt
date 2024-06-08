import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from utilities import q_mrp_to_quat

# Autodiff forward dynamics using CasADi.
# Given the robot's state, velocity and torques at all actuated joints,
# calculates all joint accelerations. If `contact` == False, the robot
# is unconstrained. Otherwise, the constrained dynamics are computed,
# assuming the feet are pinned in place (3D contact point).
# The contact forces are also returned.
class ADForwardDynamics():
    def __init__(self, cmodel, cdata, feet: list[str], act_joint_ids: list[int]):
        self.cmodel, self.cdata = cmodel, cdata

        self.feet = feet
        self.act_joint_ids = act_joint_ids

        # Initialise feet contact information with Pinocchio:
        self.contact_models = [
            cpin.RigidConstraintModel(
                cpin.ContactType.CONTACT_3D,
                cmodel.frames[f_id].parentJoint,
                cmodel.frames[f_id].placement,
                pin.LOCAL_WORLD_ALIGNED
            )

            for foot in feet
            if (f_id := cmodel.getFrameId(foot))
        ]

        # # Baumgarte constraint stabilization:
        # for cm in self.contact_models:
        #     cm.corrector.Kp = 10.0
        #     cm.corrector.Kd = 2 * math.sqrt(cm.corrector.Kp)

        self.contact_data = [cm.createData() for cm in self.contact_models]
        cpin.initConstraintDynamics(cmodel, cdata, self.contact_models)

    def __call__(self, q_mrp: ca.SX, v: ca.SX, τ_act: ca.SX, contact: bool):
        # Input:
        # q (18 x 1, MRP), v (18 x 1), τ_act (12 x 1), contact (bool)

        # Output:
        # a (18 x 1)

        # Convert the floating base orientation to quaternion for Pinocchio:
        q = q_mrp_to_quat(q_mrp)
            
        # Each actuated joint is one degree of freedom. Create a robot.nv x 1
        # torque vector with only the actuated DoFs set.
        # NOTE: We skip all unactuated joints when applying torques, and external forces.
        tau_full = ca.SX.zeros(self.cmodel.nv, 1)
        for act_dof, j_id in enumerate(self.act_joint_ids):
            tau_full[self.cmodel.joints[j_id].idx_v] = τ_act[act_dof]

        # If we're in contact, calculate the constrained (feet pinned) dynamics.
        if contact == True:
            # prox_settings = cpin.ProximalSettings(1e-12, 1e-12, 1)
            accel = cpin.constraintDynamics(
                self.cmodel, self.cdata,
                q, v, tau_full,
                self.contact_models, self.contact_data #, prox_settings
            )

            # TODO: Make sure to return only the correct feet:
            forces = ca.reshape(self.cdata.lambda_c, 4, 3)
            return accel, forces

        # Otherwise, return the free dynamics using the Articulated Body Algorithm:
        return cpin.aba(self.cmodel, self.cdata, q, v, tau_full), ca.SX.zeros(4, 3)
