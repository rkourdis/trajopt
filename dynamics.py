import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from utilities import q_mrp_to_quat

# Autodiff forward dynamics using CasADi.
# Given the robot's state, velocity and torques at all actuated joints,
# calculates all joint accelerations. If no foot is in contact, the dynamics
# are unconstrained. Otherwise, the constrained dynamics are computed with
# the feet in active contact pinned (3D contact points).
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

    def __call__(self, q_mrp: ca.SX, v: ca.SX, τ_act: ca.SX, active_conctacts: list[bool]):
        # Input:
        # q (18 x 1, MRP), v (18 x 1), τ_act (12 x 1), active_contacts (bool x 4)

        # Output:
        # a (18 x 1), λ (4 x 3)
        assert len(active_conctacts) == len(self.feet)

        # Convert the floating base orientation to quaternion for Pinocchio:
        q = q_mrp_to_quat(q_mrp)
            
        # Each actuated joint is one degree of freedom. Create a robot.nv x 1
        # torque vector with only the actuated DoFs set.
        # NOTE: We skip all unactuated joints when applying torques, and external forces.
        tau_full = ca.SX.zeros(self.cmodel.nv, 1)
        for act_dof, j_id in enumerate(self.act_joint_ids):
            tau_full[self.cmodel.joints[j_id].idx_v] = τ_act[act_dof]

        # If a foot is in contact, calculate the constrained dynamics:
        if any(active_conctacts):
            # Initialize active contact constraints:
            active_contact_models = [
                cm for idx, cm in enumerate(self.contact_models)
                if active_conctacts[idx] == True
            ]
            
            active_contact_data = [cm.createData() for cm in active_contact_models]
            cpin.initConstraintDynamics(self.cmodel, self.cdata, active_contact_models)

            # prox_settings = cpin.ProximalSettings(1e-12, 1e-12, 1)
            accel = cpin.constraintDynamics(
                self.cmodel, self.cdata,
                q, v, tau_full,
                active_contact_models, active_contact_data #, prox_settings
            )

            # Return forces only for the active contacts:
            forces = ca.SX.zeros(len(self.feet), 3)

            for f_idx, force in zip(
                (idx for idx, ac in enumerate(active_conctacts) if ac),
                ca.vertsplit(self.cdata.lambda_c, 3)
            ):
                forces[f_idx, :] = force.T

            return accel, forces

        # Otherwise, return the free dynamics using the Articulated Body Algorithm:
        return cpin.aba(self.cmodel, self.cdata, q, v, tau_full), ca.SX.zeros(4, 3)
