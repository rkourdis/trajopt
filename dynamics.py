import casadi as ca
from pinocchio import casadi as cpin

from robot import Solo12
from utilities import q_mrp_to_quat

# Autodiff forward dynamics using CasADi.
# Given the robot's state, velocity, torques at all joints and foot GRFs, this
# calculates the state acceleration using the Articulated Body Algorithm.
# The foot GRFs are expressed in each foot's local world-aligned frame.
class ADForwardDynamics():
    def __init__(self, robot: Solo12):
        self.robot = robot
        self.cmodel, self.cdata = robot.cmodel, robot.cdata

        # Frame IDs of each foot:
        self.foot_frame_ids = [self.cmodel.getFrameId(f) for f in robot.frames["feet"]]

        # Joint IDs of each foot's frame parent joint:
        self.foot_parent_joint_ids = [
            self.cmodel.frames[ff_id].parentJoint for ff_id in self.foot_frame_ids
        ]

    def __call__(self, q_mrp: ca.SX, v: ca.SX, τ_act: ca.SX, λ: ca.SX):
        # Input:
        # q (18 x 1, MRP), v (18 x 1), τ_act (12 x 1), λ (4  x 3) in local world aligned frame

        # Output:
        # a (18 x 1)

        # Convert the floating base orientation to quaternion for Pinocchio:
        q = q_mrp_to_quat(q_mrp)

        # λ contains GRFs for each foot in the local world-aligned frame.
        # Find how they're expressed in the parent joint frames at the given
        # robot state. FK will populate robot.data.oMf.
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
        fext_full = [cpin.Force.Zero() for _ in range(len(self.cmodel.joints))]

        for foot_idx, (foot_frame_id, parent_joint_id) in enumerate(
            zip(self.foot_frame_ids, self.foot_parent_joint_ids)
        ):
            grf_at_foot = cpin.Force(λ[foot_idx, :].T, ca.SX.zeros(3))

            # Local world-aligned frame at foot:
            local_wa_f = cpin.SE3(ca.SX.eye(3), self.cdata.oMf[foot_frame_id].translation)

            # Express contact force in the parent joint's frame:
            fext_full[parent_joint_id] = self.cdata.oMi[parent_joint_id].actInv(
                local_wa_f.act(grf_at_foot)
            )
            
        # Each actuated joint is one degree of freedom. Create a robot.nv x 1
        # torque vector with only the actuated DoFs set.
        # NOTE: We skip all unactuated joints when applying torques, and external forces.
        tau_full = ca.SX.zeros(self.cmodel.nv, 1)
        for act_dof, j_id in enumerate(self.robot.actuated_joints):
            tau_full[self.cmodel.joints[j_id].idx_v] = τ_act[act_dof]

        # We calculate the unconstrained dynamics using the ABA algorithm.
        # The constraint forces will be chosen by the optimization so that they balance
        # the legs on contact, as described by the contact constraints.

        # Add a bit of viscous friction to the joints:
        damping = ca.vertcat(ca.SX.zeros(6), -0.01 * v[6:])
        
        return cpin.aba(self.cmodel, self.cdata, q, v, tau_full + damping, fext_full)
