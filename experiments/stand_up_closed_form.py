import os
import hppfcl

import numpy as np
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin

def load_solo12(visualize = False):
    pkg_path = os.path.dirname(__file__)
    urdf_path = os.path.join(pkg_path, "example-robot-data/robots/solo_description/robots/solo12.urdf")

    # Load full URDF. This creates a RobotWrapper that contains both the read-only model and the data:
    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_path, package_dirs = [pkg_path], root_joint = pin.JointModelFreeFlyer()
    )

    if not visualize:
        return robot, None

    visualizer = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )

    robot.setVisualizer(visualizer)
    robot.initViewer()
    robot.loadViewerModel()

    return robot, visualizer

def create_joint_vector(robot: pin.RobotWrapper, joint_angles: dict[str, float]):
    q_quat = pin.neutral(robot.model)

    for j_name, angle in joint_angles.items():
        idx = robot.model.getJointId(j_name)
        q_quat[robot.model.joints[idx].idx_q] = angle

    return np.expand_dims(q_quat, axis = -1)     # 19x1

if __name__ == "__main__":
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

    robot, viz = load_solo12(visualize = False)
    bl_frame_id = robot.model.getFrameId('base_link')

    q = create_joint_vector(robot, UPRIGHT_JOINT_MAP)
    v, a = np.zeros(robot.nv), np.zeros(robot.nv)

    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()

    FEET = ['FR_FOOT', 'FL_FOOT', 'HR_FOOT', 'HL_FOOT']
    foot_fr_ids = [robot.model.getFrameId(f) for f in FEET]
    joint_frame_ids = [robot.model.getFrameId(j) for j in robot.model.names[1:]]
    
    g = pin.rnea(robot.model, robot.data, q, v, a)

    J_feet_wrt_q = [
        np.copy(pin.computeFrameJacobian(robot.model, robot.data, q, frame_id, pin.LOCAL))
        for frame_id in foot_fr_ids
    ]

    J_feetxyz_wrt_baseq = [np.copy(J[:3, :6]) for J in J_feet_wrt_q]

    # J_baseq_wrt_feetxyz @ [[v_FR_FOOT_x, v_FR_FOOT_y, v_FR_FOOT_z, v_FL_FOOT_x, ....]].T = v_base
    J_baseq_wrt_feetxyz = np.vstack(J_feetxyz_wrt_baseq).T              # 6x12

    ################
    feet_forces_local = np.linalg.pinv(J_baseq_wrt_feetxyz) @ g[:6]
    foot_force_local = np.split(feet_forces_local, len(FEET))

    contact_forces_at_bl = []
    pin.framesForwardKinematics(robot.model, robot.data, q)

    foot_force_local_wa = []

    for idx, lf in enumerate(foot_force_local):
        local_force = pin.Force(lf, np.zeros(3))
        local_frame = robot.data.oMf[foot_fr_ids[idx]]

        local_wa_frame = pin.SE3(np.eye(3), robot.data.oMf[foot_fr_ids[idx]].translation)
        local_wa_force = local_wa_frame.actInv(local_frame.act(local_force))

        foot_force_local_wa.append(local_wa_force.linear)

    print("LOCAL FORCES")
    print(foot_force_local)

    print("LOCAL WA FORCES")
    print(foot_force_local_wa)

    for idx, foot_frame_id in enumerate(foot_fr_ids):
        f_bl = robot.data.oMf[joint_frame_ids[0]].actInv(robot.data.oMf[foot_frame_id]).act(
            pin.Force(foot_force_local[idx], np.zeros(3))
        )

        print(f"Force at base link from {FEET[idx]}: {f_bl.vector}")
        contact_forces_at_bl.append(f_bl)

    print()

    J_feetxyz_wrt_joints = [np.copy(J[:3, 6:]) for J in J_feet_wrt_q]
    J_joints_wrt_feetxyz = np.vstack(J_feetxyz_wrt_joints).T

    tau = g[6:] - J_joints_wrt_feetxyz @ feet_forces_local
    print("MANUALLY CALCULATED TAU: ", tau)

    JOINTS = ["FR_KFE", "FL_KFE", "HR_KFE", "HL_KFE"]
    total_feet_force_at_joints = [pin.Force.Zero() for _ in range(len(joint_frame_ids) + 1)]

    for idx, c_joint in enumerate(JOINTS):
        j_id = robot.model.getJointId(c_joint)
        jf_id = robot.model.getFrameId(c_joint)

        total_feet_force_at_joints[j_id] = \
            robot.data.oMf[jf_id].actInv(robot.data.oMf[bl_frame_id]).act(
                contact_forces_at_bl[idx]
            )

    # Find torques:
    rnea_tau = pin.rnea(robot.model, robot.data, q, v, a, total_feet_force_at_joints)
    print("RNEA TAU:", rnea_tau[6:])

    a = pin.aba(
        robot.model,
        robot.data,
        q,
        v,
        np.append(np.zeros(6), tau),
        total_feet_force_at_joints
    )

    print("ALPHA WITH MANUAL TAU:", a)

    a_rnea_tau = pin.aba(
        robot.model,
        robot.data,
        q,
        v,
        rnea_tau,
        total_feet_force_at_joints
    )

    print("ALPHA WITH RNEA TAU:", a_rnea_tau)

    import pickle
    from utilities import ca_to_np

    with open("standing_pose_quat.bin", "wb") as wf:
        data = {
            "q": ca_to_np(q),
            "v": np.zeros((robot.model.nv, 1)),
            "feet": FEET,
            "tau": tau.reshape((len(tau), 1)),
            "λ_local": np.vstack(np.split(feet_forces_local, 4)),
            "λ_local_wa": np.vstack(foot_force_local_wa)
        }

        pickle.dump(data, wf)
        
    exit()

    total_contact_base = np.sum([
        robot.data.oMf[bl_frame_id].actInv(robot.data.oMf[foot_frame_id]).act(
            pin.Force(foot_force_local[foot_idx], np.zeros(3))
        )
        for foot_idx, foot_frame_id in enumerate(foot_fr_ids)
    ])

    print(total_contact_base)
    print(g[:6])
