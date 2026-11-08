"""Generate a static Go2 pose without an NLP solver.

Run from any directory with: python3 /path/to/trajopt/experiments/stand_up_closed_form.py
The contact forces use the same sphere-center point-foot approximation as the NLP.
"""

import pickle
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from configurations import UPRIGHT_JOINT_MAP, create_state_vector
from robot import Go2
from utilities import ca_to_np, q_mrp_to_quat, q_quat_to_mrp


def compute_standing_pose(go2: Go2) -> dict:
    robot = go2.robot
    model, data = robot.model, robot.data
    q = ca_to_np(q_mrp_to_quat(create_state_vector(robot, UPRIGHT_JOINT_MAP))).ravel()
    v = np.zeros(model.nv)
    foot_ids = [model.getFrameId(name) for name in go2.frames["feet"]]

    pin.framesForwardKinematics(model, data, q)
    heights = np.array([data.oMf[fid].act(go2.foot_center_offset)[2] for fid in foot_ids])
    if np.ptp(heights) > 1e-9:
        raise ValueError("Standing configuration must place all foot centers at the same height.")
    q[2] += go2.floor_z - heights.mean()
    pin.framesForwardKinematics(model, data, q)

    jacobians = []
    for fid in foot_ids:
        jac = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL_WORLD_ALIGNED).copy()
        offset_world = data.oMf[fid].rotation @ go2.foot_center_offset
        jacobians.append(jac[:3] - pin.skew(offset_world) @ jac[3:])
    jac = np.vstack(jacobians)
    gravity = pin.rnea(model, data, q, v, v).copy()

    # Balance the floating base with minimum-norm world-aligned contact forces.
    forces, _, _, _ = np.linalg.lstsq(jac[:, :6].T, gravity[:6], rcond=None)
    tau_full = gravity - jac.T @ forces
    np.testing.assert_allclose(tau_full[:6], 0.0, atol=1e-9)
    forces = forces.reshape(4, 3)
    if np.any(forces[:, 2] < 0) or np.any(np.abs(forces[:, :2]) > go2.μ * forces[:, 2, None]):
        raise ValueError("Standing contact forces violate the friction pyramid.")
    if np.any(np.abs(tau_full[6:]) > go2.τ_max):
        raise ValueError("Standing torques exceed Go2 effort limits.")
    if np.any(q[7:] < go2.q_min) or np.any(q[7:] > go2.q_max):
        raise ValueError("Standing configuration violates Go2 joint limits.")

    return {
        "robot": "go2",
        "floor_z": go2.floor_z,
        "q": ca_to_np(q_quat_to_mrp(q.reshape(-1, 1))),
        "v": v.reshape(-1, 1),
        "feet": go2.frames["feet"],
        "joint_names": list(model.names)[2:],
        "tau": tau_full[6:].reshape(-1, 1),
        "λ_local_wa": forces,
    }


if __name__ == "__main__":
    pose = compute_standing_pose(Go2())
    path = ROOT / "robot_poses" / "go2_standing_pose.bin"
    with path.open("wb") as wf:
        pickle.dump(pose, wf)
    print(f"Saved Go2 standing pose to {path}")
    print(f"Base height: {pose['q'][2, 0]:.6f} m")
    print(f"Maximum standing torque: {np.max(np.abs(pose['tau'])):.6f} Nm")
    print(f"Total normal force: {pose['λ_local_wa'][:, 2].sum():.6f} N")
