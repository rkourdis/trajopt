import os
import hppfcl
from tqdm import tqdm

import numpy as np
import casadi as ca
from tqdm import tqdm
import pinocchio as pin

def load_solo12():
    pkg_path = os.path.dirname(__file__)
    urdf_path = os.path.join(pkg_path, "example-robot-data/robots/solo_description/robots/solo12.urdf")

    # Load full URDF. This creates a RobotWrapper that contains both the read-only model and the data:
    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_path, package_dirs = [pkg_path], root_joint = pin.JointModelFreeFlyer()
    )

    return robot

from liecasadi import SO3, SO3Tangent

def integrate_se3_custom(q: np.array, v: np.array):
    q_lin, q_rot = q[:3], q[3:7]
    v, ω = v[:3], v[3:6]

    # Calculate integrated quaternion (Lie +):
    q_rot_so3, v_rot_so3t = SO3(q_rot), SO3Tangent(ω) 
    result_rot = (q_rot_so3 + v_rot_so3t).xyzw

    # Caclulate integrated linear position (q_lin + q_rot @ exp(v)_rot)
    # We'll evaluate the rotational part of exp(v) as:
    # exp(v)_rot = V @ v_lin with:
    # V = I + ((1-cosθ)/ θ**2)ωx + ((1 - sinθ/θ) / θ**2)ωx**2
    θ = np.sqrt(ω.T @ ω + 1e-6)
    θ_sq = θ * θ

    ω_hat = np.array([
        [0,    -ω[2],  ω[1]],
        [ω[2],    0,  -ω[0]],
        [-ω[1], ω[0],    0]
    ])

    A = np.sin(θ) / θ
    B = (1 - np.cos(θ)) / θ_sq
    C = (1 - A) / θ_sq

    V = np.eye(3) + B * ω_hat + C * ω_hat @ ω_hat 
    result_lin = q_lin + q_rot_so3.act(V @ v)
    return ca.vertcat(result_lin, result_rot)

if __name__ == "__main__":
    robot = load_solo12()

    q_sym = ca.SX.sym("q", robot.nq)
    v_sym = ca.SX.sym("v", robot.nv)
    # adint = ca.Function("int", [q_sym, v_sym], [AD_integrate_custom(q_sym, v_sym)])

    for _ in tqdm(range(5000)):
        # q0 = np.array([0.26196063,0.08687163,0.46204116,-0.37641723,0.11342385,-0.89211856,-0.22264224,-2.99417096,7.91924802,6.45680105,4.93209632,-6.51783807,7.17886898,4.21002838,0.27069918,-3.92010254,-9.70030824,-8.17194128,-2.7109592])
        q0 = pin.randomConfiguration(robot.model)
        q0[:3] = np.random.rand(3)

        # v0 = np.array([0.67068339, 0.9491493 , 0.46520755, 0.62076714, 0.459383, 0.67577236, 0.63556202, 0.79146051, 0.11766209, 0.16273656, 0.18897034, 0.34521361, 0.43398322, 0.47585262, 0.93481492, 0.386369, 0.25980441, 0.3334634])
        v0 = np.random.rand(robot.nv)

        p_int = pin.integrate(robot.model, q0, v0)[:7]
        c_int = integrate_se3_custom(q0, v0)

        if np.max(p_int - c_int) > 1e-5:
            print("FAIL!")
            print(q0)
            print(v0)