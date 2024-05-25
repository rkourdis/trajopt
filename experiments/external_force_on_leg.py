import os
import sys
import time
import pickle
import functools
from tqdm import tqdm
from typing import Callable
from dataclasses import dataclass

import numpy as np
import casadi as ca
import pinocchio as pin
import matplotlib.pyplot as plt

def load_limited_solo12(free_joints = ["FR_KFE"], visualize = False):
    pkg_path = "C:\\Users\\rafael\\Projects\\Quadruped\\optimization\\"
    urdf_path = os.path.join(pkg_path, "example-robot-data/robots/solo_description/robots/solo12.urdf")

    # Load full URDF. This creates a RobotWrapper that contains both the read-only model and the data:
    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_path, package_dirs = [pkg_path], root_joint = pin.JointModelFreeFlyer()
    )

    joints_to_lock = list(set(robot.model.names) - {"universe"} - set(free_joints))
    limited_robot = robot.buildReducedRobot(list_of_joints_to_lock=joints_to_lock)

    if not visualize:
        return limited_robot, None

    visualizer = pin.visualize.MeshcatVisualizer(
        limited_robot.model, limited_robot.collision_model, limited_robot.visual_model
    )

    limited_robot.setVisualizer(visualizer)
    limited_robot.initViewer()
    limited_robot.loadViewerModel()
    limited_robot.display(pin.neutral(limited_robot.model))

    return limited_robot, visualizer

@functools.cache
def get_joint_offset(joint: str, size_getter: Callable[[pin.JointModel], int], robot) -> int:
    joint_id = robot.model.getJointId(joint)
    return sum(size_getter(robot.model.joints[idx]) for idx in range(joint_id - 1))

# These return q / v vector indices for indexing the quantities inside
# the whole-robot position / velocity vectors:
def get_q_off(joint: str, robot) -> int:
    return get_joint_offset(joint, lambda j: j.nq, robot)

def get_v_off(joint: str, robot) -> int:
    return get_joint_offset(joint, lambda j: j.nv, robot)

@dataclass
class State:
    # Second order dynamics:
    #    x_dd = f(t, x, x_d, τ, λ)
    #
    # x_d tells you how x evolves
    # x_dd tells you how x_d evolves, is dependent on x, x_d, τ, λ
    t: float
    x: float
    x_d: float

@dataclass
class Trajectory:
    path: list[State]

@dataclass
class Input:
    t: float
    u: float

# Z-up force on robot foot:
@dataclass
class GRF:
    t: float
    λ: float

# class ForwardDynamics(ca.Callback):
#   def __init__(self, name: str, robot, feet: set[str], opts={}):
#     ca.Callback.__init__(self)
#     self.robot = robot
#     self.construct(name, opts)

#   # Number of inputs and outputs:
#   def get_n_in(self): return 3
#   def get_n_out(self): return 1

#   # Evaluate numerically:
#   def eval(self, arg):
#     q, v, tau, λ = \
#         np.array(arg[0]), np.array(arg[1]), np.array(arg[2]), np.array(arg[3])

#     # We'll calculate the unconstrained dynamics using the ABA algorithm.
#     # The constraint forces will be chosen by the optimization so that they balance
#     # the legs on contact, as described by the contact constraints.
#     # In constrained FD, the constraint forces will be implicitly calculated and enforced,
#     # but we want to choose them explicitly (otherwise we'll need to know whether we're in
#     # contact or not in advance).
#     # NOTE: Joints[0] is the 'universe' joint - do not apply any forces there.
#     # This is strange because when you do .njoints you get = 1.
#     return pin.aba(
#         self.robot.model, self.robot.data, q, v, tau
#     )

# This will populate robot.data.oMf with the correct frame positions
# for a given robot configuration:
def calculate_fk(robot: pin.RobotWrapper, q: np.array):
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)

def get_frame_id(robot: pin.RobotWrapper, joint_name: str):
    return robot.model.getFrameId(joint_name)

if __name__ == "__main__":
    OUTPUT_BIN = "trajectory_no_accel_bc.bin"
    foot = "FR_FOOT"

    robot, viz = load_limited_solo12(visualize = True)
    # fd = ForwardDynamics(
    #     "fd", robot, feet = {foot}, opts = {"enable_fd": True}
    # )

    # ====================================================================
    # q0 = pin.neutral(robot.model)
    # calculate_fk(robot, q0)

    # fe_global_frame = robot.data.oMf[get_frame_id(robot, "FR_HFE")]
    # knee_global_frame = robot.data.oMf[get_frame_id(robot, "FR_KFE")]
    # foot_global_frame = robot.data.oMf[get_frame_id(robot, "FR_FOOT")]

    """
    FR_HFE   R =
    1 0 0
    0 1 0
    0 0 1
    p =  0.1946 -0.1015       0

    FR_KFE   R =
    1 0 0
    0 1 0
    0 0 1
    p =   0.1946 -0.13895    -0.16

    FR_FOOT   R =
    1 0 0
    0 1 0
    0 0 1
    p =   0.1946 -0.14695    -0.32
    """
    # ====================================================================

    # Simulate with a constant force applied to the foot:
    freq_hz = 100
    delta_t_sec, duration_sec, cur_t = 1 / freq_hz, 5, 0

    # Constant external force:
    force = pin.Force(np.array([2, 0, 0]), np.array([0, 0, 0]))

    # State variables:
    q, v = pin.neutral(robot.model), np.zeros(robot.nv)
    q_hist = [q.copy()]

    """
    # print(list(robot.model.frames))
    # calculate_fk(robot, q)
    # X_o_foot = robot.data.oMf[get_frame_id(robot, "FR_FOOT")]
    # X_o_knee = robot.data.oMf[get_frame_id(robot, "FL_UPPER_LEG")]

    # print("Neutral position, force:", force)
    # print("X_o_foot:", X_o_foot)
    # print("X_o_knee:", X_o_knee)
    # print("X_o_foot.act(force):", X_o_foot.act(force))
    # print("X_o_knee.actInv(X_o_foot.act(force)):", X_o_knee.actInv(X_o_foot.act(force)))

    # print()

    # q[0] = -np.pi / 2
    # calculate_fk(robot, q)
    # X_o_foot = robot.data.oMf[get_frame_id(robot, "FR_FOOT")]
    # X_o_knee = robot.data.oMf[get_frame_id(robot, "FL_UPPER_LEG")]        # <- THIS IS WRONG! This is after the joint has moved.
    #                                                                 # What is fext expecting?

    # print("-π position, force:", force)
    # print("X_o_foot:", X_o_foot)
    # print("X_o_knee:", X_o_knee)
    # print("X_o_foot.act(force):", X_o_foot.act(force))
    # print("X_o_knee.actInv(X_o_foot.act(force)):", X_o_knee.actInv(X_o_foot.act(force)))

    # print()

    # force_knee = X_o_knee.actInv(X_o_foot.act(force))


    # viz.displayFrames(True)
    # viz.display()

    # while True:
    #     pass
    """

    while cur_t < duration_sec:
        # Perform FK to find where the foot is. We'll calculate the external
        # force as expressed in the knee's frame:
        calculate_fk(robot, q)
        X_o_foot = robot.data.oMf[get_frame_id(robot, "FR_FOOT")]
        X_o_foot_world_aligned = pin.SE3(np.eye(3), X_o_foot.translation)

        X_o_knee = robot.data.oMf[get_frame_id(robot, "FR_KFE")]
        force_knee = X_o_knee.actInv(X_o_foot_world_aligned.act(force))

        # Run the FD algorithm with no joint torques for the knee.
        # Simulate damping via a negative torque:
        damping = np.array([-v[0] * 0.01])
        a = pin.aba(robot.model, robot.data, q, v, damping, [pin.Force(), force_knee])

        # Integrate:
        v += a * delta_t_sec
        q = pin.integrate(robot.model, q, v * delta_t_sec)

        q_hist.append(q.copy())
        cur_t += delta_t_sec
    

    input("Press ENTER to start animation...")

    for state in tqdm(q_hist):
        robot.display(state)
        time.sleep(delta_t_sec)
