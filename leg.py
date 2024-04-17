import os
import sys
import time
import pickle
import functools
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

    if visualize:
        limited_robot.setVisualizer(
            pin.visualize.MeshcatVisualizer(
                limited_robot.model, limited_robot.collision_model, limited_robot.visual_model
            )
        )
        
        limited_robot.initViewer()
        limited_robot.loadViewerModel()
        limited_robot.display(pin.neutral(limited_robot.model))

    return limited_robot

@functools.cache
def get_joint_offset(joint: str, size_getter: Callable[[pin.JointModel], int], robot) -> int:
    joint_id = robot.model.getJointId(joint)
    return sum(size_getter(robot.model.joints[idx]) for idx in range(joint_id - 1))

def get_q_off(joint: str, robot) -> int:
    return get_joint_offset(joint, lambda j: j.nq, robot)

def get_v_off(joint: str, robot) -> int:
    return get_joint_offset(joint, lambda j: j.nv, robot)

@dataclass
class State:
    # Second order dynamics:
    #    x_dd = f(t, x, x_d, τ)
    #
    # x_d tells you how x evolves
    # x_dd tells you how x_d evolves, is dependent on x, x_d, τ
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

class ForwardDynamics(ca.Callback):
  def __init__(self, name: str, robot, opts={}):
    ca.Callback.__init__(self)
    self.robot = robot
    self.construct(name, opts)

  # Number of inputs and outputs:
  def get_n_in(self): return 3
  def get_n_out(self): return 1

  # Evaluate numerically:
  def eval(self, arg):
    # TODO: How to enable finite differences?
    q, v, tau = np.array(arg[0]), np.array(arg[1]), np.array(arg[2])

    return pin.aba(
        self.robot.model, self.robot.data, q, v, tau
    )

if __name__ == "__main__":
    OUTPUT_BIN = "trajectory_no_accel_bc.bin"

    robot = load_limited_solo12(visualize = True)
    fd = ForwardDynamics("fd", robot, opts = {"enable_fd": True})
    
    freq = 100
    delta_t = 1 / freq
    
    # q0, v0, tau0 = pin.neutral(robot.model), pin.utils.zero(robot.nv), [0.1]
    # q0 = ca.MX.sym("q0", 1)
    # v0 = ca.MX.sym("v0", 1)
    # tau0 = ca.MX.sym("tau0", 1)

    # J = ca.jacobian(fd(q0, v0, tau0), q0)
    # J_func = ca.Function('J_func', [q0, v0, tau0], [J])
    # J_numerical = J_func([0], [0], [0])

    if not os.path.exists(OUTPUT_BIN):
        # We'll optimize one dimension for now, FR_KFE:
        initial_state = State(0, 0, 0)
        final_state   = State(3, np.pi / 2, 0)

        N_knots = (final_state.t - initial_state.t) * freq + 1

        q_k, v_k, a_k, tau_k = [], [], [], []
        g_i = []

        for k in range(N_knots):
            # Create decision variables at collocation points:
            q_k.append(ca.MX.sym(f"q_{k}"))
            v_k.append(ca.MX.sym(f"v_{k}"))
            a_k.append(ca.MX.sym(f"a_{k}"))
            tau_k.append(ca.MX.sym(f"τ_{k}"))

            # CasADi bounds the g vector between the provided limits. We'll use
            # this to bound all below expressions between zero and zero, effectively
            # creating equality constraints.

            # Ask Pinocchio to calculate the accelerations from the robot dynamics 
            # and the torques (FD). Add these as constraints for all knots:
            g_i.append(a_k[k] - fd(q_k[k], v_k[k], tau_k[k]))

            # We'll add integration constraints for all knots.
            # Current ones wrt previous ones:
            if k == 0:
                continue    
            
            # Create dynamics constraint for velocities (using trapezoidal integration):
            # v_k[k] = v_k[k - 1] + 1/2 * Δt * (a_k[k] + a_k[k-1])
            g_i.append(v_k[k] - v_k[k-1] - 0.5 * delta_t * (a_k[k] + a_k[k-1]))

            # Same for positions:
            g_i.append(q_k[k] - q_k[k-1] - 0.5 * delta_t * (v_k[k] + v_k[k-1]))

        # Create optimization objective:
        obj = sum(0.5 * delta_t * (tau_k[idx]**2 + tau_k[idx+1]**2) for idx in range(N_knots-1))

        # Add equality constraint for trajectory boundaries.
        g_i.append(q_k[0] - initial_state.x)
        g_i.append(q_k[-1] - final_state.x)
        g_i.append(v_k[0] - initial_state.x_d)
        g_i.append(v_k[-1] - final_state.x_d)

        g_i.append(a_k[-1])          # Stabilize at the end

        # Create the NLP problem:
        nlp = {
            "x": ca.vertcat(*q_k, *v_k, *a_k, *tau_k),
            "f": obj,
            "g": ca.vertcat(*g_i)
        }

        solver = ca.nlpsol("S", "ipopt", nlp)

        # Construct initial guess for all decision variables:
        x0 = []

        const_velocity = (final_state.x - initial_state.x) / (final_state.t - initial_state.t)
        x0 += list(const_velocity * delta_t * idx for idx in range(N_knots))
        x0 += list(const_velocity for _ in range(N_knots))
        x0 += list(0 for _ in range(N_knots))                                # No acceleration
        x0 += list(0 for _ in range(N_knots))                                # No torque

        soln = solver(x0 = x0, lbg = 0, ubg = 0)
        
        traj = Trajectory(path = [
            State(t = idx * delta_t, x = float(soln["x"][idx]), x_d = float(soln["x"][N_knots + idx]))
            for idx in range(N_knots)
        ])
   
        torques = [
            Input(idx * delta_t, float(soln["x"][idx]))
            for idx in range(3 * N_knots, 4 * N_knots)
        ]

        with open(OUTPUT_BIN, "wb") as wf:
            pickle.dump((traj, torques), wf)

    with open(OUTPUT_BIN, "rb") as rf:
        traj, torques = pickle.load(rf)

    # Simulate dynamics using Pinocchio and the optimized torques:
    q_sim = [pin.neutral(robot.model)]
    v_sim = [pin.utils.zero(robot.nv)]

    for idx in range(len(torques)):
        accel = pin.aba(robot.model, robot.data, q_sim[-1], v_sim[-1], np.array([torques[idx].u]))

        # Integrate the stuff:
        v_sim.append(v_sim[-1] + accel * delta_t)
        q_sim.append(pin.integrate(robot.model, q_sim[-1], v_sim[-1]*delta_t))
    
    #################################################################

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    times = [delta_t * k for k in range(len(torques))]

    ax1.plot(times, q_sim[:-1], label = "q (rad), simulated")
    ax1.plot(times, [pt.x for pt in traj.path], label = "q (rad), optimized")

    ax2.plot(times, v_sim[:-1], label = "v (rad/s), simulated")
    ax2.plot(times, [pt.x_d for pt in traj.path], label = "v (rad/s), optimized")

    ax3.plot(times, [tk.u for tk in torques], label = "a (N*m), optimized")

    fig.legend()
    fig.show()

    #################################################################

    input("Press ENTER to start animation...")

    cur_idx = 0

    while cur_idx < len(q):
        robot.display(q[cur_idx])
        cur_idx += 1
        time.sleep(delta_t)
