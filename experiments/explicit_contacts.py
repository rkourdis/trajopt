import os
import time
import hppfcl
import pickle
import functools
from tqdm import tqdm
from typing import Callable
from dataclasses import dataclass

import numpy as np
import casadi as ca
import pinocchio as pin
import intervaltree as ivt
import matplotlib.pyplot as plt

def load_limited_solo12(free_joints, floor_z = 0.0, visualize = False):
    pkg_path = os.path.dirname(__file__)
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

    # # Add floor visual geometry:
    # floor_obj = pin.GeometryObject(
    #     "floor", 0, 0, hppfcl.Box(2, 2, 0.005),
    #     pin.SE3(np.eye(3), np.array([0, 0, floor_z]))
    # )

    # floor_obj.meshColor = np.array([0.3, 0.3, 0.3, 1])
    # visualizer.addGeometryObject(floor_obj)

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
    x_dd: float

@dataclass
class Input:
    t: float
    u: float

# Z-up force on robot foot:
@dataclass
class GRF:
    t: float
    λ: float

class FootholdKinematics(ca.Callback):
    def __init__(self, name: str, robot, feet: list[str], opts={}):
        ca.Callback.__init__(self)
        self.robot = robot
        self.ff_ids = [robot.model.getFrameId(f) for f in feet]
        self.construct(name, opts)

    def get_n_in(self): return 3    # q, v, a of robot (joint space)
    def get_sparsity_in(self, idx: int):
        return ca.Sparsity.dense(
            self.robot.nq if idx == 0 else self.robot.nv, 1
        )
    
    # foot frame positions (oMf), velocities, accelerations (in local world-aligned coords)
    def get_n_out(self): return 3
    def get_sparsity_out(self, _: int):
        return ca.Sparsity.dense(len(self.ff_ids), 3)
    
    def eval(self, arg):
        q, v, a = np.array(arg[0]), np.array(arg[1]), np.array(arg[2])

        # This runs the second-order FK algorithm and returns:
        # - The positions of all feet wrt the origin
        # - The velocities of all feet in the local world-aligned frame
        # - The accelerations of all feet in the local world-aligned frame

        pin.forwardKinematics(robot.model, robot.data, q, v, a)
        pin.updateFramePlacements(robot.model, robot.data)

        get_pos = lambda fid: robot.data.oMf[fid].translation
        get_vel = lambda fid: pin.getFrameVelocity(robot.model, robot.data, fid, pin.LOCAL_WORLD_ALIGNED).linear
        get_acc = lambda fid: pin.getFrameAcceleration(robot.model, robot.data, fid, pin.LOCAL_WORLD_ALIGNED).linear

        return (
            np.stack([get_pos(f) for f in self.ff_ids]),
            np.stack([get_vel(f) for f in self.ff_ids]),
            np.stack([get_acc(f) for f in self.ff_ids])
        )

class ForwardDynamics(ca.Callback):
  def __init__(self, name: str, robot, joints: list[str], feet: list[str], opts={}):
    ca.Callback.__init__(self)
    
    self.robot = robot
    self.joint_frame_ids = [robot.model.getFrameId(j) for j in joints]
    self.foot_frame_ids = [robot.model.getFrameId(f) for f in feet]

    self.construct(name, opts)

  # Number of inputs and outputs:
  def get_n_in(self): return 4
  def get_n_out(self): return 1

  # Evaluate numerically:
  def eval(self, arg):
    q, v, tau, λ = \
        np.array(arg[0]), np.array(arg[1]), np.array(arg[2]), np.array(arg[3])

    # λ contains normal GRFs for each foot.
    # Find how they're expressed in the joint frames at the provided
    # robot state. FK will populate robot.data.oMf.
    grf_at_joints = []

    pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
    pin.updateFramePlacements(self.robot.model, self.robot.data)

    for j_fr_id in self.joint_frame_ids:
        total_grf = pin.Force.Zero()
        X_o_joint = robot.data.oMf[j_fr_id]

        for f_idx, f_fr_id in enumerate(self.foot_frame_ids):
            X_o_foot = robot.data.oMf[f_fr_id]
            X_o_foot_world_aligned = pin.SE3(np.eye(3), X_o_foot.translation)

            # We assume the GRF always points Z-up (it's expressed in the world aligned frame)
            grf_at_foot = pin.Force(np.array([0, 0, λ[0][f_idx]]), np.zeros(3))

            # Express foot GRF in the joint frame and add up for all feet:
            total_grf += X_o_joint.actInv(X_o_foot_world_aligned.act(grf_at_foot))

        grf_at_joints.append(total_grf)

    # We'll calculate the unconstrained dynamics using the ABA algorithm.
    # The constraint forces will be chosen by the optimization so that they balance
    # the legs on contact, as described by the contact constraints.
    # In constrained FD, the constraint forces will be implicitly calculated and enforced,
    # but we want to choose them explicitly (otherwise we'll need to know whether we're in
    # contact or not in advance).
    # NOTE: Joints[0] is the 'universe' joint - do not apply any forces there.
    # This is strange because when you do .njoints you get = 1.
    return pin.aba(
        self.robot.model, self.robot.data, q, v, tau, [pin.Force.Zero(), *grf_at_joints]
    )

if __name__ == "__main__":
    OUTPUT_BIN = "trajectory_with_contact.bin"
    JOINTS = ["FR_KFE"]
    FEET = ["FR_FOOT"]

    FREQ_HZ = 200
    DELTA_T = 1 / FREQ_HZ
    FLOOR_Z = -0.3

    # TODO: Make sure the order of joints is preserved after robot creation.
    # It should be the same order that goes in aba().
    robot, viz = load_limited_solo12(
        free_joints=JOINTS, floor_z = FLOOR_Z, visualize = True
    )

    fd = ForwardDynamics("fd", robot, joints = JOINTS, feet = FEET, opts = {"enable_fd": True})
    fk = FootholdKinematics("fk", robot, FEET, opts = {"enable_fd": True})

    if not os.path.exists(OUTPUT_BIN):
        constraints = []

        # Create a new inequality constraint: expr >= 0
        def add_inequality(expr: ca.MX):
            constraints.append((expr, 0, np.inf))

        # Create a new equality constraint: expr == 0
        def add_equality(expr: ca.MX):
            constraints.append((expr, 0, 0))

        initial_state = State(0, np.deg2rad(40), 0, None) # At 40deg the foot is roughly at -0.28m
        final_state   = State(0.5, np.pi / 2, 0, None)
        duration = final_state.t - initial_state.t

        contact_times = ivt.IntervalTree([
            ivt.Interval(duration / 4, duration / 2)    # Contact for 125ms, then swing up
        ])

        N_knots = int((final_state.t - initial_state.t) * FREQ_HZ) + 1
        q_k, v_k, a_k, tau_k, λ_k, foot_z_k = [], [], [], [], [], []  # Collocation variables

        for k in range(N_knots):
            # Create decision variables at collocation points:
            q_k.append(ca.MX.sym(f"q_{k}"))
            v_k.append(ca.MX.sym(f"v_{k}"))
            a_k.append(ca.MX.sym(f"a_{k}"))
            tau_k.append(ca.MX.sym(f"τ_{k}"))

            λ_k.append(ca.MX.sym(f"λ_{k}"))
            foot_z_k.append(ca.MX.sym(f"foot_z_{k}"))

            #### DYNAMICS CONSTRAINTS ####
            # Residual constraints for accelerations (= 0) at all collocation points:
            add_equality(a_k[k] - fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))
            ##############################

            #### CONTACT CONSTRAINTS ####
            f_pos, _, f_acc = fk(q_k[k], v_k[k], a_k[k])
            add_equality(foot_z_k[k] - f_pos[2])          # foot_z[k] should be the foot height

            if contact_times.overlaps(k * DELTA_T):
                add_equality(foot_z_k[k] - FLOOR_Z)          # Foot should be stable on the floor
                add_inequality(λ_k[k])                       # Contact forces available
                
                # Foot Z acceleration should be zero - this is to avoid issues with the optimizer
                # tricking the integration scheme and applying oscillating torques and GRFs while
                # on the floor (such that foot_z = 0 after trapezoidal integration...)
                add_equality(f_acc[2])
            else:
                add_inequality(foot_z_k[k] - FLOOR_Z)        # TODO: Change this to strict inequality?
                add_equality(λ_k[k])                         # Contact forces unavailable
            #############################

            #### JOINT LIMITS ####
            # # NOTE: The solver fails without these, why?
            # add_inequality(tau_k[k] + 4)
            # add_inequality(-tau_k[k] + 4)
            ######################

            # We'll add integration constraints for all knots, wrt their previous points:
            if k == 0:
                continue
            
            #### INTEGRATION CONSTRAINTS ####
            # Velocities - trapezoidal integration:
            # v_k[k] = v_k[k - 1] + 1/2 * Δt * (a_k[k] + a_k[k-1])
            add_equality(v_k[k] - v_k[k-1] - 0.5 * DELTA_T * (a_k[k] + a_k[k-1]))

            # Same for positions:
            add_equality(q_k[k] - q_k[k-1] - 0.5 * DELTA_T * (v_k[k] + v_k[k-1]))
            ##################################

        # Create optimization objective - min(Integrate[τ^2[t], {t, 0, T}]).
        # Use trapezoidal integration to approximate:
        obj = sum(0.5 * DELTA_T * (tau_k[idx]**2 + tau_k[idx+1]**2) for idx in range(N_knots-1))

        #### BOUNDARY CONSTRAINTS ####
        add_equality(q_k[0] - initial_state.x)      # Initial q
        add_equality(q_k[-1] - final_state.x)       # Final q
        add_equality(v_k[0] - initial_state.x_d)    # Initial v
        add_equality(v_k[-1] - final_state.x_d)     # Final v
        ###############################

        # Create the NLP problem:
        g, lbg, ubg = zip(*constraints)
        nlp = {
            "x": ca.vertcat(*q_k, *v_k, *a_k, *tau_k, *λ_k, *foot_z_k),
            "f": obj,
            "g": ca.vertcat(*g)
        }

        ipopt_settings = {"linear_solver": "ma57"}
        solver = ca.nlpsol("S", "ipopt", nlp, {"ipopt": ipopt_settings})

        #### INITIAL GUESS ####
        # Assume that we'll be falling for the first 1/4 of the trajectory.
        # Then, we'll have contact to the ground so we'll say that the foot's on the ground.
        # Then we'll slowly climb up.
        min_leg_z, max_leg_z = float(fk(0, 0, 0)[0][2]), float(fk(np.pi/2, 0, 0)[0][2])
        contact_leg_angle = np.arccos((max_leg_z - FLOOR_Z) / (max_leg_z - min_leg_z))

        fall_velocity = (contact_leg_angle - initial_state.x) / (duration / 4)
        climb_velocity = (final_state.x - contact_leg_angle) / (duration / 2)

        q_g, v_g, a_g, τ_g, λ_g, fz_g = [], [], [], [], [], []

        for idx in range(N_knots):
            t = idx * DELTA_T

            if t < duration / 4:
                q_g.append(initial_state.x + fall_velocity * t)
                v_g.append(fall_velocity)

            elif t < duration / 2:
                q_g.append(contact_leg_angle)
                v_g.append(0)

            else:
                q_g.append(contact_leg_angle + climb_velocity * (t - duration / 2))
                v_g.append(climb_velocity)

            a_g.append(0)
            τ_g.append(0)
            λ_g.append(0)
            fz_g.append(float(fk(q_g[-1], 0, 0)[0][2]))
        ########################

        # Solve the problem!
        soln = solver(
            x0 = [*q_g, *v_g, *a_g, *τ_g, *λ_g, *fz_g],
            lbg = ca.vertcat(*lbg),
            ubg = ca.vertcat(*ubg)
        )["x"]
        
        if not solver.stats()["success"]:
            print("Solver failed to find solution. Exitting...")
            exit()

        # Extract variables from the solution:
        trajectory, torques, grfs, foot_zs = [], [], [], []
        
        for idx in range(N_knots):
            t = DELTA_T*idx
            q = float(soln[idx])
            v = float(soln[N_knots + idx])
            a = float(soln[2 * N_knots + idx])
            τ = float(soln[3 * N_knots + idx])
            λ = float(soln[4 * N_knots + idx])
            foot_z = float(soln[5 * N_knots + idx])

            trajectory.append(State(t, q, v, a))
            torques.append(Input(t, τ))
            grfs.append(GRF(t, λ))
            foot_zs.append((t, foot_z))

        with open(OUTPUT_BIN, "wb") as wf:
            pickle.dump((trajectory, torques, grfs, foot_zs), wf)

    with open(OUTPUT_BIN, "rb") as rf:
        trajectory, torques, grfs, foot_zs = pickle.load(rf)

    ###################################################
    plt.scatter([x.t for x in trajectory], np.zeros(len(trajectory)), label = "collocation points")
    plt.plot([λ.t for λ in grfs], [λ.λ for λ in grfs], label = "z-normal reaction force")
    plt.plot([τ.t for τ in torques], [τ.u for τ in torques], label = "torques")
    plt.plot([z[0] for z in foot_zs], [z[1] for z in foot_zs], label = "foot_zs")
    plt.plot([x.t for x in trajectory], [x.x_dd for x in trajectory], label = "joint accelerations")
    plt.plot([x.t for x in trajectory], [x.x_d for x in trajectory], label = "joint velocities")

    plt.legend()
    plt.show()
    ###################################################
    input("Press ENTER to play trajectory...")

    for state in tqdm(trajectory):
        robot.display(np.array([state.x]))
        time.sleep(DELTA_T*3)