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
import matplotlib.pyplot as plt

def load_limited_solo12(free_joints, floor_z = 0.0, visualize = False):
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

    # Add floor visual geometry:
    floor_obj = pin.GeometryObject(
        "floor", 0, 0, hppfcl.Box(2, 2, 0.005),
        pin.SE3(np.eye(3), np.array([0, 0, floor_z]))
    )

    floor_obj.meshColor = np.array([0.3, 0.3, 0.3, 1])
    visualizer.addGeometryObject(floor_obj)

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
        self.foot_frame_ids = [robot.model.getFrameId(f) for f in feet]
        self.construct(name, opts)

    def get_n_in(self): return 1

    # Returns a tuple of 1x1 DMs for multiple feet:
    def get_n_out(self): return len(self.foot_frame_ids)

    def eval(self, arg):
        q = np.array(arg[0])    # np.array(ca.DM of size 1x1)

        # This runs the FK algorithm and returns the Z-heights of all feet
        # at a given state. Heights are in the origin frame.
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        return np.array([
            self.robot.data.oMf[fid].translation[2]
            for fid in self.foot_frame_ids
        ])

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
        total_grf = pin.Force()
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
        self.robot.model, self.robot.data, q, v, tau, [pin.Force(), *grf_at_joints]
    )

if __name__ == "__main__":
    OUTPUT_BIN = "trajectory_with_contact.bin"
    JOINTS = ["FR_KFE"]
    FEET = ["FR_FOOT"]

    FREQ_HZ = 100
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
        # NOTE: Changing the foot_z - FLOOR_Z >= 0 constraint
        #       below to foot_z >= FLOOR_Z makes the solver unable
        #       to optimize for some reason...
        def add_inequality(expr: ca.MX):
            constraints.append((expr, 0, np.inf))

        # Create a new equality constraint: expr == 0
        def add_equality(expr: ca.MX):
            constraints.append((expr, 0, 0))

        initial_state = State(0, np.deg2rad(40), 0) # At 40deg the foot is roughly at -0.28m
        final_state   = State(0.5, np.pi / 2, 0)      

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
            # TODO: Do for all feet, define multiple λ_k.
            add_equality(foot_z_k[k] - fk(q_k[k])[0])        # foot_z[k] should be the foot height
            add_inequality(foot_z_k[k] - FLOOR_Z)            # No penetration of feet
            add_inequality(λ_k[k])                           # No attractive forces

            # add_equality((foot_z_k[k] - FLOOR_Z) * λ_k[k])   # Complementarity - no force at a distance

            # "Soft" LCP:
            # TODO: See Posa '13 for techniques on how to solve this problem. It's difficult.
            # TODO: Change to a contact explicit scheme - it would be the same, but for the 
            # times when there's contact you have to FORCE foot_z == GROUND.
            # Then, you can also describe the friction cone dynamics and the fact that there
            # MUST NOT be any slipping, this will choose τ such that the normal force
            # is enough for that (if feasible).
            add_inequality((foot_z_k[k] - FLOOR_Z) * λ_k[k]) # >= 0
            add_inequality(-(foot_z_k[k] - FLOOR_Z) * λ_k[k] + 1e-3) # <= 1e-3
            #############################

            #### JOINT LIMITS ####
            # NOTE: The solver fails without these, why?
            add_inequality(tau_k[k] + 4)
            add_inequality(-tau_k[k] + 4)
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

        solver = ca.nlpsol("S", "ipopt", nlp)

        #### INITIAL GUESS ####
        # Assume foot falls for 1/4 of the time, stays at 0 for 1/4
        # then goes up:
        traj_dur = final_state.t - initial_state.t

        fall_vel = (initial_state.x - 0) / (traj_dur / 4)
        rise_vel = (final_state.x - 0) / (traj_dur / 2)

        q_guess = [
            *[initial_state.x - fall_vel * idx * DELTA_T for idx in range(N_knots // 4)],
            *[0 for _ in range(N_knots // 4)],
            *[rise_vel * idx * DELTA_T for idx in range(N_knots // 2)],
            *[final_state.x for _ in range(N_knots - N_knots // 2 - 2 * (N_knots // 4))]
        ]

        v_guess = [
            *[-fall_vel for _ in range(N_knots // 4)],
            *[0 for _ in range(N_knots // 4)],
            *[rise_vel for _ in range(N_knots // 2)],
            *[0 for _ in range(N_knots - N_knots // 2 - 2 * (N_knots // 4))]
        ]

        a_guess = [0 for _ in range(N_knots)]
        tau_guess = list(0 for _ in range(N_knots))
        
        foot_z_guess = [float(fk([q])[0]) for q in q_guess]

        λ_guess = [
            *[0 for _ in range(N_knots // 4)],
            *[1 for _ in range(N_knots // 4)],
            *[0 for _ in range(N_knots - 2 * (N_knots // 4))],
        ]

        ########################

        # Solve the problem!
        soln = solver(
            x0 = [*q_guess, *v_guess, *a_guess, *tau_guess, *λ_guess, *foot_z_guess],
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
            τ = float(soln[3 * N_knots + idx])
            λ = float(soln[4 * N_knots + idx])
            foot_z = float(soln[5 * N_knots + idx])

            trajectory.append(State(t, q, v))
            torques.append(Input(t, τ))
            grfs.append(GRF(t, λ))
            foot_zs.append((t, foot_z))

        with open(OUTPUT_BIN, "wb") as wf:
            pickle.dump((trajectory, torques, grfs, foot_zs), wf)

    with open(OUTPUT_BIN, "rb") as rf:
        trajectory, torques, grfs, foot_zs = pickle.load(rf)

    ###################################################
    plt.plot([λ.t for λ in grfs], [λ.λ for λ in grfs], label = "z-normal reaction force")
    plt.plot([τ.t for τ in torques], [τ.u for τ in torques], label = "torques")
    plt.plot([z[0] for z in foot_zs], [z[1] for z in foot_zs], label = "foot_zs")
    plt.legend()
    plt.show()
    ###################################################
    input("Press ENTER to play trajectory...")

    for state in tqdm(trajectory):
        robot.display(np.array([state.x]))
        time.sleep(DELTA_T*3)
