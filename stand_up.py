import os
import time
import hppfcl
import pickle
import functools
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass

import numpy as np
import casadi as ca
import pinocchio as pin
import intervaltree as ivt
import matplotlib.pyplot as plt

#region Structs
@dataclass
class State:
    # Second order dynamics:
    #   a = f(t, q, v, τ, λ)
    t: float
    q: np.array                     # 19x
    v: np.array                     # 18x
    a: Optional[np.array] = None    # For logging purposes

@dataclass
class Input:
    t: float
    τ: np.array                     # One for each actuated joint (12x)

# Z-up force on robot feet:
@dataclass
class GRFs:
    t: float
    λ: np.array                     # One for each foot (4x)
#endregion

def load_solo12(floor_z = 0.0, visualize = False):
    pkg_path = "C:\\Users\\rafael\\Projects\\Quadruped\\optimization\\"
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

    # Add floor visual geometry:
    floor_obj = pin.GeometryObject(
        "floor", 0, 0, hppfcl.Box(2, 2, 0.005),
        pin.SE3(np.eye(3), np.array([0, 0, floor_z]))
    )

    floor_obj.meshColor = np.array([0.3, 0.3, 0.3, 1])
    visualizer.addGeometryObject(floor_obj)

    robot.display(pin.neutral(robot.model))
    return robot, visualizer

def create_joint_vector(robot: pin.RobotWrapper, joint_angles: dict[str, float]):
    q = pin.neutral(robot.model)

    for j_name, angle in joint_angles.items():
        idx = robot.model.getJointId(j_name)
        q[robot.model.joints[idx].idx_q] = angle

    return np.expand_dims(q, axis = -1)     # 19x1

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

class StateIntegrator(ca.Callback):
    def __init__(self, name: str, robot: pin.RobotWrapper, opts = {}):
        ca.Callback.__init__(self)
        self.robot = robot
        self.construct(name, opts)
    
    def get_n_in(self): return 2
    def get_sparsity_in(self, idx: int):
        return [
            ca.Sparsity.dense(self.robot.nq, 1),
            ca.Sparsity.dense(self.robot.nv, 1)
        ][idx]

    def get_n_out(self): return 1
    def get_sparsity_out(self, _: int):
        return ca.Sparsity.dense(self.robot.nq, 1)
    
    def eval(self, arg):
        q, v = np.array(arg[0]), np.array(arg[1])
        q_new = pin.integrate(robot.model, q, v)
        return [np.expand_dims(q_new, axis = -1)]

class ForwardDynamics(ca.Callback):
    def __init__(self, name: str, robot: pin.RobotWrapper, act_joint_ids: list[int], feet: list[str], opts={}):
        ca.Callback.__init__(self)
        self.robot = robot

        # Actuated joint IDs (in robot.model.joints):
        self.act_joint_ids = act_joint_ids

        # Get frame IDs for actuated joints. We'll use these to express GRFs
        # in the correct frames:
        self.act_j_frame_ids = [
            robot.model.getFrameId(robot.model.names[j])
            for j in self.act_joint_ids
        ]

        self.foot_frame_ids = [robot.model.getFrameId(f) for f in feet]
        self.construct(name, opts)

    def get_n_in(self): return 4
    def get_sparsity_in(self, idx: int):
        return [
            ca.Sparsity.dense(self.robot.nq, 1),            # q (19 x 1)
            ca.Sparsity.dense(self.robot.nv, 1),            # v (18 x 1)
            ca.Sparsity.dense(len(self.act_joint_ids), 1),  # τ (12 x 1)
            ca.Sparsity.dense(len(self.foot_frame_ids), 1)  # λ (4  x 1), z-up
        ][idx]

    def get_n_out(self): return 1
    def get_sparsity_out(self, _: int):
        return ca.Sparsity.dense(self.robot.nv, 1)          # (18 x 1)

    def eval(self, arg):
        q, v, tau_act, λ = \
            np.array(arg[0]), np.array(arg[1]), np.array(arg[2]), np.array(arg[3])
        
        # λ contains normal GRFs for each foot.
        # Find how they're expressed in all actuated joint frames at
        # the provided robot state. FK will populate robot.data.oMf.
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)

        # We assume the GRF always points Z-up (expressed in the world aligned frame):
        X_o_feet_world_aligned = [
            pin.SE3(np.eye(3), robot.data.oMf[f_fr_id].translation)
            for f_fr_id in self.foot_frame_ids
        ]

        grf_at_joints = []

        # For each actuated joint's frame:
        for j_fr_id in self.act_j_frame_ids:
            total_grf = pin.Force()
            X_o_joint = robot.data.oMf[j_fr_id]

            # For each foot:
            for foot_idx in range(len(self.foot_frame_ids)):
                grf_at_foot = pin.Force(np.array([0, 0, λ[foot_idx][0]]), np.zeros(3))

                # Express foot GRF in the joint frame and add up for all feet:
                total_grf += X_o_joint.actInv(
                    X_o_feet_world_aligned[foot_idx].act(grf_at_foot)
                )

            grf_at_joints.append(total_grf)

        # We'll calculate the unconstrained dynamics using the ABA algorithm.
        # The constraint forces will be chosen by the optimization so that they balance
        # the legs on contact, as described by the contact constraints.
        # In constrained FD, the constraint forces will be implicitly calculated and enforced,
        # but we want to choose them explicitly (otherwise we'll need to know whether we're in
        # contact or not in advance).
        # NOTE: We'll skip all unactuated joints when applying torques, and external forces.

        # Each actuated joint is one degree of freedom. Create a robot.nv x 1
        # torque vector with only the actuated DoFs set:
        tau_full = np.zeros((robot.nv, 1))
        for j_id, τ in zip(self.act_joint_ids, tau_act):
            tau_full[robot.model.joints[j_id].idx_v] = τ

        # Add the unactuated joints to the full fext vector:
        fext_full = [pin.Force() for _ in range(robot.model.njoints)]
        for j_id, force in zip(self.act_joint_ids, grf_at_joints):
            fext_full[j_id] = force

        # Run the articulated body algorithm:
        accel = pin.aba(self.robot.model, self.robot.data, q, v, tau_full, fext_full)

        # Convert into robot.nv x 1 matrix:
        return [np.expand_dims(accel, axis = -1)]

if __name__ == "__main__":
    OUTPUT_BIN = "trajectory_stand.bin"

    # The order here does NOT correspond to the order of joints
    # in the model, or the actuated_joints list below!
    FOLDED_JOINT_MAP = {
        "FR_HAA": 0,
        "FL_HAA": 0,
        "HR_HAA": 0,
        "HL_HAA": 0,
        "FR_KFE": -np.pi,
        "FR_HFE": np.pi / 2,
        "FL_KFE": -np.pi,
        "FL_HFE": np.pi / 2,
        "HR_KFE": np.pi,
        "HR_HFE": -np.pi / 2,
        "HL_KFE": np.pi,
        "HL_HFE": -np.pi / 2,
    }

    # The order of forces per foot WILL be in this order:
    FEET = ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]

    FREQ_HZ = 1
    DELTA_T = 1 / FREQ_HZ
    FLOOR_Z = -0.3
    
    robot, viz = load_solo12(floor_z = FLOOR_Z, visualize = False)

    # Start with the robot folded:
    q0 = create_joint_vector(robot, FOLDED_JOINT_MAP)
    v0 = np.zeros((robot.nv, 1))

    # Skip 'universe' and 'root_joint' as unactuated joints:
    actuated_joints = [j.id for j in robot.model.joints[2:]]

    fd = ForwardDynamics("fd", robot, actuated_joints, feet = FEET, opts = {"enable_fd": True}) #, "jit": True})
    fk = FootholdKinematics("fk", robot, FEET, opts = {"enable_fd": True}) #, "jit": True})
    si = StateIntegrator("si", robot, opts = {"enable_fd": True}) #, "jit": True})

    # import cProfile
    # with cProfile.Profile() as prof:
    #     for _ in range(1000):
    #         a = fd(q0, v0, np.zeros((len(actuated_joints), 1)), np.zeros((len(FEET), 1)))
    #     prof.dump_stats("fd_1000_calls.prof")

    # exit()

    if not os.path.exists(OUTPUT_BIN):
        q_k, v_k, a_k, tau_k, λ_k, foot_z_k = [], [], [], [], [], []  # Collocation variables

        #region Constraint utilities
        constraints = []

        # Create a new elementwise inequality constraint: expr[:] >= 0
        def add_inequality(expr: ca.MX):
            constraints.append((expr, np.zeros(expr.shape), np.full(expr.shape, np.inf)))

        # Create a new elementwise equality constraint: expr[:] == 0
        def add_equality(expr: ca.MX):
            constraints.append((expr, np.zeros(expr.shape), np.zeros(expr.shape)))
        #endregion

        #region Task parameters

        # Trajectory duration - the task will be to keep the body CoM at a given
        # height as well as possible for the entirety of this duration:
        duration = 1              
        initial_state = State(0, q0, v0)

        # Contact times per foot, we'll assume the robot can fall for ~0.15m 
        # (half its initial height) before the feet are on the ground:
        # contact_times = [ivt.IntervalTree([ivt.Interval(0.17, duration)]) for _ in FEET]
        contact_times = [ivt.IntervalTree([ivt.Interval(0.0, duration)]) for _ in FEET]

        # Trajectory error function. Given t, q, v, a, τ at a collocation point
        # it returns how far away the trajectory is from the desired one at that
        # time:
        def traj_err(t, q, v, a, τ):
            DESIRED_COM_Z = -0.1
            return (q[2] - DESIRED_COM_Z) ** 2

        #endregion

        N_knots = int(duration * FREQ_HZ) + 1
        print("Creating variables and constraints...")

        for k in range(N_knots):
            #region Collocation variable creation

            # Create decision variables at collocation points:
            q_k.append(ca.MX.sym(f"q_{k}", robot.nq))   # 19 x 1
            v_k.append(ca.MX.sym(f"v_{k}", robot.nv))   # 18 x 1
            a_k.append(ca.MX.sym(f"a_{k}", robot.nv))   # 18 x 1

            tau_k.append(ca.MX.sym(f"τ_{k}", len(actuated_joints)))  # 12 x 1 
            λ_k.append(ca.MX.sym(f"λ_{k}", len(FEET)))               # 4  x 1
            foot_z_k.append(ca.MX.sym(f"foot_z_{k}", len(FEET)))     # 4  x 1
            
            #endregion

            #region Pointwise constraints (accelerations, contact, limits)

            #### DYNAMICS CONSTRAINTS ####
            # Residual constraints for accelerations (= 0) at all collocation points:
            add_equality(a_k[k] - fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))
            ##############################

            #### CONTACT CONSTRAINTS ####
            f_pos, _, f_acc = fk(q_k[k], v_k[k], a_k[k])
            add_equality(foot_z_k[k] - f_pos[:, 2])          # foot_z[k] should contain all foot heights

            # Enforce constraints for all feet:
            for idx, foot in enumerate(FEET):

                # If that particular foot is in contact with the ground:
                if contact_times[idx].overlaps(k * DELTA_T):
                    add_equality(foot_z_k[k][idx] - FLOOR_Z)     # Foot should be stable on the floor
                    add_inequality(λ_k[k][idx])                  # Contact forces available
                    
                    # Foot Z acceleration should be zero - this is to avoid issues with the optimizer
                    # tricking the integration scheme and applying oscillating torques and GRFs while
                    # on the floor (such that foot_z = 0 after trapezoidal integration...)
                    add_equality(f_acc[idx, 2])
                else:
                    add_inequality(foot_z_k[k][idx] - FLOOR_Z)   # TODO: Change this to strict inequality?
                    add_equality(λ_k[k][idx])                    # Contact forces unavailable
            #############################

            #### JOINT LIMITS ####
            # TODO.
            ######################

            #endregion

            # We'll add integration constraints for all knots, wrt their previous points:
            if k == 0:
                continue

            #region Integration constraints (residuals)

            #### INTEGRATION CONSTRAINTS ####
            # Velocities - trapezoidal integration:
            # v_k[k] = v_k[k - 1] + 1/2 * Δt * (a_k[k] + a_k[k-1])
            add_equality(v_k[k] - v_k[k-1] - 0.5 * DELTA_T * (a_k[k] + a_k[k-1]))

            # Use pin.integrate() to integrate the positions. Orientations
            # can't be integrated like velocities:
            add_equality(q_k[k] - si(q_k[k-1], 0.5 * DELTA_T * (v_k[k] + v_k[k-1])))
            ##################################

            #endregion
        
        #region Objective and boundaries

        print("Creating optimization objective...")

        # Create optimization objective - hold the body frame's Z as constant as possible.
        # Integrate the error over time, use trapezoidal approximation:
        err = lambda q: traj_err(None, q, None, None, None)
        obj = sum(0.5 * DELTA_T * (err(q_k[idx]) + err(q_k[idx+1])) for idx in range(N_knots-1))

        print("Adding boundary constraints...")

        #### BOUNDARY CONSTRAINTS ####
        add_equality(q_k[0] - initial_state.q)      # Initial q
        add_equality(v_k[0] - initial_state.v)    # Initial v
        ###############################

        #endregion

        #region NLP setup
        
        print("Creating NLP description...")

        # Create the NLP problem:
        g, lbg, ubg = zip(*constraints)

        nlp = {
            "x": ca.vertcat(*q_k, *v_k, *a_k, *tau_k, *λ_k, *foot_z_k),
            "f": obj,
            "g": ca.vertcat(*g)
        }

        print("Instantiating solver...")
        solver = ca.nlpsol("S", "ipopt", nlp)

        #endregion

        #region Initial guess

        #### INITIAL GUESS ####

        print("Creating initial guesses...")

        # Assume that nothing changes as I'm very lazy:
        q_g  = [initial_state.q for _ in range(N_knots)]
        v_g  = [initial_state.v for _ in range(N_knots)]
        a_g  = [np.zeros((robot.nv, 1)) for _ in range(N_knots)]
        τ_g  = [np.zeros((len(actuated_joints), 1)) for _ in range(N_knots)]
        λ_g  = [np.zeros((len(FEET), 1)) for _ in range(N_knots)]
        fz_g = [np.array(fk(q_g[0], v_g[0], a_g[0])[0][:, 2]) for _ in range(N_knots)]
        ########################

        #endregion

        # Solve the problem!
        soln = solver(
            x0  = ca.vertcat(*q_g, *v_g, *a_g, *τ_g, *λ_g, *fz_g),
            lbg = ca.vertcat(*lbg),
            ubg = ca.vertcat(*ubg)
        )["x"]
        
        if not solver.stats()["success"]:
            print("Solver failed to find solution. Exitting...")
            exit()

        # # Extract variables from the solution:
        # trajectory, torques, grfs, foot_zs = [], [], [], []
        
        # for idx in range(N_knots):
        #     t = DELTA_T*idx
        #     q = float(soln[idx])
        #     v = float(soln[N_knots + idx])
        #     a = float(soln[2 * N_knots + idx])
        #     τ = float(soln[3 * N_knots + idx])
        #     λ = float(soln[4 * N_knots + idx])
        #     foot_z = float(soln[5 * N_knots + idx])

        #     trajectory.append(State(t, q, v, a))
        #     torques.append(Input(t, τ))
        #     grfs.append(GRF(t, λ))
        #     foot_zs.append((t, foot_z))

        # with open(OUTPUT_BIN, "wb") as wf:
        #     pickle.dump((trajectory, torques, grfs, foot_zs), wf)

    # with open(OUTPUT_BIN, "rb") as rf:
    #     trajectory, torques, grfs, foot_zs = pickle.load(rf)

    # ###################################################
    # plt.scatter([x.t for x in trajectory], np.zeros(len(trajectory)), label = "collocation points")
    # plt.plot([λ.t for λ in grfs], [λ.λ for λ in grfs], label = "z-normal reaction force")
    # plt.plot([τ.t for τ in torques], [τ.u for τ in torques], label = "torques")
    # plt.plot([z[0] for z in foot_zs], [z[1] for z in foot_zs], label = "foot_zs")
    # plt.plot([x.t for x in trajectory], [x.x_dd for x in trajectory], label = "joint accelerations")
    # plt.plot([x.t for x in trajectory], [x.x_d for x in trajectory], label = "joint velocities")

    # plt.legend()
    # plt.show()
    # ###################################################
    # input("Press ENTER to play trajectory...")

    # for state in tqdm(trajectory):
    #     robot.display(np.array([state.x]))
    #     time.sleep(DELTA_T*3)
