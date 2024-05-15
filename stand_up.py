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
from pinocchio import casadi as cpin

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

    # Add floor visual geometry:
    floor_obj = pin.GeometryObject("floor", 0, 0, hppfcl.Box(2, 2, 0.005), pin.SE3.Identity())
    visualizer.loadViewerGeometryObject(floor_obj, pin.GeometryType.VISUAL, np.array([0.3, 0.3, 0.3, 1]))
    
    floor_obj_name = visualizer.getViewerNodeName(floor_obj, pin.GeometryType.VISUAL)

    # Manually set the transform because the GeometryObject() constructor doesn't work:
    visualizer.viewer[floor_obj_name].set_transform(
        pin.SE3(np.eye(3), np.array([0, 0, floor_z])).homogeneous
    )

    robot.display(pin.neutral(robot.model))
    return robot, visualizer

def create_joint_vector(robot: pin.RobotWrapper, joint_angles: dict[str, float]):
    q = pin.neutral(robot.model)

    for j_name, angle in joint_angles.items():
        idx = robot.model.getJointId(j_name)
        q[robot.model.joints[idx].idx_q] = angle

    return np.expand_dims(q, axis = -1)     # 19x1

class ADFootholdKinematics():
    def __init__(self, cmodel, cdata, feet: list[str]):
        self.cmodel, self.cdata = cmodel, cdata
        self.ff_ids = [robot.model.getFrameId(f) for f in feet]

    def __call__(self, q: ca.SX, v: ca.SX, a: ca.SX):
        # This runs the second-order FK algorithm and returns:
        # - The positions of all feet wrt the origin
        # - The velocities of all feet in the local world-aligned frame
        # - The accelerations of all feet in the local world-aligned frame

        cpin.forwardKinematics(self.cmodel, self.cdata, q, v, a)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        get_pos = lambda fid: self.cdata.oMf[fid].translation
        get_vel = lambda fid: cpin.getFrameVelocity(self.cmodel, self.cdata, fid, pin.LOCAL_WORLD_ALIGNED).linear
        get_acc = lambda fid: cpin.getFrameAcceleration(self.cmodel, self.cdata, fid, pin.LOCAL_WORLD_ALIGNED).linear

        # 4 x 3 outputs
        return (
            ca.horzcat(*iter(get_pos(f) for f in self.ff_ids)).T,
            ca.horzcat(*iter(get_vel(f) for f in self.ff_ids)).T,
            ca.horzcat(*iter(get_acc(f) for f in self.ff_ids)).T
        )

class ADForwardDynamics():
    def __init__(self, crobot, cdata, act_joint_ids: list[int], feet: list[str]):
        self.cmodel, self.cdata = cmodel, cdata

        # Actuated joint IDs (in robot.model.joints):
        self.act_joint_ids = act_joint_ids

        # Get frame IDs for actuated joints. We'll use these to express GRFs
        # in the correct frames:
        self.act_j_frame_ids = [
            cmodel.getFrameId(robot.model.names[j])
            for j in self.act_joint_ids
        ]

        self.foot_frame_ids = [cmodel.getFrameId(f) for f in feet]

    def __call__(self, q: ca.SX, v: ca.SX, τ_act: ca.SX, λ: ca.SX):
        """
        Input:
        --------
        q (19 x 1), v (18 x 1), τ_act (12 x 1), λ (4  x 1), z-up

        Output:
        --------
        (18 x 1)
        """

        # λ contains normal GRFs for each foot.
        # Find how they're expressed in all actuated joint frames at
        # the provided robot state. FK will populate robot.data.oMf.
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)

        # We assume the GRF always points Z-up (expressed in the world aligned frame):
        X_o_feet_world_aligned = [
            cpin.SE3(ca.SX.eye(3), self.cdata.oMf[f_fr_id].translation)
            for f_fr_id in self.foot_frame_ids
        ]

        grf_at_joints = []

        # For each actuated joint's frame:
        for j_fr_id in self.act_j_frame_ids:
            total_grf = cpin.Force.Zero()
            X_o_joint = self.cdata.oMf[j_fr_id]

            # For each foot:
            for foot_idx in range(len(self.foot_frame_ids)):
                grf_lin = ca.SX.zeros(3)
                grf_lin[2] = λ[foot_idx][0]
                grf_at_foot = cpin.Force(grf_lin, ca.SX.zeros(3))

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
        # torque vector with only the actuated DoFs set.
        # Do the same thing with fext.
        tau_full = ca.SX.zeros(robot.nv, 1)
        fext_full = [cpin.Force.Zero() for _ in range(self.cmodel.njoints)]

        for act_dof, j_id in enumerate(self.act_joint_ids):
            tau_full[self.cmodel.joints[j_id].idx_v] = τ_act[act_dof]
            fext_full[j_id] = grf_at_joints[act_dof]

        # Run the articulated body algorithm:
        accel = cpin.aba(self.cmodel, self.cdata, q, v, tau_full, fext_full)

        # Convert into robot.nv x 1 matrix: # TODO
        return accel

# =====================================================
# Custom state integration functions. This is to avoid 
# numerical issues with pin3's integrate during Hessian calculation.
#   Please see: https://github.com/stack-of-tasks/pinocchio/issues/2050 for a similar issue
from liecasadi import SO3, SO3Tangent

# TODO: Calculate this with liecasadi.SE3 - use the act() operator

def integrate_se3_custom(q: ca.SX, v: ca.SX):
    q_lin, q_rot = q[:3], q[3:7]
    v, ω = v[:3], v[3:6]

    # Calculate integrated quaternion (Lie +):
    q_rot_so3, v_rot_so3t = SO3(q_rot), SO3Tangent(ω) 
    result_rot = (q_rot_so3 + v_rot_so3t).xyzw

    # Caclulate integrated linear position (q_lin + q_rot @ exp(v)_rot)
    # We'll evaluate the rotational part of exp(v) as:
    # exp(v)_rot = V @ v_lin with:
    # V = I + ((1-cosθ)/ θ**2)ωx + ((1 - sinθ/θ) / θ**2)ωx**2
    θ = ca.sqrt(ω.T @ ω + 1e-6)
    θ_sq = θ * θ

    # ω_hat = [
    #     [0,    -ω[2],  ω[1]],
    #     [ω[2],    0,  -ω[0]],
    #     [-ω[1], ω[0],    0]
    # ]
    ω_hat = ca.SX.zeros(3, 3)
    ω_hat[0, 1] = -ω[2]
    ω_hat[0, 2] =  ω[1]
    ω_hat[1, 0] =  ω[2]
    ω_hat[1, 2] =  -ω[0]
    ω_hat[2, 0] =  -ω[1]
    ω_hat[2, 1] =   ω[0]

    A = ca.sin(θ) / θ
    B = (1 - ca.cos(θ)) / θ_sq
    C = (1 - A) / θ_sq

    V = ca.SX.eye(3) + B * ω_hat + C * ω_hat @ ω_hat 
    result_lin = q_lin + q_rot_so3.act(V @ v)
    return ca.vertcat(result_lin, result_rot)

def integrate_custom(q: ca.SX, v: ca.SX):
    q_se3, v_se3 = q[:7], v[:6]

    # Integrate the floating joint using the Lie
    # operation:
    floating_res = integrate_se3_custom(q_se3, v_se3)

    # Integrate revolute joints normally:
    revolute_res = q[7:] + v[6:]
    return ca.vertcat(floating_res, revolute_res)
# =====================================================

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

    FREQ_HZ = 10
    DELTA_T = 1 / FREQ_HZ
    FLOOR_Z = -0.3
    
    robot, viz = load_solo12(floor_z = FLOOR_Z, visualize = True)
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()

    # Start with the robot folded:
    q0 = create_joint_vector(robot, FOLDED_JOINT_MAP)
    v0 = np.zeros((robot.nv, 1))

    # Skip 'universe' and 'root_joint' as unactuated joints:
    actuated_joints = [j.id for j in robot.model.joints[2:]]

    fd = ADForwardDynamics(cmodel, cdata, actuated_joints, feet = FEET)
    fk = ADFootholdKinematics(cmodel, cdata, feet = FEET)

    q_k, v_k, a_k, tau_k, λ_k, foot_z_k = [], [], [], [], [], []  # Collocation variables

    #region Constraint utilities
    constraints = []

    # Create a new elementwise inequality constraint: expr[:] >= 0
    def add_inequality(expr: ca.SX):
        constraints.append((expr, np.zeros(expr.shape), np.full(expr.shape, np.inf)))

    # Create a new elementwise equality constraint: expr[:] == 0
    def add_equality(expr: ca.SX):
        constraints.append((expr, np.zeros(expr.shape), np.zeros(expr.shape)))
    #endregion

    #region Task parameters

    # Trajectory duration - the task will be to keep the body CoM at a given
    # height as well as possible for the entirety of this duration:
    duration = 1
    initial_state = State(0, q0, v0)

    # Contact times per foot, we'll assume the robot can fall for ~0.15m 
    # (half its initial height) before the feet are on the ground:
    contact_times = [ivt.IntervalTree([ivt.Interval(0.17, duration)]) for _ in FEET]

    # Trajectory error function. Given t, q, v, a, τ at a collocation point
    # it returns how far away the trajectory is from the desired one at that
    # time:
    def traj_err(t, q, v, a, τ):
        DESIRED_COM_Z = -0.1
        return (q[2] - DESIRED_COM_Z) ** 2

    #endregion

    N_knots = int(duration * FREQ_HZ) + 1
    
    # print("Creating variables and constraints...")

    for k in range(N_knots):
        #region Collocation variable creation

        # Create decision variables at collocation points:
        q_k.append(ca.SX.sym(f"q_{k}", robot.nq))   # 19 x 1
        v_k.append(ca.SX.sym(f"v_{k}", robot.nv))   # 18 x 1
        a_k.append(ca.SX.sym(f"a_{k}", robot.nv))   # 18 x 1

        tau_k.append(ca.SX.sym(f"τ_{k}", len(actuated_joints)))  # 12 x 1 
        λ_k.append(ca.SX.sym(f"λ_{k}", len(FEET)))               # 4  x 1
        foot_z_k.append(ca.SX.sym(f"foot_z_{k}", len(FEET)))     # 4  x 1
        
        #endregion

        #region Pointwise constraints (accelerations, contact, limits)

        # Manifold constraint for free joint:
        add_equality(ca.dot(q_k[k][3:7], q_k[k][3:7]) - 1)

        #### DYNAMICS CONSTRAINTS ####
        # Residual constraints for accelerations (= 0) at all collocation points:
        add_equality(a_k[k] - fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))

        ##############################
        #### CONTACT CONSTRAINTS #####
        ##############################
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
        add_inequality(tau_k[k] + 4)    # τ >= -4
        add_inequality(4 - tau_k[k])    # τ <= 4
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

        # Integrate the positions. Orientations have to be treated specially:
        add_equality(q_k[k] - integrate_custom(q_k[k-1], 0.5 * DELTA_T * (v_k[k] + v_k[k-1])))
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
    
    ipopt_settings = {
        "linear_solver": "ma57",
        "max_iter": 999999
        # "hessian_approximation": "limited-memory"
    }

    solver = ca.nlpsol("S", "ipopt", nlp, {"ipopt": ipopt_settings})

    #endregion

    #region Initial guess

    #### INITIAL GUESS ####

    print("Creating initial guesses...")

    # Assume that nothing changes as I'm very lazy:
    q_sym, v_sym, a_sym = ca.SX.sym("q_s", cmodel.nq), ca.SX.sym("v_s", cmodel.nv), ca.SX.sym("a_s", cmodel.nv)
    num_fk = ca.Function("num_fk", [q_sym, v_sym, a_sym], fk(q_sym, v_sym, a_sym))

    q_g  = [initial_state.q for _ in range(N_knots)]
    v_g  = [initial_state.v for _ in range(N_knots)]
    a_g  = [np.zeros((robot.nv, 1)) for _ in range(N_knots)]
    τ_g  = [np.zeros((len(actuated_joints), 1)) for _ in range(N_knots)]
    λ_g  = [np.zeros((len(FEET), 1)) for _ in range(N_knots)]
    fz_g = [np.array(num_fk(q_g[0], v_g[0], a_g[0])[0][:, 2]) for _ in range(N_knots)]
    ########################

    #endregion

    # Solve the problem!
    soln = solver(
        x0  = ca.vertcat(*q_g, *v_g, *a_g, *τ_g, *λ_g, *fz_g),
        lbg = ca.vertcat(*lbg),
        ubg = ca.vertcat(*ubg)
    )

    success = solver.stats()["success"] 
    with open(f"solution_{FREQ_HZ}hz_{duration}sec{'_failed' if not success else ''}.bin", "wb") as wf:
        pickle.dump(soln, wf)
    
    if not success:
        print("Solver failed to find solution. Exitting...")
        exit()


    # ####################################################################

    exit()

    with open("solution_10hz_1sec.bin", "rb") as rf:
        soln = pickle.load(rf)

    variables = soln["x"]

    # Extract variables from the solution:
    o = 0
    qs = [np.array(variables[o + idx * robot.nq : o + (idx + 1) * robot.nq]) for idx in range(N_knots)]

    o += N_knots * robot.nq
    vs = [np.array(variables[o + idx * robot.nv : o +(idx + 1) * robot.nv]) for idx in range(N_knots)]

    # # Validate the integration scheme:
    # err = []
    # for k in range(1, N_knots):
    #     err = np.max(
    #         qs[k].T - pin.integrate(robot.model, qs[k-1], 0.5 * DELTA_T * (vs[k-1] + vs[k]))
    #     )

    #     if err > 1e-5:
    #         print("FAILING FOR k = ", k)
    #         print("errors:", qs[k].T - pin.integrate(robot.model, qs[k-1], 0.5 * DELTA_T * (vs[k-1] + vs[k])))
    #         exit()

    input("START THE THING!")
    for idx, q in enumerate(qs):
        robot.display(pin.normalize(robot.model, q))
        time.sleep(DELTA_T)