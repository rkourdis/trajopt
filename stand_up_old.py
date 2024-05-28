import os
import time
import hppfcl
import pickle
import argparse
import functools
from tqdm import tqdm
from itertools import chain
from typing import Optional
from dataclasses import dataclass

from utilities import flatten, unflatten

import numpy as np
import casadi as ca
import pinocchio as pin
import intervaltree as ivt
import matplotlib.pyplot as plt
from pinocchio import casadi as cpin

# region Structs
@dataclass
class State:
    # Second order dynamics:
    #   a = f(t, q, v, τ, λ)
    t: float
    q: np.array                     # 18x (orientations using MRP)
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

def quat2mrp(xyzw: ca.SX) -> ca.SX:
    norm = xyzw / ca.sqrt(xyzw.T @ xyzw)
    return norm[:3] / (1 + norm[3]) 

def mrp2quat(xyz: ca.SX) -> ca.SX:
    normsq = xyz.T @ xyz
    w = (ca.SX.ones(1) - normsq) / (1 + normsq)
    return ca.vertcat(2 * xyz / (1 + normsq), w)

def ca_to_np(x: ca.SX) -> np.array:
    result = np.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            result[i, j] = x[i, j]
    
    return result

def q_mrp_to_quat(q_mrp: ca.SX) -> ca.SX:
    return ca.vertcat(q_mrp[:3], mrp2quat(q_mrp[3:6]), q_mrp[6:])

def q_quat_to_mrp(q_quat: ca.SX) -> ca.SX:
    return ca.vertcat(q_quat[:3], quat2mrp(q_quat[3:7]), q_quat[7:])

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
    q_quat = pin.neutral(robot.model)

    for j_name, angle in joint_angles.items():
        idx = robot.model.getJointId(j_name)
        q_quat[robot.model.joints[idx].idx_q] = angle

    q_mrp = np.concatenate((q_quat[:3], quat2mrp(q_quat[3:7]), q_quat[7:]))
    return np.expand_dims(q_mrp, axis = -1)     # 18x1

class ADFootholdKinematics():
    def __init__(self, cmodel, cdata, feet: list[str]):
        self.cmodel, self.cdata = cmodel, cdata
        self.ff_ids = [robot.model.getFrameId(f) for f in feet]

    def __call__(self, q_mrp: ca.SX, v: ca.SX, a: ca.SX):
        # This runs the second-order FK algorithm and returns:
        # - The positions of all feet wrt the origin
        # - The velocities of all feet in the local world-aligned frame
        # - The accelerations of all feet in the local world-aligned frame

        q = ca.vertcat(q_mrp[:3], mrp2quat(q_mrp[3:6]), q_mrp[6:])

        cpin.forwardKinematics(self.cmodel, self.cdata, q, v, a)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        get_pos = lambda fid: self.cdata.oMf[fid].translation
        # get_vel = lambda fid: cpin.getFrameVelocity(self.cmodel, self.cdata, fid, pin.LOCAL_WORLD_ALIGNED).linear
        # get_acc = lambda fid: cpin.getFrameAcceleration(self.cmodel, self.cdata, fid, pin.LOCAL_WORLD_ALIGNED).linear

        # 4 x 3 outputs
        return (
            ca.horzcat(*iter(get_pos(f) for f in self.ff_ids)).T,
            ca.SX.zeros(4, 3), # ca.horzcat(*iter(get_vel(f) for f in self.ff_ids)).T,
            ca.SX.zeros(4, 3), # ca.horzcat(*iter(get_acc(f) for f in self.ff_ids)).T
        )

class ADForwardDynamics():
    def __init__(self, crobot, cdata, feet: list[str], feet_parent_joints: list[str], act_joint_ids: list[int]):
        self.cmodel, self.cdata = cmodel, cdata

        self.feet = feet
        self.act_joint_ids = act_joint_ids

        self.foot_frame_ids = [cmodel.getFrameId(f) for f in feet]
        self.foot_par_joint_ids = [cmodel.getJointId(j) for j in feet_parent_joints]

    def __call__(self, q_mrp: ca.SX, v: ca.SX, τ_act: ca.SX, λ: ca.SX):
        """
        Input:
        --------
        q (18 x 1, MRP), v (18 x 1), τ_act (12 x 1), λ (4  x 3) in local frame

        Output:
        --------
        (18 x 1)
        """

        # Convert the floating base orientation to quaternion for Pinocchio:
        q = ca.vertcat(q_mrp[:3], mrp2quat(q_mrp[3:6]), q_mrp[6:])

        # λ contains GRFs for each foot.
        # Find how they're expressed in all actuated joint frames at
        # the provided robot state. FK will populate robot.data.oMf.
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
        fext_full = [cpin.Force.Zero() for _ in range(len(self.cmodel.joints))]

        for foot_id, joint_id, foot_frame_id in zip(
            range(len(self.feet)), self.foot_par_joint_ids, self.foot_frame_ids
        ):
            grf_at_foot = cpin.Force(λ[foot_id, :].T, ca.SX.zeros(3))

            fext_full[joint_id] = self.cdata.oMi[joint_id].actInv(
                self.cdata.oMf[foot_frame_id].act(grf_at_foot)
            )
            
        # We'll calculate the unconstrained dynamics using the ABA algorithm.
        # The constraint forces will be chosen by the optimization so that they balance
        # the legs on contact, as described by the contact constraints.
        # In constrained FD, the constraint forces will be implicitly calculated and enforced,
        # but we want to choose them explicitly (otherwise we'll need to know whether we're in
        # contact or not in advance).
        # NOTE: We'll skip all unactuated joints when applying torques, and external forces.

        # Each actuated joint is one degree of freedom. Create a robot.nv x 1
        # torque vector with only the actuated DoFs set.
        tau_full = ca.SX.zeros(robot.nv, 1)
        for act_dof, j_id in enumerate(self.act_joint_ids):
            tau_full[self.cmodel.joints[j_id].idx_v] = τ_act[act_dof]

        # Run the articulated body algorithm:
        accel = cpin.aba(self.cmodel, self.cdata, q, v, tau_full, fext_full)

        # Convert into robot.nv x 1 matrix: # TODO
        return accel

# =====================================================
# Custom state integration functions. This is to avoid 
# numerical issues with pin3's integrate during Hessian calculation.
#   Please see: https://github.com/stack-of-tasks/pinocchio/issues/2050 for a similar issue
from liecasadi import SE3, SE3Tangent

def integrate_custom(q: ca.SX, v: ca.SX):
    # Floating base orientation uses MRP:
    q_se3 = SE3(pos = q[:3], xyzw = mrp2quat(q[3:6]))
    v_se3 = SE3Tangent(v[:6])

    # Integrate the floating joint using the Lie operation:
    fb_r_se3 = q_se3 * v_se3.exp()

    # Integrate revolute joints normally:
    r_r = q[6:] + v[6:]
    return ca.vertcat(fb_r_se3.pos, quat2mrp(fb_r_se3.xyzw), r_r)
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualise', action='store_true')
    options = parser.parse_args()

    # The order of forces per foot WILL be in this order:
    FEET = ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]
    F_PAR_J = ["FR_KFE", "FL_KFE", "HR_KFE", "HL_KFE"]

    FREQ_HZ = 20
    DELTA_T = 1 / FREQ_HZ
    FLOOR_Z = -0.226274
    TRAJ_DURATION = 2
    N_KNOTS = int(TRAJ_DURATION * FREQ_HZ)
    OUTPUT_FILENAME = f"solution_{FREQ_HZ}hz_{TRAJ_DURATION}sec.bin"
    # OUTPUT_FILENAME = "jump_20hz_2000ms.bin" #f"jump_{FREQ_HZ}hz_2.0sec.bin"

    robot, viz = load_solo12(floor_z = FLOOR_Z, visualize = options.visualise)
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()

    with open("initial_guesses/standing_pose.bin", "rb") as rf:
        standup = pickle.load(rf)

    q0 = standup["q"]
    v0 = np.zeros(robot.nv)
    tau0 = standup["tau"]
    λ0 = np.vstack(np.split(standup["λ_local_wa"], 4))

    # Skip 'universe' and 'root_joint' as unactuated joints:
    actuated_joints = [j.id for j in robot.model.joints[2:]]

    fd = ADForwardDynamics(cmodel, cdata, feet = FEET, feet_parent_joints = F_PAR_J, act_joint_ids=actuated_joints)
    fk = ADFootholdKinematics(cmodel, cdata, feet = FEET)

    if not options.visualise:
        q_k, v_k, a_k, tau_k, λ_k, f_pos_k = [], [], [], [], [], []  # Collocation variables

        #region Constraint utilities
        constraints = []
        boundaries: dict[str, tuple[float, float]] = {}

        # Create a new elementwise constraint: lb <= expr[:] <= ub
        def add_constraint(expr: ca.SX, lb: float = 0.0, ub: float = 0.0):
            assert lb <= ub
            constraints.append((expr, np.full(expr.shape, lb), np.full(expr.shape, ub)))

        def add_boundary(var: ca.SX, lb: float = -np.inf, ub: float = np.inf):
            assert var.shape == (1, 1)
            assert var.is_symbolic()
            assert lb <= ub
            boundaries[var.name()] = (lb, ub)

        #endregion

        #region Task parameters

        initial_state = State(0, q0, v0)

        # Contact times per foot, we'll assume the robot can fall for ~0.15m 
        # (half its initial height) before the feet are on the ground:
        # NOTE: The intervals do not include the end!
        contact_times = [
            ivt.IntervalTree([
                ivt.Interval(0.0, 0.3 + DELTA_T / 2),
                ivt.Interval(0.7, TRAJ_DURATION + DELTA_T / 2),
            ])
            for _ in FEET
        ]

        # Trajectory error function. Given t, q, v, a, τ at a collocation point
        # it returns how far away the trajectory is from the desired one at that
        # time:
        def traj_err(t, q, v, a, τ):
            if contact_times[0].overlaps(t):
                return τ.T @ τ #+ (q[:3].T @ q[:3]) * 1e-1

            return τ.T @ τ
            # if 1.0 <= t and t < 1.5:
            #     com_orientation_err = q[:6] - ca.SX([0, 0, 0.2, 0.0, 0.0, 0.0])
            #     return com_orientation_err.T @ com_orientation_err + v[6:].T @ v[6:]

            # z_des = ca.sin(2 * ca.pi * t) * 0.1
            # com_orientation_err = q[:6] - ca.SX([0, 0, z_des, 0.0, 0.0, 0.0])
            # return com_orientation_err.T @ com_orientation_err + v[6:].T @ v[6:]
        
        #endregion
        
        # print("Creating variables and constraints...")
        # """

        for k in range(N_KNOTS):
            #region Collocation variable creation

            # Create decision variables at collocation points:
            q_k.append(ca.SX.sym(f"q_{k}", robot.nq - 1)) # We will represent orientations with MRP instead of quaternions
            v_k.append(ca.SX.sym(f"v_{k}", robot.nv))     # 18 x 1
            a_k.append(ca.SX.sym(f"a_{k}", robot.nv))     # 18 x 1

            tau_k.append(ca.SX.sym(f"τ_{k}", len(actuated_joints)))  # 12 x 1 
            λ_k.append(ca.SX.sym(f"λ_{k}", len(FEET), 3))            # 4  x 3
            f_pos_k.append(ca.SX.sym(f"f_pos_{k}", len(FEET), 3))    # 4  x 3
            
            #endregion

            #region Pointwise constraints (accelerations, contact, limits)
            for idx in range(3):
                add_boundary(q_k[-1][idx], FLOOR_Z)

            #### DYNAMICS CONSTRAINTS ####
            # Residual constraints for accelerations (= 0) at all collocation points:
            add_constraint(a_k[k] - fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))

            #### JOINT LIMITS ####
            for j in range(len(actuated_joints)):
                add_boundary(tau_k[k][j], -6, 6)
            ######################

            #endregion

            # We'll add integration constraints for all knots, wrt their previous points:
            if k == 0:
                continue

            #region Integration constraints (residuals)

            #### INTEGRATION CONSTRAINTS ####
            # Velocities - integrate using a single Euler step. This is because
            # the velocity you use for FK and for calculating the next state needs to match!
            add_constraint(v_k[k] - v_k[k-1] - DELTA_T * a_k[k-1])

            # Integrate the positions. Orientations have to be treated specially:
            add_constraint(q_k[k] - integrate_custom(q_k[k-1], DELTA_T * v_k[k-1]))
            ##################################

            #endregion

        ##############################
        #### CONTACT CONSTRAINTS #####
        ##############################
        for k in range(N_KNOTS):
            f_pos, _, _ = fk(q_k[k], v_k[k], a_k[k])         # All 4 x 3
            add_constraint(f_pos_k[k] - f_pos)

            # Enforce constraints for all feet:
            for idx, foot in enumerate(FEET):

                # TODO: FIX THE OFF BY ONE, THERE'S A BUG SOMEWHERE

                # If that particular foot is in contact with the ground:
                if contact_times[idx].overlaps(k * DELTA_T):
                    # Forces must result in the foot being pinned on the ground
                    # at the next knot:
                    # TODO: What's going to happen at the last knot?
                    if (k < N_KNOTS - 1):
                        add_constraint(f_pos_k[k+1][idx, 2] - FLOOR_Z)     

                    # For some reason, the foot velocity is calculated as non-zero by FK,
                    # even though when integrating the foot stays put. The following doesn't work:
                    # add_equality(f_vel[idx, :])                  # Foot should not move
                    # I'm not sure why. For now, force the foot position to be the same as before
                    # if this isn't the first contact knot:

                    #TODO: FIX THIS!
                    if contact_times[idx].overlaps((k-1) * DELTA_T):
                        add_constraint(f_pos_k[k][idx, :] - f_pos_k[k-1][idx, :])

                    add_boundary(λ_k[k][idx, 2], 0, np.inf)     # Z-up contact force available

                    # # Foot Z acceleration should be zero - this is to avoid issues with the optimizer
                    # # tricking the integration scheme and applying oscillating torques and GRFs while
                    # # on the floor (such that foot_z = 0 after trapezoidal integration...)
                    # add_equality(f_acc[idx, 2])
                else:
                    # Allow the next knot's feet to be free:
                    if (k < N_KNOTS - 1):
                        add_constraint(f_pos_k[k+1][idx, 2] - FLOOR_Z, 0.0, np.inf)

                    add_constraint(λ_k[k][idx, :])                 # Contact forces unavailable
            #############################
        
        #region Objective and boundaries

        print("Creating optimization objective...")

        # Create optimization objective.
        # Integrate the error over time, use trapezoidal approximation:
        err_k = lambda k: traj_err(k * DELTA_T, q_k[k], v_k[k], a_k[k], tau_k[k])
        obj = sum(DELTA_T / TRAJ_DURATION * ca.sqrt(err_k(idx)) for idx in range(N_KNOTS))

        # TODO: MAKING THE ERROR TO BE JUST THE V WORKS, WITH THE Q FORMULATION IT GETS STUCK!
        # CAN I ASK IT TO STOP EARLY? IS THERE SOMETHING WRONG WITH THE INTEGRAL?

        print("Adding boundary constraints...")

        #### BOUNDARY CONSTRAINTS ####
        legs = np.array([
            [0.1946, -0.14695, -0.22627417],
            [ 0.1946, 0.14695, -0.22627417],
            [-0.1946, -0.14695, -0.22627417],
            [-0.1946,  0.14695, -0.22627417]
        ])

        for i in range(4):
            for j in range(3):
                add_boundary(f_pos_k[0][i, j], legs[i, j], legs[i, j])
                add_boundary(f_pos_k[-1][i, j], legs[i, j], legs[i, j])

        add_constraint(q_k[0] - initial_state.q)     # Initial q
        add_constraint(v_k[0] - initial_state.v)    # Initial v

        for idx in range(3):
            add_boundary(v_k[-1][idx], 0, 0)

        # add_boundary(q_k[-1][2], FLOOR_Z + 0.2, FLOOR_Z + 0.2)
        ###############################

        #endregion

        #region NLP setup
        
        print("Creating NLP description...")

        # Create the NLP problem:
        g, lbg, ubg = zip(*constraints)
    
        # # print("G:", ca.vertcat(*flatten(g)))
        # print("UBG:", ca.vertcat(*flatten(ubg)))
        # print("LBG:", ca.vertcat(*flatten(lbg)))

        # import code
        # code.interact(local=locals())
        # exit()
        
        variables = ca.vertcat(*q_k, *v_k, *a_k, *tau_k, flatten(λ_k), flatten(f_pos_k))

        nlp = {
            "x": variables,
            "f": obj,
            "g": flatten(g)
        }

        print("Instantiating solver...")
        
        # ipopt_settings = {
        #     "linear_solver": "ma57",
        #     "tol": 1e-6,
        #     "max_iter": 999999,
        #     "replace_bounds": "no"
        #     # "hessian_approximation": "limited-memory"
        # }
        import knitro

        knitro_settings = {
            "hessopt": knitro.KN_HESSOPT_LBFGS,
            "algorithm": knitro.KN_ALG_BAR_DIRECT,
            "bar_murule": knitro.KN_BAR_MURULE_ADAPTIVE,
            "linsolver": knitro.KN_LINSOLVER_MA57,
            "feastol": 1e-3,
            # "feastol_abs": 1e-3
            "ftol": 1e-4,
            # "numthreads": 8
        }

        # solver = ca.nlpsol("S", "ipopt", nlp, {"ipopt": ipopt_settings})

        # TODO: add {'verbose':True} to find whether it calculates hessians when it doesn't need them!
        solver = ca.nlpsol("S", "knitro", nlp, {"knitro": knitro_settings})

        #endregion

        #region Initial guess

        #### INITIAL GUESS ####

        print("Creating initial guesses...")

        # Assume that nothing changes as I'm very lazy:
        q_sym, v_sym, a_sym = ca.SX.sym("q_s", cmodel.nq - 1), ca.SX.sym("v_s", cmodel.nv), ca.SX.sym("a_s", cmodel.nv)
        num_fk = ca.Function("num_fk", [q_sym, v_sym, a_sym], fk(q_sym, v_sym, a_sym))

        #########################

        """
        N_KNOTS_LOW_FREQ = int(TRAJ_DURATION * 20)
        with open("solution_20hz_2sec.bin", "rb") as rf:
            soln = pickle.load(rf)["x"]

        # Extract variables from the solution:
        o = 0
        
        q_lowfreq = unflatten(soln[o : o + 18 * N_KNOTS_LOW_FREQ], (18, 1))
        o += N_KNOTS_LOW_FREQ * 18
        
        v_lowfreq = unflatten(soln[o : o + 18 * N_KNOTS_LOW_FREQ], (18, 1))
        o += N_KNOTS_LOW_FREQ * 18

        a_lowfreq = unflatten(soln[o : o + 18 * N_KNOTS_LOW_FREQ], (18, 1))
        o += N_KNOTS_LOW_FREQ * 18

        τ_lowfreq = unflatten(soln[o : o + 12 * N_KNOTS_LOW_FREQ], (12, 1))
        o += N_KNOTS_LOW_FREQ * 12
        
        λ_lowfreq = unflatten(soln[o : o + 12 * N_KNOTS_LOW_FREQ], (4, 3))
        o += N_KNOTS_LOW_FREQ * len(FEET) * 3

        f_pos_lowfreq = unflatten(soln[o : o + 12 * N_KNOTS_LOW_FREQ], (4, 3))
        o += N_KNOTS_LOW_FREQ * len(FEET) * 3

        q_g  = chain.from_iterable([[q_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])
        v_g  = chain.from_iterable([[v_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])
        a_g  = chain.from_iterable([[a_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])
        τ_g  = chain.from_iterable([[τ_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])
        λ_g  = chain.from_iterable([[λ_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])
        f_pos_g  = chain.from_iterable([[f_pos_lowfreq[idx]] * (N_KNOTS // N_KNOTS_LOW_FREQ) for idx in range(N_KNOTS_LOW_FREQ)])

        """
        q_g  = [initial_state.q for _ in range(N_KNOTS)]
        v_g  = [initial_state.v for _ in range(N_KNOTS)]
        a_g  = [np.zeros((robot.nv, 1)) for _ in range(N_KNOTS)]
        τ_g  = [np.copy(tau0) for _ in range(N_KNOTS)]
        λ_g  = [np.copy(λ0) for _ in range(N_KNOTS)]
        f_pos_g = [np.array(num_fk(q_g[0], v_g[0], a_g[0])[0]) for _ in range(N_KNOTS)]
        # """
        ########################

        #endregion

        lbx, ubx = [], []
        for idx in range(variables.shape[0]):
            b = boundaries.get(variables[idx].name())

            lbx.append(b[0] if b else -np.inf)
            ubx.append(b[1] if b else np.inf)

        # Solve the problem!
        soln = solver(
            x0  = ca_to_np(ca.vertcat(*q_g, *v_g, *a_g, *τ_g, flatten(λ_g), flatten(f_pos_g))),
            lbg = flatten(lbg),
            ubg = flatten(ubg),
            lbx = np.array(lbx),
            ubx = np.array(ubx)
        )

        success = solver.stats()["success"] 
        with open(OUTPUT_FILENAME, "wb") as wf:
            pickle.dump(soln, wf)
        
        if not success:
            print("Solver failed to find solution. Exitting...")

        exit()

    # ####################################################################

    with open(OUTPUT_FILENAME, "rb") as rf:
        soln = pickle.load(rf)

    variables = soln["x"]

    # Extract variables from the solution:
    o = 0
    
    qv = [np.array(variables[o + idx * (robot.nq - 1) : o + (idx + 1) * (robot.nq - 1)]) for idx in range(N_KNOTS)]
    o += N_KNOTS * (robot.nq - 1)

    vv = [np.array(variables[o + idx * robot.nv : o + (idx + 1) * robot.nv]) for idx in range(N_KNOTS)]
    o += N_KNOTS * robot.nv
    
    av = [np.array(variables[o + idx * robot.nv : o + (idx + 1) * robot.nv]) for idx in range(N_KNOTS)]
    o += N_KNOTS * robot.nv

    # tau
    o += N_KNOTS * len(actuated_joints)

    λs = unflatten(variables[o : o + 12 * N_KNOTS], (4, 3))
    o += N_KNOTS * len(FEET) * 3

    # q_s, v_s, a_s = ca.SX.sym("q", 18, 1), ca.SX.sym("v", 18, 1), ca.SX.sym("a", 18, 1)
    # fkv = ca.Function("fkv", [q_s, v_s, a_s], [fk(q_s, v_s, a_s)[1]])
    # fkx = ca.Function("fkv", [q_s, v_s, a_s], [fk(q_s, v_s, a_s)[0]])

    # pos_hist, vel_hist = [], []
    # err = []

    # for k in range(N_KNOTS):
    #     pos = fkx(qv[k], vv[k], av[k])[0, :]
    #     vel = fkv(qv[k], vv[k], av[k])[0, :]

    #     if (k > 0):
    #         integr = pos_hist[-1] + DELTA_T * vel_hist[-1]
    #         err.append(pos - integr)

    #     pos_hist.append(pos)
    #     vel_hist.append(vel)
        
    #     # print("pos:", pos, "\t\tvel:", vel)
    
    # print(np.sum(np.abs(err), axis = 0))

    # # With 30 Hz:
    # # LOCAL_WORLD_ALIGNED: [[0.00219949 0.00270747 0.00322794]]
    # # LOCAL: [[0.04122901 0.00220468 0.08294682]]
    # # WORLD: [[0.21330099 0.09794994 0.27863034]]
    # exit()

    # print(qv)
    # print(λs)

    input("Press ENTER to play trajectory...")
    
    """
        TODO: While the constraint violation is as follows below,
        integrating manually the positions results in bad movement.

        The robot flies, and the legs move off the ground.
        ALSO, there's a difference between pinocchio integrate and custom

        TODO: Once you figure this out, you have to make sure that FK_vel_z = 0
        if it _actually_ looks like it is...
        If you remove that constraint everything optimizes so, LOCAL_WORLD_ALIGNED
        might not be correct. Have a look.


    """
    # q_mrp = qv[0]
    # q = ca_to_np(q_mrp_to_quat(qv[0].T[0]))

    for idx, q_mrp in tqdm(enumerate(qv[:-1])):
        # vel_avg = 0.5 * (vv[idx] + vv[idx+1])
        # q = pin.integrate(robot.model, q,  vv[idx] * DELTA_T)
        # q = ca_to_np(q_mrp_to_quat(integrate_custom(q_mrp, vel_avg * DELTA_T)))
        q = ca_to_np(q_mrp_to_quat(qv[idx]))
        robot.display(q)
        time.sleep(DELTA_T * 2)
        # input()
