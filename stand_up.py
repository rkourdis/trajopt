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

    # The order here does NOT correspond to the order of joints
    # in the model, or the actuated_joints list below!
    # FOLDED_JOINT_MAP = {
    #     "FR_HAA": 0,
    #     "FL_HAA": 0,
    #     "HR_HAA": 0,
    #     "HL_HAA": 0,
    #     "FR_KFE": -np.pi,
    #     "FR_HFE": np.pi / 2,
    #     "FL_KFE": -np.pi,
    #     "FL_HFE": np.pi / 2,
    #     "HR_KFE": np.pi,
    #     "HR_HFE": -np.pi / 2,
    #     "HL_KFE": np.pi,
    #     "HL_HFE": -np.pi / 2,
    # }

    # UPRIGHT_JOINT_MAP = {
    #     "FR_HAA": 0,
    #     "FL_HAA": 0,
    #     "HR_HAA": 0,
    #     "HL_HAA": 0,
    #     "FR_KFE": -np.pi / 2,
    #     "FR_HFE": np.pi / 4,
    #     "FL_KFE": -np.pi / 2,
    #     "FL_HFE": np.pi / 4,
    #     "HR_KFE": np.pi / 2,
    #     "HR_HFE": -np.pi / 4,
    #     "HL_KFE": np.pi / 2,
    #     "HL_HFE": -np.pi / 4,
    # }

    # The order of forces per foot WILL be in this order:
    FEET = ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]
    F_PAR_J = ["FR_KFE", "FL_KFE", "HR_KFE", "HL_KFE"]

    FREQ_HZ = 50
    DELTA_T = 1 / FREQ_HZ
    FLOOR_Z = -0.226274
    TRAJ_DURATION = 1
    N_KNOTS = int(TRAJ_DURATION * FREQ_HZ) + 1
    OUTPUT_FILENAME = f"solution_{FREQ_HZ}hz_{TRAJ_DURATION}sec.bin"

    robot, viz = load_solo12(floor_z = FLOOR_Z, visualize = options.visualise)
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()

    with open("standing_pose.bin", "rb") as rf:
        standup = pickle.load(rf)

    q0 = ca_to_np(q_quat_to_mrp(standup["q"]))
    v0 = np.zeros(robot.nv)
    tau0 = standup["tau"]
    λ0 = np.vstack(np.split(standup["λ_local"], 4))

    # Skip 'universe' and 'root_joint' as unactuated joints:
    actuated_joints = [j.id for j in robot.model.joints[2:]]

    fd = ADForwardDynamics(cmodel, cdata, feet = FEET, feet_parent_joints = F_PAR_J, act_joint_ids=actuated_joints)
    fk = ADFootholdKinematics(cmodel, cdata, feet = FEET)

    if not options.visualise:

        q_k, v_k, a_k, tau_k, λ_k, f_pos_k = [], [], [], [], [], []  # Collocation variables

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

        initial_state = State(0, q0, v0)

        # Contact times per foot, we'll assume the robot can fall for ~0.15m 
        # (half its initial height) before the feet are on the ground:
        # NOTE: The intervals do not include the end!
        contact_times = [
            ivt.IntervalTree([
                ivt.Interval(0.0, TRAJ_DURATION + DELTA_T / 2),
            ])
            for _ in FEET
        ]

        # Trajectory error function. Given t, q, v, a, τ at a collocation point
        # it returns how far away the trajectory is from the desired one at that
        # time:
        def traj_err(t, q, v, a, τ):
            z_des = ca.sin(2 * ca.pi * t) * 0.1
            com_pos_err = q[:3] - ca.SX([0, 0, z_des])
            return com_pos_err.T @ com_pos_err
        
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

            #### DYNAMICS CONSTRAINTS ####
            # Residual constraints for accelerations (= 0) at all collocation points:
            add_equality(a_k[k] - fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))

            ##############################
            #### CONTACT CONSTRAINTS #####
            ##############################
            f_pos, _, _ = fk(q_k[k], v_k[k], a_k[k])         # All 4 x 3
            add_equality(f_pos_k[k] - f_pos)

            # Enforce constraints for all feet:
            for idx, foot in enumerate(FEET):

                # If that particular foot is in contact with the ground:
                if contact_times[idx].overlaps(k * DELTA_T):
                    add_equality(f_pos_k[k][idx, 2] - FLOOR_Z)     # Foot should be at the floor height

                    # For some reason, the foot velocity is calculated as non-zero by FK,
                    # even though when integrating the foot stays put. The following doesn't work:
                    # add_equality(f_vel[idx, :])                  # Foot should not move
                    # I'm not sure why. For now, force the foot position to be the same as before
                    # if this isn't the first contact knot:
                    # if contact_times[idx].overlaps((k-1) * DELTA_T):
                    #     add_equality(f_pos_k[k][idx, :] - f_pos_k[k-1][idx, :])

                    # TODO: Make sure the contact force is not Z-down...
                    # add_inequality(λ_k[k][idx, 2])               # Z-up contact force available
                    
                    # # Foot Z acceleration should be zero - this is to avoid issues with the optimizer
                    # # tricking the integration scheme and applying oscillating torques and GRFs while
                    # # on the floor (such that foot_z = 0 after trapezoidal integration...)
                    # add_equality(f_acc[idx, 2])
                else:
                    add_inequality(f_pos[idx, 2] - FLOOR_Z)      # TODO: Change this to strict inequality?
                    add_equality(λ_k[k][idx, :])                 # Contact forces unavailable
            #############################

            #### JOINT LIMITS ####
            # add_inequality(tau_k[k] + 4)    # τ >= -4
            # add_inequality(4 - tau_k[k])    # τ <= 4
            ######################

            #endregion

            # We'll add integration constraints for all knots, wrt their previous points:
            if k == 0:
                continue

            #region Integration constraints (residuals)

            #### INTEGRATION CONSTRAINTS ####
            # Velocities - integrate using a single Euler step. This is because
            # the velocity you use for FK and for calculating the next state needs to match!
            add_equality(v_k[k] - v_k[k-1] - DELTA_T * a_k[k-1])

            # Integrate the positions. Orientations have to be treated specially:
            add_equality(q_k[k] - integrate_custom(q_k[k-1], DELTA_T * v_k[k-1]))
            ##################################

            #endregion
        
        #region Objective and boundaries

        print("Creating optimization objective...")

        # Create optimization objective.
        # Integrate the error over time, use trapezoidal approximation:
        err_k = lambda k: traj_err(k * DELTA_T, q_k[k], v_k[k], a_k[k], tau_k[k])
        obj = sum(DELTA_T * err_k(idx) for idx in range(N_KNOTS))

        # TODO: MAKING THE ERROR TO BE JUST THE V WORKS, WITH THE Q FORMULATION IT GETS STUCK!
        # CAN I ASK IT TO STOP EARLY? IS THERE SOMETHING WRONG WITH THE INTEGRAL?

        print("Adding boundary constraints...")

        #### BOUNDARY CONSTRAINTS ####
        add_equality(q_k[0] - initial_state.q)     # Initial q
        add_equality(v_k[0] - initial_state.v)    # Initial v
        ###############################

        #endregion

        #region NLP setup
        
        print("Creating NLP description...")

        # Create the NLP problem:
        def flatten(mats: list[ca.SX]):
            return chain.from_iterable(ca.horzsplit(m) for m in mats)
        
        g, lbg, ubg = zip(*constraints)

        nlp = {
            "x": ca.vertcat(*q_k, *v_k, *a_k, *tau_k, *flatten(λ_k), *flatten(f_pos_k)),
            "f": obj,
            "g": ca.vertcat(*flatten(g))
        }

        print("Instantiating solver...")
        
        ipopt_settings = {
            "linear_solver": "ma57",
            "tol": 1e-6,
            "max_iter": 999999
            # "hessian_approximation": "limited-memory"
        }

        solver = ca.nlpsol("S", "ipopt", nlp, {"ipopt": ipopt_settings})

        #endregion

        #region Initial guess

        #### INITIAL GUESS ####

        print("Creating initial guesses...")

        # Assume that nothing changes as I'm very lazy:
        q_sym, v_sym, a_sym = ca.SX.sym("q_s", cmodel.nq - 1), ca.SX.sym("v_s", cmodel.nv), ca.SX.sym("a_s", cmodel.nv)
        num_fk = ca.Function("num_fk", [q_sym, v_sym, a_sym], fk(q_sym, v_sym, a_sym))

        q_g  = [initial_state.q for _ in range(N_KNOTS)]
        v_g  = [initial_state.v for _ in range(N_KNOTS)]
        a_g  = [np.zeros((robot.nv, 1)) for _ in range(N_KNOTS)]
        τ_g  = [np.copy(tau0) for _ in range(N_KNOTS)]
        λ_g  = [np.copy(λ0) for _ in range(N_KNOTS)]
        f_pos_k = [np.array(num_fk(q_g[0], v_g[0], a_g[0])[0]) for _ in range(N_KNOTS)]
        ########################

        #endregion

        # Solve the problem!
        soln = solver(
            x0  = ca.vertcat(*q_g, *v_g, *a_g, *τ_g, *flatten(λ_g), *flatten(f_pos_k)),
            lbg = ca.vertcat(*flatten(lbg)),
            ubg = ca.vertcat(*flatten(ubg))
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

    o += 2 * N_KNOTS * robot.nq
    av = [np.array(variables[o + idx * robot.nv : o + (idx + 1) * robot.nv]) for idx in range(N_KNOTS)]

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
        time.sleep(DELTA_T * 5)


"""
******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.16, running with linear solver ma57.

Number of nonzeros in equality constraint Jacobian...:    40972
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:        0

Total number of variables............................:     2418
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:     1798
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  3.2000000e-03 1.70e-07 5.30e-03   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  3.1999774e-03 1.36e-07 5.08e-04 -11.0 5.08e-04    -  1.00e+00 1.00e+00h  1
Reallocating memory for MA57: lfact (352901)
Reallocating memory for MA57: lfact (448172)
Reallocating memory for MA57: lfact (476050)
Reallocating memory for MA57: lfact (612312)
Reallocating memory for MA57: lfact (659991)
   2  3.1977234e-03 8.47e-05 7.95e-04 -11.0 3.32e-03  -2.0 1.00e+00 1.00e+00h  1
   3  3.1909953e-03 5.23e-04 8.42e-04 -11.0 4.88e-03  -2.5 1.00e+00 1.00e+00h  1
   4  3.1710943e-03 2.51e-03 1.25e-03 -11.0 1.42e-02  -3.0 1.00e+00 1.00e+00h  1
   5  3.1139016e-03 2.04e-02 1.63e-04 -11.0 4.12e-02  -3.4 1.00e+00 1.00e+00h  1
   6  3.1129307e-03 5.16e-02 8.42e-04 -11.0 1.17e-01  -3.9 1.00e+00 5.00e-01h  2
   7  2.7003739e-03 1.06e+00 5.81e-03 -11.0 2.93e-01  -4.4 1.00e+00 1.00e+00h  1
   8  2.4278686e-03 1.87e+00 1.91e-02 -11.0 4.85e-01  -4.9 1.00e+00 1.00e+00h  1
   9  2.6478059e-03 3.09e-01 2.49e-03 -11.0 2.02e-01  -5.3 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  2.4697670e-03 1.70e-01 1.60e-02 -11.0 1.17e-01  -5.8 1.00e+00 1.00e+00h  1
  11  2.4536985e-03 7.56e-02 2.25e-02 -11.0 6.71e-01    -  1.00e+00 1.00e+00h  1
  12  2.5454717e-03 8.82e-02 2.27e-02 -11.0 2.49e-01  -6.3 1.00e+00 1.00e+00h  1
  13  2.3556564e-03 1.54e-01 9.67e-03 -11.0 3.69e-01  -6.8 1.00e+00 1.00e+00h  1
  14  2.4389431e-03 4.05e-02 1.96e-03 -11.0 1.31e-01  -7.2 1.00e+00 1.00e+00h  1
Reallocating memory for MA57: lfact (793219)
Reallocating memory for MA57: lfact (834274)
  15  2.4052387e-03 4.80e-02 9.20e-04 -11.0 8.02e-02  -7.7 1.00e+00 1.00e+00h  1
  16  2.3745314e-03 7.51e-02 7.01e-03 -11.0 1.18e+00  -8.2 1.00e+00 2.50e-01h  3
Reallocating memory for MA57: lfact (890668)
  17  2.3376655e-03 8.89e-03 2.31e-03 -11.0 8.74e-02  -8.7 1.00e+00 1.00e+00h  1
  18  2.3415512e-03 1.70e-03 3.70e-04 -11.0 1.32e-02  -9.2 1.00e+00 1.00e+00h  1
  19  2.3339462e-03 8.24e-04 8.33e-05 -11.0 1.17e-02  -9.6 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  2.3335030e-03 5.71e-05 5.48e-05 -11.0 5.60e-03 -10.1 1.00e+00 1.00e+00h  1
  21  2.3445248e-03 8.54e-06 1.21e-02 -11.0 2.45e-01 -10.6 1.00e+00 1.00e+00H  1
Reallocating memory for MA57: lfact (1032881)
  22  2.3434188e-03 2.62e-05 1.26e-02 -11.0 5.11e+00 -11.1 1.00e+00 1.95e-03h 10
  23  2.3329231e-03 1.17e-02 2.31e-02 -11.0 1.33e+01 -11.5 1.00e+00 1.56e-02h  7
Reallocating memory for MA57: lfact (1092464)
  24  2.3310432e-03 1.12e-02 2.54e-02 -11.0 4.82e+00 -12.0 1.00e+00 1.56e-02h  7
  25  2.4636113e-03 4.22e-06 3.68e-02 -11.0 8.94e-01 -12.5 1.00e+00 1.00e+00H  1
  26  2.3100659e-03 6.03e-01 1.13e-01 -11.0 3.09e+01 -13.0 1.00e+00 6.25e-02h  5
  27  2.2749189e-03 6.51e-01 1.45e-01 -11.0 4.25e+02 -12.5 1.00e+00 1.95e-03h 10
Reallocating memory for MA57: lfact (1294222)
  28  2.2529473e-03 7.34e-01 1.82e-01 -11.0 1.24e+02 -13.0 1.00e+00 7.81e-03h  8
  29  2.2734245e-03 7.04e-01 2.90e+00 -11.0 1.62e+02 -13.5 1.00e+00 3.91e-03h  9
Reallocating memory for MA57: lfact (1372638)
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  30  2.2411053e-03 7.32e-01 2.85e+00 -11.0 5.03e+01 -14.0 1.00e+00 1.56e-02h  7
  31  1.9877491e-03 5.06e-01 2.38e+00 -11.0 5.20e-01  -4.1 1.00e+00 1.00e+00h  1
  32  2.1107707e-03 4.27e-01 2.93e+00 -11.0 3.38e-01  -4.6 1.00e+00 2.50e-01h  3
  33  1.9197394e-03 8.03e-03 6.20e-01 -11.0 1.06e-01  -5.1 1.00e+00 1.00e+00h  1
  34  1.8795986e-03 3.63e-02 5.04e-01 -11.0 3.10e-01  -5.6 1.00e+00 1.00e+00h  1
  35  1.8732026e-03 3.83e-02 5.05e-01 -11.0 6.08e+00  -6.0 1.00e+00 3.91e-03h  9
  36  1.8460452e-03 3.55e-01 4.82e-01 -11.0 2.88e+00  -6.5 1.00e+00 2.50e-01h  3
  37  1.7980429e-03 1.15e+00 4.65e-01 -11.0 2.12e+01  -7.0 1.00e+00 3.12e-02h  6
  38  1.7464245e-03 1.67e+00 4.58e-01 -11.0 4.28e+01  -7.5 1.00e+00 1.56e-02h  7
  39  1.7034167e-03 2.22e+00 4.50e-01 -11.0 3.90e+01  -8.0 1.00e+00 3.12e-02h  6
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  40  1.7007072e-03 2.22e+00 4.52e-01 -11.0 5.30e+01  -7.5 1.00e+00 7.81e-03h  8
  41  1.6763508e-03 2.22e+00 4.37e-01 -11.0 2.19e+01  -8.0 1.00e+00 3.12e-02h  6
  42  1.6847133e-03 2.16e+00 3.98e-01 -11.0 1.02e+01  -6.7 1.00e+00 3.12e-02h  6
  43  1.6732004e-03 2.17e+00 3.64e-01 -11.0 3.84e+00  -7.2 1.00e+00 6.25e-02h  5
  44  1.6318259e-03 2.35e+00 4.47e-01 -11.0 1.62e+01  -7.6 1.00e+00 1.56e-02h  7
  45  3.3087154e-02 2.18e+02 1.91e+01 -11.0 1.06e+01  -8.1 1.00e+00 1.00e+00w  1
  46  2.2174532e-02 1.01e+02 1.75e+01 -11.0 2.05e+00  -8.6 1.00e+00 1.00e+00w  1
  47  6.6414717e-03 4.53e+01 1.46e+01 -11.0 1.65e+00  -9.1 1.00e+00 1.00e+00w  1
  48  1.6250919e-03 2.53e+00 4.70e-01 -11.0 1.79e+00  -9.5 1.00e+00 6.25e-02h  4
  49  1.5951336e-03 2.54e+00 4.40e-01 -11.0 9.50e+00  -6.4 1.00e+00 3.91e-03h  9
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  50  1.5415350e-03 4.28e-02 1.54e+00 -11.0 4.53e-01  -6.9 1.00e+00 1.00e+00h  1
  51  1.5447479e-03 3.81e-02 1.67e+00 -11.0 2.44e+00  -7.4 1.00e+00 3.12e-02h  6
  52  1.7039876e-03 3.28e-03 1.34e+00 -11.0 6.85e-01  -7.8 1.00e+00 1.00e+00H  1
  53  1.6390537e-03 2.14e-02 1.96e+00 -11.0 4.25e+00  -8.3 1.00e+00 1.25e-01h  4
  54  1.5777091e-03 2.35e-02 2.21e+00 -11.0 1.47e+01  -7.0 1.00e+00 1.56e-02h  7
  55  1.6147913e-03 2.08e-02 1.91e+00 -11.0 1.16e+00  -7.5 1.00e+00 1.25e-01h  4
  56  1.5005794e-03 2.68e-02 4.99e-01 -11.0 3.64e-01  -7.9 1.00e+00 1.00e+00h  1
  57  1.4841053e-03 2.81e-02 4.23e-01 -11.0 3.32e+00  -8.4 1.00e+00 6.25e-02h  5
  58  1.4728442e-03 2.90e-02 3.78e-01 -11.0 4.76e+01  -8.9 1.00e+00 3.91e-03h  9
  59  1.4542513e-03 4.86e-02 2.15e-01 -11.0 2.32e+01  -9.4 1.00e+00 3.12e-02h  6
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  60  1.4491270e-03 5.15e-02 1.10e-01 -11.0 3.81e+01    -  1.00e+00 1.56e-02h  7
  61  1.4240332e-03 9.84e-02 3.48e-01 -11.0 7.61e+01  -9.8 1.00e+00 1.56e-02h  7
  62  1.3910470e-03 2.61e-01 8.50e-01 -11.0 2.53e+02  -8.5 1.00e+00 7.81e-03h  8
  63  1.4140032e-03 2.60e-01 8.65e-01 -11.0 3.08e+01  -8.1 1.00e+00 1.95e-03h 10
  64  1.3817976e-03 1.36e-02 6.78e-01 -11.0 2.33e-01  -8.6 1.00e+00 1.00e+00h  1
  65  1.4391152e-03 1.55e-02 7.12e-01 -11.0 1.04e+00  -9.0 1.00e+00 6.25e-02h  5
  66  1.3653031e-03 2.86e-01 5.24e-01 -11.0 2.29e+00    -  1.00e+00 1.00e+00H  1
  67  1.6261855e-03 2.82e-03 3.48e+00 -11.0 4.16e+00  -9.5 1.00e+00 1.00e+00H  1
  68  1.6125121e-03 6.10e-03 3.64e+00 -11.0 1.09e+02  -8.2 1.00e+00 1.95e-03h 10
  69  1.6018708e-03 7.36e-03 3.75e+00 -11.0 3.32e+01  -7.8 1.00e+00 3.91e-03h  9
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  70  1.5917898e-03 9.49e-03 3.92e+00 -11.0 4.99e+01  -8.2 1.00e+00 3.91e-03h  9
  71  1.5274874e-03 9.56e-02 4.05e+00 -11.0 4.99e+01  -7.8 1.00e+00 7.81e-03h  8
  72  1.5018131e-03 1.11e-01 4.02e+00 -11.0 1.39e+01  -8.3 1.00e+00 7.81e-03h  8
  73  1.5407937e-03 2.54e-02 1.36e+00 -11.0 3.28e+00  -8.8 1.00e+00 1.00e+00H  1
  74  1.5289598e-03 2.57e-02 1.34e+00 -11.0 5.54e+01  -8.3 1.00e+00 9.77e-04h 11
  75  1.3020076e-03 4.98e-01 5.15e-01 -11.0 1.41e+00  -8.8 1.00e+00 1.00e+00h  1
  76  1.2518735e-03 1.02e+00 5.90e-01 -11.0 7.15e+01  -9.3 1.00e+00 3.12e-02h  6
  77  1.2991033e-03 1.02e+00 5.86e-01 -11.0 2.56e+01  -8.0 1.00e+00 3.91e-03h  9
  78  1.1800698e-03 2.49e-02 2.21e-02 -11.0 9.30e-02  -8.4 1.00e+00 1.00e+00h  1
  79  1.2715456e-03 5.02e-04 4.31e-02 -11.0 4.04e-01  -8.9 1.00e+00 1.00e+00H  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  80  1.2626095e-03 2.06e-02 4.03e-02 -11.0 1.14e+01  -9.4 1.00e+00 3.12e-02h  6
  81  1.2565225e-03 2.26e-02 3.94e-02 -11.0 3.70e+01  -9.9 1.00e+00 3.91e-03h  9
  82  1.1739354e-03 1.15e+00 6.91e-02 -11.0 1.87e+02 -10.4 1.00e+00 1.56e-02h  7
  83  1.1506752e-03 1.42e+00 8.43e-02 -11.0 6.80e+02  -9.0 1.00e+00 1.95e-03h 10
  84  1.1425420e-03 1.58e+00 9.79e-02 -11.0 6.90e+01  -7.7 1.00e+00 1.56e-02h  7
  85  1.1314372e-03 2.14e+00 1.24e-01 -11.0 1.93e+03    -  1.00e+00 9.77e-04h 11
  86  1.1025245e-03 2.17e+00 1.22e-01 -11.0 8.75e+01  -8.2 1.00e+00 3.91e-03h  9
  87  1.0885402e-03 2.10e+00 1.07e-01 -11.0 1.25e+01  -8.7 1.00e+00 6.25e-02h  5
  88  1.0117880e-03 3.16e+00 6.80e-02 -11.0 1.86e+02  -9.1 1.00e+00 1.56e-02h  7
  89  1.0054924e-03 3.17e+00 6.47e-02 -11.0 4.96e+02  -8.7 1.00e+00 4.88e-04h 12
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  90  2.6947537e-02 7.06e+01 1.25e+00 -11.0 1.28e+01  -9.2 1.00e+00 1.00e+00w  1
  91  8.4336907e-01 2.88e+02 1.01e+00 -11.0 1.08e+01  -9.7 1.00e+00 1.00e+00w  1
  92  1.1065172e-02 2.57e+02 2.69e+01 -11.0 1.03e+01 -10.1 1.00e+00 1.00e+00w  1
  93  9.6193719e-04 3.08e+00 4.70e-02 -11.0 2.60e+00 -10.6 1.00e+00 6.25e-02h  4
  94  1.0029291e-03 2.99e+00 4.65e-02 -11.0 4.21e+00  -6.6 1.00e+00 3.12e-02h  6
  95  9.9119741e-04 2.26e+00 4.44e-02 -11.0 1.33e+00  -7.0 1.00e+00 2.50e-01h  3
  96  9.5930901e-04 2.01e+00 4.54e-02 -11.0 4.38e+00  -7.5 1.00e+00 1.25e-01h  4
  97  9.3707154e-04 2.01e+00 6.05e-02 -11.0 3.00e+01  -8.0 1.00e+00 1.56e-02h  7
  98  1.0674576e-03 1.57e+00 7.37e-02 -11.0 2.93e+00  -8.5 1.00e+00 2.50e-01h  3
  99  1.1305431e-03 1.40e+00 7.42e-02 -11.0 1.67e+00  -9.0 1.00e+00 2.50e-01h  3
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 100  5.9811862e-03 2.83e-04 1.78e-01 -11.0 2.36e+00  -9.4 1.00e+00 1.00e+00H  1
 101  5.9532935e-03 6.70e-04 1.77e-01 -11.0 7.17e+00  -9.9 1.00e+00 1.95e-03h 10
 102  5.9335562e-03 7.70e-04 1.76e-01 -11.0 1.86e+02 -10.4 1.00e+00 6.10e-05h 15
 103  5.9273391e-03 8.05e-04 1.75e-01 -11.0 2.54e+01 -10.9 1.00e+00 2.44e-04h 13
 104  5.7339296e-03 4.24e+00 7.92e-02 -11.0 3.37e+02    -  1.00e+00 3.91e-03h  9
 105  4.2058230e-03 1.97e+01 1.71e-01 -11.0 4.38e+01 -11.3 1.00e+00 6.25e-02h  5
 106  3.6502498e-03 2.44e+01 3.53e-01 -11.0 7.18e+01    -  1.00e+00 3.12e-02h  6
 107  2.9765617e-03 2.36e+01 4.09e-01 -11.0 1.38e+01 -11.8 1.00e+00 6.25e-02h  5
 108  2.4780218e-03 2.33e+01 3.70e-01 -11.0 2.78e+01 -12.3 1.00e+00 1.56e-02h  7
 109  3.2410377e-03 2.08e+01 2.88e-01 -11.0 3.16e+00 -12.8 1.00e+00 1.25e-01h  4
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 110  1.6463799e-03 1.12e+01 1.86e-01 -11.0 9.71e-01 -13.3 1.00e+00 5.00e-01h  2
 111  2.2013575e-03 1.00e+00 1.79e-01 -11.0 8.20e-01 -13.7 1.00e+00 1.00e+00h  1
 112  1.7800937e-03 1.11e+00 2.34e-01 -11.0 3.55e+00 -14.2 1.00e+00 1.25e-01h  4
 113  8.7250420e-04 4.02e-01 1.01e-01 -11.0 6.33e-01 -14.7 1.00e+00 1.00e+00h  1
 114  1.1346191e-02 1.00e-04 1.70e-01 -11.0 1.26e+00 -15.2 1.00e+00 1.00e+00H  1
 115  8.3580103e-04 2.12e+00 2.61e-01 -11.0 1.17e+00 -15.6 1.00e+00 1.00e+00h  1
 116  7.4180394e-04 2.00e+00 2.41e-01 -11.0 9.88e-01 -16.1 1.00e+00 6.25e-02h  5
 117  9.1374518e-04 3.18e-02 3.09e-02 -11.0 1.53e-01 -16.6 1.00e+00 1.00e+00h  1
 118  5.6730079e-04 1.00e-02 1.63e-02 -11.0 9.83e-02 -17.1 1.00e+00 1.00e+00h  1
 119  5.4857841e-04 2.10e-04 4.92e-03 -11.0 3.24e-02 -17.5 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 120  5.3172669e-04 8.47e-03 2.97e-03 -11.0 6.76e-02 -18.0 1.00e+00 1.00e+00h  1
 121  5.2111796e-04 2.48e-03 1.21e-03 -11.0 2.53e-02 -18.5 1.00e+00 1.00e+00h  1
 122  5.0444046e-04 1.37e-02 2.07e-03 -11.0 5.58e-02 -19.0 1.00e+00 1.00e+00h  1
 123  4.9985385e-04 1.48e-02 2.62e-03 -11.0 1.18e+00 -19.5 1.00e+00 1.56e-02h  7
 124  4.7656002e-04 1.28e-01 6.58e-03 -11.0 1.88e-01 -19.9 1.00e+00 1.00e+00h  1
 125  4.7282247e-04 1.71e-01 1.02e-02 -11.0 7.69e+00 -20.0 1.00e+00 1.56e-02h  7
 126  9.9100306e-04 4.04e-05 8.97e-03 -11.0 3.64e-01 -20.0 1.00e+00 1.00e+00H  1
 127  1.6708487e-03 1.73e-05 1.42e-02 -11.0 5.05e-01 -20.0 1.00e+00 1.00e+00H  1
 128  5.9600736e-04 2.57e-04 4.67e-02 -11.0 5.72e-01 -20.0 1.00e+00 1.00e+00H  1
 129  4.9008439e-04 2.53e-02 3.14e-02 -11.0 1.09e+00 -20.0 1.00e+00 1.25e-01h  4
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 130  1.0294183e-02 1.67e-06 5.35e-02 -11.0 6.71e-01 -20.0 1.00e+00 1.00e+00H  1
 131  4.2873958e-04 9.27e-01 8.36e-02 -11.0 6.16e-01 -20.0 1.00e+00 1.00e+00f  1
 132  4.2294340e-04 7.64e-03 1.85e-03 -11.0 4.46e-02 -20.0 1.00e+00 1.00e+00h  1
 133  4.1881450e-04 1.48e-03 1.16e-03 -11.0 3.32e-02 -20.0 1.00e+00 1.00e+00h  1
 134  4.2806012e-04 1.51e-04 1.84e-03 -11.0 5.30e-02 -20.0 1.00e+00 1.00e+00H  1
 135  4.0845653e-04 4.14e-03 1.55e-03 -11.0 3.13e-02 -20.0 1.00e+00 1.00e+00h  1
 136  4.0519894e-04 9.07e-04 7.33e-04 -11.0 2.08e-02 -20.0 1.00e+00 1.00e+00h  1
 137  3.9923821e-04 4.87e-03 8.18e-04 -11.0 3.28e-02 -20.0 1.00e+00 1.00e+00h  1
 138  3.9066354e-04 2.69e-02 1.64e-03 -11.0 8.91e-02 -20.0 1.00e+00 1.00e+00h  1
 139  3.8838838e-04 3.08e-02 2.17e-03 -11.0 1.01e+01 -20.0 1.00e+00 3.91e-03h  9
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 140  4.2124946e-04 3.56e-04 5.75e-03 -11.0 3.41e-01 -20.0 1.00e+00 1.00e+00H  1
 141  4.1921019e-04 6.71e-04 5.68e-03 -11.0 1.61e+00 -20.0 1.00e+00 7.81e-03h  8
 142  5.3038631e-04 7.47e-05 2.69e-02 -11.0 2.29e-01 -20.0 1.00e+00 1.00e+00H  1
 143  5.2957958e-04 7.93e-05 2.68e-02 -11.0 7.08e-01 -20.0 1.00e+00 1.95e-03h 10
 144  5.2839798e-04 8.13e-05 2.68e-02 -11.0 2.63e+00 -20.0 1.00e+00 4.88e-04h 12
 145  9.6544771e-04 5.17e-06 2.30e-02 -11.0 2.53e-01 -20.0 1.00e+00 1.00e+00H  1
 146  7.2504861e-04 1.17e-04 2.41e-02 -11.0 2.34e-01 -20.0 1.00e+00 1.00e+00H  1
 147  7.2455205e-04 1.17e-04 2.41e-02 -11.0 2.56e+00 -20.0 1.00e+00 1.22e-04h 14
 148  7.2421752e-04 1.17e-04 2.41e-02 -11.0 1.55e+00 -20.0 1.00e+00 2.44e-04h 13
 149  7.2354519e-04 1.17e-04 2.41e-02 -11.0 3.23e+00 -20.0 1.00e+00 1.22e-04h 14
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 150  4.1352727e-04 5.36e-04 3.78e-02 -11.0 7.96e-01 -20.0 1.00e+00 1.00e+00H  1
 151  4.0506034e-04 4.47e-02 3.36e-02 -11.0 1.59e+00 -20.0 1.00e+00 6.25e-02h  5
 152  4.5925550e-04 1.24e-04 1.59e-02 -11.0 5.89e-01 -20.0 1.00e+00 1.00e+00H  1
 153  4.5715465e-04 3.50e-04 1.60e-02 -11.0 2.20e+00 -20.0 1.00e+00 3.91e-03h  9
 154  4.5670646e-04 3.72e-04 1.60e-02 -11.0 5.48e+00 -20.0 1.00e+00 4.88e-04h 12
 155  4.5585963e-04 3.83e-04 1.60e-02 -11.0 1.03e+00 -20.0 1.00e+00 1.95e-03h 10
 156  4.5515234e-04 3.86e-04 1.59e-02 -11.0 2.84e+00 -20.0 1.00e+00 4.88e-04h 12
 157  4.4010910e-04 9.95e-02 1.24e-02 -11.0 5.33e+00 -20.0 1.00e+00 3.12e-02h  6
 158  9.9923187e-03 1.90e-04 9.26e-02 -11.0 8.64e-01 -20.0 1.00e+00 1.00e+00H  1
 159  4.1870354e-04 2.05e+00 1.06e-01 -11.0 7.47e-01 -20.0 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 160  4.3645121e-04 1.09e+00 5.88e-02 -11.0 4.99e-01 -20.0 1.00e+00 5.00e-01h  2
 161  4.2614047e-04 9.58e-01 5.26e-02 -11.0 2.98e-01 -20.0 1.00e+00 1.25e-01h  4
 162  3.5452058e-04 2.66e-01 4.25e-02 -11.0 2.83e-01 -20.0 1.00e+00 1.00e+00h  1
 163  4.2044871e-03 3.83e-04 1.63e-01 -11.0 1.08e+00 -20.0 1.00e+00 1.00e+00H  1
 164  2.8649128e-03 1.57e-04 7.08e-02 -11.0 6.22e-01 -20.0 1.00e+00 1.00e+00H  1
 165  2.8430990e-03 2.17e-04 7.07e-02 -11.0 2.49e+00 -20.0 1.00e+00 1.95e-03h 10
 166  2.8108576e-03 2.79e-04 7.04e-02 -11.0 1.17e+00 -20.0 1.00e+00 3.91e-03h  9
 167  1.8035301e-03 5.96e-04 2.79e-01 -11.0 9.59e-01 -20.0 1.00e+00 1.00e+00H  1
 168  1.6288058e-02 1.47e-04 1.66e-01 -11.0 9.91e-01 -20.0 1.00e+00 1.00e+00H  1
 169  1.6662131e-03 1.23e+00 4.27e-01 -11.0 1.03e+00 -20.0 1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 170  1.6180817e-03 1.39e+00 2.68e-01 -11.0 5.80e-01 -20.0 1.00e+00 1.00e+00h  1
 171  1.6403535e-03 1.93e-01 7.49e-02 -11.0 7.46e-01 -20.0 1.00e+00 1.00e+00H  1
 172  1.4013576e-03 4.47e-01 5.98e-02 -11.0 1.39e+00 -20.0 1.00e+00 2.50e-01h  3
 173  9.1043739e-04 9.56e-01 5.91e-02 -11.0 9.69e-01 -20.0 1.00e+00 5.00e-01h  2
 174  7.5527711e-04 1.18e+00 5.08e-02 -11.0 8.33e+00 -20.0 1.00e+00 3.12e-02h  6
 175  1.6722038e-03 2.05e-02 6.22e-02 -11.0 7.31e-01 -20.0 1.00e+00 1.00e+00H  1
 176  5.0909201e-03 8.50e-05 1.20e-01 -11.0 5.92e-01 -20.0 1.00e+00 1.00e+00H  1
 177  4.8834600e-03 1.64e-04 5.59e-01 -11.0 8.53e-01 -20.0 1.00e+00 1.00e+00H  1
 178  2.2243972e-03 3.25e-04 7.03e-01 -11.0 9.06e-01 -20.0 1.00e+00 1.00e+00H  1
 179  1.5541698e-03 2.97e-01 2.19e-01 -11.0 3.37e-01 -20.0 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 180  3.5511382e-04 1.13e-01 4.78e-02 -11.0 1.80e-01 -20.0 1.00e+00 1.00e+00h  1
 181  8.8598915e-04 1.88e-02 1.45e-02 -11.0 2.42e-01 -20.0 1.00e+00 1.00e+00h  1
 182  3.4147843e-04 1.67e-02 5.45e-03 -11.0 2.18e-01 -20.0 1.00e+00 1.00e+00h  1
 183  3.3982834e-04 1.94e-04 4.22e-04 -11.0 7.81e-03 -20.0 1.00e+00 1.00e+00h  1
 184  4.1759835e-04 2.96e-04 2.03e-03 -11.0 9.25e-02 -20.0 1.00e+00 1.00e+00H  1
 185  3.4061626e-04 8.81e-05 3.10e-03 -11.0 7.27e-02 -20.0 1.00e+00 1.00e+00H  1
 186  3.6681976e-04 2.67e-05 8.94e-04 -11.0 3.78e-02 -20.0 1.00e+00 1.00e+00H  1
 187  3.3732163e-04 1.50e-03 2.96e-03 -11.0 2.71e-02 -20.0 1.00e+00 1.00e+00h  1
 188  3.3713850e-04 5.53e-06 1.37e-05 -11.0 1.34e-03 -20.0 1.00e+00 1.00e+00h  1

Number of Iterations....: 188

                                   (scaled)                 (unscaled)
Objective...............:   3.3713849838351302e-04    3.3713849838351302e-04
Dual infeasibility......:   1.3682520740526252e-05    1.3682520740526252e-05
Constraint violation....:   6.1529343751220556e-07    5.5271591464789438e-06
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
Overall NLP error.......:   7.7956386991585388e-07    1.3682520740526252e-05


Number of objective function evaluations             = 1036
Number of objective gradient evaluations             = 189
Number of equality constraint evaluations            = 1096
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 189
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 0
Total seconds in IPOPT                               = 38.201

EXIT: Optimal Solution Found.
           S  :   t_proc      (avg)   t_wall      (avg)    n_eval
       nlp_f  | 160.41ms (154.84us)  11.22ms ( 10.83us)      1036
       nlp_g  |  38.46 s ( 35.09ms)   2.68 s (  2.44ms)      1096
  nlp_grad_f  |  73.58ms (387.27us)   5.65ms ( 29.72us)       190
   nlp_jac_g  |  73.03 s (384.34ms)   5.71 s ( 30.04ms)       190
       total  | 510.73 s (510.73 s)  38.20 s ( 38.20 s)         1
"""