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
from pinocchio import casadi as cpin

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

    # Add floor visual geometry:
    floor_obj = pin.GeometryObject("floor", 0, 0, hppfcl.Box(2, 2, 0.005), pin.SE3.Identity())
    visualizer.loadViewerGeometryObject(floor_obj, pin.GeometryType.VISUAL, np.array([0.3, 0.3, 0.3, 1]))
    
    floor_obj_name = visualizer.getViewerNodeName(floor_obj, pin.GeometryType.VISUAL)

    # Manually set the transform because the GeometryObject() constructor doesn't work:
    visualizer.viewer[floor_obj_name].set_transform(
        pin.SE3(np.eye(3), np.array([0, 0, floor_z])).homogeneous
    )

    visualizer.display(pin.neutral(limited_robot.model))
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

class ADFootholdKinematics():
    def __init__(self, cmodel, cdata, feet: list[str]):
        self.cmodel = cmodel
        self.cdata = cdata
        self.ff_ids = [robot.model.getFrameId(f) for f in feet]

    # foot frame positions (oMf), velocities, accelerations (in local world-aligned coords)
    def __call__(self, q: ca.SX, v: ca.SX, a: ca.SX):
        # This runs the second-order FK algorithm and returns:
        # - The positions of all feet wrt the origin
        # - The velocities of all feet in the local world-aligned frame
        # - The accelerations of all feet in the local world-aligned frame

        cpin.forwardKinematics(cmodel, cdata, q, v, a)
        cpin.updateFramePlacements(cmodel, cdata)

        get_pos = lambda fid: cdata.oMf[fid].translation
        get_vel = lambda fid: cpin.getFrameVelocity(cmodel, cdata, fid, pin.LOCAL_WORLD_ALIGNED).linear
        get_acc = lambda fid: cpin.getFrameAcceleration(cmodel, cdata, fid, pin.LOCAL_WORLD_ALIGNED).linear

        return (
            ca.vertcat(*iter(get_pos(f) for f in self.ff_ids)),
            ca.vertcat(*iter(get_vel(f) for f in self.ff_ids)),
            ca.vertcat(*iter(get_acc(f) for f in self.ff_ids))
        )

class ADForwardDynamics():
  def __init__(self, cmodel, cdata, joints: list[str], feet: list[str]):
    self.cmodel, self.cdata = cmodel, cdata
    self.joint_frame_ids = [cmodel.getFrameId(j) for j in joints]
    self.foot_frame_ids = [cmodel.getFrameId(f) for f in feet]

  def __call__(self, q: ca.SX, v: ca.SX, τ: ca.SX, λ: ca.SX):
    # λ contains normal GRFs for each foot.
    # Find how they're expressed in the joint frames at the provided
    # robot state. FK will populate robot.data.oMf.
    grf_at_joints = [cpin.Force.Zero()]

    cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
    cpin.updateFramePlacements(self.cmodel, self.cdata)

    for j_fr_id in self.joint_frame_ids:
        total_grf = cpin.Force.Zero()
        X_o_joint = self.cdata.oMf[j_fr_id]

        for f_idx, f_fr_id in enumerate(self.foot_frame_ids):
            X_o_foot = self.cdata.oMf[f_fr_id]
            X_o_foot_world_aligned = cpin.SE3(ca.SX.eye(3), X_o_foot.translation)

            # We assume the GRF always points Z-up (it's expressed in the world aligned frame)
            lin = ca.SX.zeros(3)
            lin[2] = λ[0][f_idx]
            grf_at_foot = cpin.Force(lin, ca.SX.zeros(3))

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
    return cpin.aba(self.cmodel, self.cdata, q, v, τ, grf_at_joints)
  
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

    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()

    # This is insane: @1=0.1946, ((1842.91*(τ+((@1*λ)-((@1+(-0.16*sin(q)))*λ))))+(-7.09199*(9.81*sin(q))))
    fd = ADForwardDynamics(cmodel, cdata, joints = JOINTS, feet = FEET)

    # Same for this:
    # (SX(@1=-0.16, [(0.1946+(@1*sin(q))), -0.14695, (@1+(@1*cos(q)))]), SX(@1=(-0.16*v), [(cos(q)*@1), 0, (-(sin(q)*@1))]), SX(@1=(-0.16*a), [(cos(q)*@1), 0, (-(sin(q)*@1))]))
    fk = ADFootholdKinematics(cmodel, cdata, FEET)

    if not os.path.exists(OUTPUT_BIN):
        constraints = []

        # Create a new inequality constraint: expr >= 0
        def add_inequality(expr: ca.SX):
            constraints.append((expr, 0, np.inf))

        # Create a new equality constraint: expr == 0
        def add_equality(expr: ca.SX):
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
            q_k.append(ca.SX.sym(f"q_{k}"))
            v_k.append(ca.SX.sym(f"v_{k}"))
            a_k.append(ca.SX.sym(f"a_{k}"))
            tau_k.append(ca.SX.sym(f"τ_{k}"))

            λ_k.append(ca.SX.sym(f"λ_{k}"))
            foot_z_k.append(ca.SX.sym(f"foot_z_{k}"))

            #### DYNAMICS CONSTRAINTS ####
            # Residual constraints for accelerations (= 0) at all collocation points:
            add_equality(a_k[k] - fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))
            ##############################

            print(fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))

            J = ca.Function(
                "J",
                [q_k[k], v_k[k], tau_k[k], λ_k[k]],
                [ca.jacobian(fd(q_k[k], v_k[k], tau_k[k], λ_k[k]), q_k[k])]
            )

            print(fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))
            print(J(q_k[k], v_k[k], tau_k[k], λ_k[k]))
            exit()


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
        q_sym, v_sym, a_sym = ca.SX.sym("q_sym"), ca.SX.sym("v_sym"), ca.SX.sym("a_sym")
        num_fk = ca.Function("numerical_fk", [q_sym, v_sym, a_sym], fk(q_sym, v_sym, a_sym))

        min_leg_z, max_leg_z = float(num_fk(0, 0, 0)[0][2]), float(num_fk(np.pi/2, 0, 0)[0][2])
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
            fz_g.append(float(num_fk(q_g[-1], 0, 0)[0][2]))
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

# WITH AUTODIFF:
"""
******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.16, running with linear solver ma57.

Number of nonzeros in equality constraint Jacobian...:     1561
Number of nonzeros in inequality constraint Jacobian.:      101
Number of nonzeros in Lagrangian Hessian.............:      328

Reallocating memory for MA57: lfact (17489)
Total number of variables............................:      606
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:      532
Total number of inequality constraints...............:      101
        inequality constraints with only lower bounds:      101
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  0.0000000e+00 6.96e+01 1.00e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  8.7705310e-01 4.71e+01 3.72e+00  -1.0 1.32e+04    -  6.03e-01 1.00e+00f  1
   2  1.0644076e+00 9.20e+01 5.56e+00  -1.0 6.25e+03    -  4.30e-01 1.00e+00f  1
   3  1.0762615e+00 1.39e-05 6.28e-02  -1.0 6.53e-01  -4.0 9.91e-01 1.00e+00h  1
   4  6.0584434e-01 5.70e+00 1.28e+00  -2.5 3.47e+03    -  8.92e-01 1.00e+00f  1
   5  1.0667461e-01 9.53e+01 5.81e-01  -2.5 5.11e+03    -  7.90e-01 1.00e+00h  1
   6  8.6669577e-02 2.90e+00 1.49e-03  -2.5 1.02e+03    -  1.00e+00 1.00e+00h  1
   7  1.0098232e-01 3.16e+00 6.73e-04  -2.5 5.10e+02    -  1.00e+00 1.00e+00h  1
   8  9.7770004e-02 1.07e-01 2.01e-05  -2.5 8.62e+01    -  1.00e+00 1.00e+00h  1
   9  3.3745470e-02 7.32e+00 1.15e-03  -3.8 1.25e+03    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  9.4872039e-03 2.03e+01 2.51e-04  -3.8 7.10e+02    -  1.00e+00 1.00e+00h  1
  11  5.7913790e-03 1.02e+00 2.65e-05  -3.8 2.03e+02    -  1.00e+00 1.00e+00h  1
  12  5.5441812e-03 6.10e-03 5.84e-07  -3.8 2.44e+01    -  1.00e+00 1.00e+00h  1
  13  1.5096078e-03 5.49e+00 6.61e-05  -5.7 3.48e+02    -  1.00e+00 1.00e+00h  1
  14  6.6664087e-04 1.70e+00 1.58e-05  -5.7 1.72e+02    -  1.00e+00 1.00e+00h  1
  15  5.3176554e-04 1.71e-01 3.15e-06  -5.7 6.90e+01    -  1.00e+00 1.00e+00h  1
  16  5.1389933e-04 5.46e-03 3.06e-07  -5.7 1.80e+01    -  1.00e+00 1.00e+00h  1
  17  5.1255975e-04 2.10e-05 3.71e-09  -5.7 1.63e+00    -  1.00e+00 1.00e+00h  1
  18  5.0018573e-04 2.16e-02 6.18e-07  -8.6 2.82e+01    -  1.00e+00 1.00e+00h  1
  19  4.9966147e-04 2.90e-04 3.94e-08  -8.6 5.34e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  4.9965784e-04 2.59e-07 2.49e-10  -8.6 2.39e-01    -  1.00e+00 1.00e+00h  1
  21  4.9965780e-04 2.68e-08 1.62e-12  -9.0 3.72e-02    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.9965780084401245e-04    4.9965780084401245e-04
Dual infeasibility......:   1.6231630839760869e-12    1.6231630839760869e-12
Constraint violation....:   1.4527197314362285e-09    2.6772360328664032e-08
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   9.0998898014423950e-10    9.0998898014423950e-10
Overall NLP error.......:   1.4527197314362285e-09    2.6772360328664032e-08


Number of objective function evaluations             = 22
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 22
Number of inequality constraint evaluations          = 22
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 0.031

EXIT: Optimal Solution Found.
           S  :   t_proc      (avg)   t_wall      (avg)    n_eval
       nlp_f  | 120.00us (  5.45us) 126.55us (  5.75us)        22
       nlp_g  | 736.00us ( 33.45us) 762.24us ( 34.65us)        22
  nlp_grad_f  | 223.00us (  9.70us) 237.90us ( 10.34us)        23
  nlp_hess_l  | 455.00us ( 21.67us) 486.22us ( 23.15us)        21
   nlp_jac_g  | 947.00us ( 41.17us)   1.03ms ( 44.66us)        23
       total  |  29.58ms ( 29.58ms)  32.04ms ( 32.04ms)         1

"""

# WITHOUT AUTODIFF + UNOPTIMIZED FUNCTION CALLING:
"""
******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.16, running with linear solver ma57.

Number of nonzeros in equality constraint Jacobian...:     1889
Number of nonzeros in inequality constraint Jacobian.:      101
Number of nonzeros in Lagrangian Hessian.............:     1313

Total number of variables............................:      606
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:      532
Total number of inequality constraints...............:      101
        inequality constraints with only lower bounds:      101
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  0.0000000e+00 6.96e+01 1.00e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
Reallocating memory for MA57: lfact (24719)
   1  8.7705310e-01 4.71e+01 3.72e+00  -1.0 1.32e+04    -  6.03e-01 1.00e+00f  1
   2  1.0644326e+00 9.20e+01 5.56e+00  -1.0 6.25e+03    -  4.30e-01 1.00e+00f  1
   3  1.0762869e+00 1.39e-05 6.28e-02  -1.0 6.53e-01  -4.0 9.91e-01 1.00e+00h  1
   4  6.0585935e-01 5.69e+00 1.28e+00  -2.5 3.47e+03    -  8.92e-01 1.00e+00f  1
   5  1.0666545e-01 9.53e+01 5.82e-01  -2.5 5.11e+03    -  7.90e-01 1.00e+00h  1
   6  8.6669948e-02 2.90e+00 1.49e-03  -2.5 1.02e+03    -  1.00e+00 1.00e+00h  1
   7  1.0098272e-01 3.16e+00 6.74e-04  -2.5 5.10e+02    -  1.00e+00 1.00e+00h  1
   8  9.7770333e-02 1.07e-01 2.01e-05  -2.5 8.62e+01    -  1.00e+00 1.00e+00h  1
   9  3.3744276e-02 7.32e+00 1.15e-03  -3.8 1.25e+03    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  9.4869361e-03 2.03e+01 2.51e-04  -3.8 7.10e+02    -  1.00e+00 1.00e+00h  1
  11  5.7913535e-03 1.02e+00 2.65e-05  -3.8 2.03e+02    -  1.00e+00 1.00e+00h  1
  12  5.5441842e-03 6.10e-03 5.85e-07  -3.8 2.44e+01    -  1.00e+00 1.00e+00h  1
  13  1.5096277e-03 5.49e+00 6.61e-05  -5.7 3.48e+02    -  1.00e+00 1.00e+00h  1
  14  6.6664077e-04 1.70e+00 1.58e-05  -5.7 1.72e+02    -  1.00e+00 1.00e+00h  1
  15  5.3176559e-04 1.71e-01 3.15e-06  -5.7 6.90e+01    -  1.00e+00 1.00e+00h  1
  16  5.1389884e-04 5.46e-03 3.06e-07  -5.7 1.80e+01    -  1.00e+00 1.00e+00h  1
  17  5.1255974e-04 2.10e-05 3.71e-09  -5.7 1.63e+00    -  1.00e+00 1.00e+00h  1
  18  5.0018573e-04 2.16e-02 6.18e-07  -8.6 2.82e+01    -  1.00e+00 1.00e+00h  1
  19  4.9966147e-04 2.90e-04 3.94e-08  -8.6 5.34e+00    -  1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  4.9965784e-04 2.57e-07 2.49e-10  -8.6 2.39e-01    -  1.00e+00 1.00e+00h  1
  21  4.9965780e-04 2.68e-08 1.66e-12  -9.0 3.72e-02    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 21

                                   (scaled)                 (unscaled)
Objective...............:   4.9965780084377240e-04    4.9965780084377240e-04
Dual infeasibility......:   1.6553841741968393e-12    1.6553841741968393e-12
Constraint violation....:   1.4518105947599545e-09    2.6755605730954812e-08
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   9.0998898014466643e-10    9.0998898014466643e-10
Overall NLP error.......:   1.4518105947599545e-09    2.6755605730954812e-08


Number of objective function evaluations             = 22
Number of objective gradient evaluations             = 22
Number of equality constraint evaluations            = 22
Number of inequality constraint evaluations          = 22
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 22
Number of Lagrangian Hessian evaluations             = 21
Total seconds in IPOPT                               = 74.030
"""