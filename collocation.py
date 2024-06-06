import pickle
import argparse

import knitro
import casadi as ca
from pinocchio import casadi as cpin

from poses import Pose
from robot import load_solo12
from guesses import *
from visualisation import visualise_solution
from utilities import integrate_state, flatten

from transcription import Constraint, VariableBounds
from dynamics import ADForwardDynamics
from kinematics import ADFootholdKinematics

from tasks import *

if __name__ == "__main__":
    TASK = BACKFLIP_LAUNCH_TASK

    parser = argparse.ArgumentParser()
    parser.add_argument('--visualise', action='store_true')
    options = parser.parse_args()

    # The order of forces per foot WILL be as in this list:
    FEET = ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]

    MU = 0.7
    BAUMGARTE_ALPHA = 2
    FREQ_HZ = 30
    DELTA_T = 1 / FREQ_HZ
    FLOOR_Z = -0.226274
    N_KNOTS = int(TASK.duration * FREQ_HZ)
    OUTPUT_FILENAME = f"{TASK.name}_{FREQ_HZ}hz_{int(TASK.duration*1e+3)}ms.bin"

    # Load robot model:
    robot, viz = load_solo12(floor_z = FLOOR_Z, visualise = options.visualise)
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()

    # Skip 'universe' and 'root_joint' as they're not actuated:
    actuated_joints = [j.id for j in robot.model.joints[2:]]

    # Instantiate dynamics and kinematics routines:
    fd = ADForwardDynamics(cmodel, cdata, feet = FEET, act_joint_ids = actuated_joints)
    fk = ADFootholdKinematics(cmodel, cdata, feet = FEET)

    if options.visualise:
        visualise_solution(OUTPUT_FILENAME, N_KNOTS, DELTA_T, robot)
        exit()

    #region Collocation variables
    print("Creating collocation decision variables...")
    q_k, v_k, a_k, tau_k, λ_k, f_pos_k, f_vel_k, f_acc_k = [], [], [], [], [], [], [], []

    for k in range(N_KNOTS):
        q_k.append(ca.SX.sym(f"q_{k}", robot.nq - 1))   # We will represent orientations with MRP instead of quaternions
        v_k.append(ca.SX.sym(f"v_{k}", robot.nv))       # 18 x 1
        a_k.append(ca.SX.sym(f"a_{k}", robot.nv))       # 18 x 1

        tau_k.append(ca.SX.sym(f"τ_{k}", len(actuated_joints)))  # 12 x 1 
        λ_k.append(ca.SX.sym(f"λ_{k}", len(FEET), 3))            # 4  x 3

        f_pos_k.append(ca.SX.sym(f"f_pos_{k}", len(FEET), 3))    # 4 x 3
        f_vel_k.append(ca.SX.sym(f"f_vel_{k}", len(FEET), 3))    # 4 x 3
        f_acc_k.append(ca.SX.sym(f"f_acc_{k}", len(FEET), 3))    # 4 x 3
        
    #endregion

    #region Constraints and bounds
    print("Creating constraints and bounds...")
    constraints = []
    bounds = VariableBounds()

    for k in range(N_KNOTS):
        t = k * DELTA_T
        f_pos, f_vel, f_acc = fk(q_k[k], v_k[k], a_k[k])

        # Pointwise constraints (dynamics, kinematics, limits):
        # =========================================
        constraints.append(
            # Forward dynamics accelerations:
            Constraint(a_k[k] - fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))
        )

        # Robot torso cannot go below the ground:
        bounds.add_expr(q_k[k][2], lb = FLOOR_Z + 0.08, ub = ca.inf)

        # # Joint torque limits in N*m:
        # bounds.add_expr(tau_k[k], lb = -2, ub = 2)

        # Forward foothold kinematics:
        constraints.append(Constraint(f_pos_k[k] - f_pos))
        constraints.append(Constraint(f_vel_k[k] - f_vel))
        constraints.append(Constraint(f_acc_k[k] - f_acc))
        
        # Integration constraints:
        # ========================
        if k > 0:
            constraints.append(
                # Euler step for velocities:
                Constraint(v_k[k] - v_k[k-1] - DELTA_T * a_k[k-1])
            )

            constraints.append(
                # Similar for state, but manually handle SE3 integration: 
                Constraint(q_k[k] - integrate_state(q_k[k-1], DELTA_T * v_k[k-1]))
            )

        # Feet contact constraints:
        # =========================
        for foot_idx in range(len(FEET)):
            contact_ivt = TASK.contact_periods[foot_idx]

            # Contact Forces
            # --------------
            # If the foot is on the ground, contact forces should be available.
            # They must be such that the foot position constraints are satisfied:
            #   1. fx, fy = const
            #   2. fz = FLOOR_Z = const
            # The forces need to be computed _for that contact knot_, as they needs to balance
            # other second order phenomena such as gravity and torque in the dynamics equations!
            # If we measure the violation by seeing how fx, fy, fz change between the current
            # and the next knot, the GRFs might be inaccurate as accelerations take two knots
            # to cause an effect. For example, before the last contact knot ANY force can be
            # applied as the acceleration will not change the foot placement at all.
            # At that point, the torques required to keep the body up are not accurate.

            if contact_ivt.overlaps(t):
                # Z contact force must be pointing up (repulsive).
                # TODO: Add limit, normalise by Δt?
                bounds.add_expr(λ_k[k][foot_idx, 2], lb = 0.0, ub = ca.inf)

                # Friction cone constraints (pyramidal approximation):
                constraints.append(
                    Constraint(
                        # abs(fx) <= fz * μ:
                        MU * λ_k[k][foot_idx, 2] - ca.fabs(λ_k[k][foot_idx, 0]),
                        lb = 0.0, ub = ca.inf
                    )
                )

                constraints.append(
                    Constraint(
                        # abs(fy) <= fz * μ
                        MU * λ_k[k][foot_idx, 2] - ca.fabs(λ_k[k][foot_idx, 1]),
                        lb = 0.0, ub = ca.inf
                    )
                )

                # Foothold constraint: the feet are on the ground and aren't slipping:
                #   [f_x, f_y, f_z] - [fc_x, fc_y, fc_z = FLOOR_Z] = 0  \forall t,
                # where `fc` is the foot contact point.
                # There are two strategies for handling this constraint:
                #
                #   1. We can set its acceleration to zero and find forces that achieve that.
                #      To make sure the position constraint holds numerically, we can
                #      dampen the system using Baugmarte stabilization:
                #           f_acc_k[k] = -2*α*f_vel_k[k] - α^2 *(f_pos_k[k] - [...])
                #
                #      This sadly didn't work too well as the damping wasn't
                #      enough to prevent the feet from falling into the ground.
                #      It maybe is tricky with a coarsely discretized problem?
                #      Does the constraint f_grf_z >= 0 make it extra difficult?
                #
                #   2. We can enforce the constraint directly on the foot kinematics
                #      variables: f_pos_k[k] = const = f_pos_k[c_k] and
                #      f_pos_k[c_k][2] = FLOOR_Z.
                #      To make sure that torques are always accurate and no funky
                #      business is happening with the integrator (as described
                #      above), we can also set the derivatives of the constraint to zero,
                #      using second-order kinematics.
                #      NOTE: This seems to work only with exact Hessians.

                # Find the knot where contact starts:
                c_k = math.ceil(next(iter(contact_ivt.at(t)))[0] * FREQ_HZ - ε)

                target_pos = ca.horzcat(
                    f_pos_k[c_k][foot_idx, 0],
                    f_pos_k[c_k][foot_idx, 1],
                    FLOOR_Z
                )

                # Sadly, these constraints still make the optimization difficult...
                # I'm not sure what's the problem, maybe it's struggling to lift the legs
                # at the last knot? Even removing the >= FLOOR_Z constraint below doesn't
                # work...
                constraints.append(Constraint(f_pos_k[k][foot_idx, :] - target_pos))
                constraints.append(Constraint(f_vel_k[k][foot_idx, :]))
                constraints.append(Constraint(f_acc_k[k][foot_idx, :]))

            else:
                # No contact forces available:
                bounds.add_expr(λ_k[k][foot_idx, :], lb = 0, ub = 0)

                # If not in contact, foot X and Y are free, and Z >= floor:
                bounds.add_expr(f_pos_k[k][foot_idx, 2], lb = FLOOR_Z, ub = ca.inf)

    #endregion

    #region Objective
    print("Creating optimization objective...")

    # Integrate trajectory error to minimize, normalised by the trajectory duration:
    objective = sum(
        DELTA_T / TASK.duration * 
            TASK.traj_error(k * DELTA_T, q_k[k], v_k[k], a_k[k], tau_k[k], λ_k[k])

        for k in range(N_KNOTS)
    )

    #endregion

    #region Kinematic constraints
    print("Adding trajectory kinematic constraints...")
    constraints += TASK.get_kinematic_constraints(
        q_k, v_k, a_k, f_pos_k, {"FLOOR_Z": FLOOR_Z, "FREQ_HZ": FREQ_HZ, "N_KNOTS": N_KNOTS}
    )
    #############################################################

    #endregion

    #region Problem and solver setup
    print("Creating NLP description...")

    decision_vars = ca.vertcat(
        *q_k, *v_k, *a_k, *tau_k, flatten(λ_k),
        flatten(f_pos_k), flatten(f_vel_k), flatten(f_acc_k)
    )
    
    problem = {
        # Decision variables:
        "x": decision_vars,

        # Optimisation objective:
        "f": objective,

        # Constrained expressions:
        "g": flatten([c.expr for c in constraints])
    }

    print("Instantiating solver...")
    
    knitro_settings = {
        # "hessopt":      knitro.KN_HESSOPT_LBFGS,
        "algorithm":    knitro.KN_ALG_BAR_DIRECT,
        "bar_murule":   knitro.KN_BAR_MURULE_ADAPTIVE,
        "linsolver":    knitro.KN_LINSOLVER_MA57,
        "feastol":      1e-3,
        "ftol":         1e-4,
        # "bar_feasible": knitro.KN_BAR_FEASIBLE_GET_STAY,
        # "ms_enable":    True,
        # "ms_numthreads": 8

    }

    solver = ca.nlpsol(
        "S", "knitro", problem, {"knitro": knitro_settings} #, "verbose": True}
    )

    #endregion


    #region Solution
    print("Calling solver...")

    x_bounds = [
        bounds.get_bounds(decision_vars[v_idx].name())
        for v_idx in range(decision_vars.shape[0])
    ]

    soln = solver(
        x0  = const_pose_guess(N_KNOTS, fk, Pose.STANDING_V).flatten(),
        # x0  = prev_soln_guess(N_KNOTS, robot, "trajectories/backflip_launch_20hz_1000ms.bin").flatten(),

        # x0  = prev_soln_guess(
        #     80, robot, "trajectories/backflip_land_80hz_1000ms.bin", interp_knots = N_KNOTS
        # ).flatten(),

        lbg = flatten([c.lb for c in constraints]),
        ubg = flatten([c.ub for c in constraints]),
        lbx = [b[0] for b in x_bounds],
        ubx = [b[1] for b in x_bounds],
    )

    with open(OUTPUT_FILENAME, "wb") as wf:
        pickle.dump(soln, wf)
    