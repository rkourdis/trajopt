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
    TASK = JUMP_TASK

    parser = argparse.ArgumentParser()
    parser.add_argument('--visualise', action='store_true')
    options = parser.parse_args()

    # The order of forces per foot WILL be as in this list:
    FEET = ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]

    MU = 0.7
    FREQ_HZ = 40
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
    q_k, v_k, a_k, tau_k, λ_k, f_pos_k = [], [], [], [], [], []

    for k in range(N_KNOTS):
        q_k.append(ca.SX.sym(f"q_{k}", robot.nq - 1))   # We will represent orientations with MRP instead of quaternions
        v_k.append(ca.SX.sym(f"v_{k}", robot.nv))       # 18 x 1
        a_k.append(ca.SX.sym(f"a_{k}", robot.nv))       # 18 x 1

        tau_k.append(ca.SX.sym(f"τ_{k}", len(actuated_joints)))  # 12 x 1 
        λ_k.append(ca.SX.sym(f"λ_{k}", len(FEET), 3))            # 4  x 3
        f_pos_k.append(ca.SX.sym(f"f_pos_{k}", len(FEET), 3))    # 4 x 3
        
    #endregion

    #region Constraints and bounds
    print("Creating constraints and bounds...")
    constraints = []
    bounds = VariableBounds()

    for k in range(N_KNOTS):
        t_prev, t = (k-1) * DELTA_T, k * DELTA_T

        # Pointwise constraints (dynamics, kinematics, limits):
        # =========================================
        constraints.append(
            # Forward dynamics accelerations:
            Constraint(a_k[k] - fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))
        )

        # Robot torso cannot go below the ground:
        bounds.add(Bound(q_k[k][2], lb = FLOOR_Z + 0.08, ub = ca.inf))

        # # Joint torque limits in N*m:
        # bounds.add_bound(Bound(tau_k[k], lb = -2, ub = 2))

        # Forward foothold kinematics:
        constraints.append(Constraint(f_pos_k[k] - fk(q_k[k])))
        
        # Integration constraints, using implicit Euler:
        # ==============================================
        if k > 0:
            constraints.append(Constraint(v_k[k] - v_k[k-1] - DELTA_T * a_k[k]))
            constraints.append(Constraint(q_k[k] - integrate_state(q_k[k-1], DELTA_T * v_k[k])))

        # Feet contact constraints:
        # TODO: Clean up...
        # =========================
        for foot_idx in range(len(FEET)):
            contact_ivt = TASK.contact_periods[foot_idx]

            if contact_ivt.overlaps(t):
                # TODO: Add limit, normalise by Δt?
                bounds.add(Bound(λ_k[k][foot_idx, 2], lb = 0.0, ub = ca.inf))

                # Without friction cone constraints:
                """
                    496    1.380891e+00   2.859e-06   6.342e-03   3.203e-03        1
                    
                    Final Statistics
                    ----------------
                    Final objective value               =   1.38089102961507e+00
                    Final feasibility error (abs / rel) =   2.86e-06 / 1.78e-09
                    Final optimality error  (abs / rel) =   6.34e-03 / 6.34e-03
                    # of iterations                     =        496
                    # of CG iterations                  =        106
                    # of function evaluations           =        896
                    # of gradient evaluations           =        498
                    # of Hessian evaluations            =        496
                    Total program time (secs)           =      21.82947 (    21.826 CPU time)
                    Time spent in evaluations (secs)    =      15.51438

                    ===============================================================================

                            S  :   t_proc      (avg)   t_wall      (avg)    n_eval
                        nlp_fg  | 658.96ms (734.63us) 655.84ms (731.15us)       897
                    nlp_gf_jg  |   4.16 s (  8.36ms)   4.17 s (  8.37ms)       498
                    nlp_hess_l  |  10.58 s ( 21.34ms)  10.59 s ( 21.34ms)       496
                        total  |  21.83 s ( 21.83 s)  21.84 s ( 21.84 s)         1
                """

                # With friction cone constraints:
                """
                    2673    1.958227e-01   2.195e-04   3.838e-03   1.993e-02        0

                    EXIT: Primal feasible solution; terminate because the relative change in
                        the objective function < 1.000000e-04 for 5 consecutive feasible iterations.
                        Decrease ftol or increase ftol_iters to try for more accuracy.

                    Final Statistics
                    ----------------
                    Final objective value               =   1.95822681409181e-01
                    Final feasibility error (abs / rel) =   2.20e-04 / 1.37e-07
                    Final optimality error  (abs / rel) =   3.84e-03 / 3.84e-03
                    # of iterations                     =       2673
                    # of CG iterations                  =      10966
                    # of function evaluations           =       9019
                    # of gradient evaluations           =       2675
                    # of Hessian evaluations            =       2673
                    Total program time (secs)           =     139.36000 (   139.344 CPU time)
                    Time spent in evaluations (secs)    =      84.96542

                    ===============================================================================

                            S  :   t_proc      (avg)   t_wall      (avg)    n_eval
                        nlp_fg  |   6.64 s (735.61us)   6.62 s (734.41us)      9020
                    nlp_gf_jg  |  22.34 s (  8.35ms)  22.36 s (  8.36ms)      2675
                    nlp_hess_l  |  55.38 s ( 20.72ms)  55.40 s ( 20.72ms)      2673
                        total  | 139.35 s (139.35 s) 139.37 s (139.37 s)         1
                """

                # TODO: Add these, but introduce slack variables for the absolute values:
                
                # # Friction cone constraints (pyramidal approximation):
                # constraints.append(
                #     Constraint(
                #         # abs(fx) <= fz * μ:
                #         MU * λ_k[k][foot_idx, 2] - ca.fabs(λ_k[k][foot_idx, 0]),
                #         lb = 0.0, ub = ca.inf
                #     )
                # )

                # constraints.append(
                #     Constraint(
                #         # abs(fy) <= fz * μ
                #         MU * λ_k[k][foot_idx, 2] - ca.fabs(λ_k[k][foot_idx, 1]),
                #         lb = 0.0, ub = ca.inf
                #     )
                # )

                # Foothold constraints:
                bounds.add(Bound(f_pos_k[k][foot_idx, 2], lb = FLOOR_Z, ub = FLOOR_Z))

                if k - 1 >= 0 and contact_ivt.overlaps(t_prev):
                    constraints.append(
                        Constraint(f_pos_k[k][foot_idx, :2] - f_pos_k[k-1][foot_idx, :2])
                    )

            else:
                bounds.add(Bound(λ_k[k][foot_idx, :], lb = 0, ub = 0))

                # If not in contact, foot X and Y are free, and Z >= floor:
                bounds.add(Bound(f_pos_k[k][foot_idx, 2], lb = FLOOR_Z, ub = ca.inf))

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

    kin_constr = TASK.get_kinematic_constraints(
        q_k, v_k, a_k, f_pos_k, {"FLOOR_Z": FLOOR_Z}
    )

    for constr in kin_constr:
        if isinstance(constr, Bound):
            bounds.add(constr)
        elif isinstance(constr, Constraint):
            constraints.append(constr)
        else:
            assert False
    #############################################################

    #endregion

    #region Problem and solver setup
    print("Creating NLP description...")

    decision_vars = ca.vertcat(
        *q_k, *v_k, *a_k, *tau_k, flatten(λ_k), flatten(f_pos_k)
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
    
    # TODO: PRESOLVE SETTINGS!
    # https://or.stackexchange.com/questions/3128/can-tuning-knitro-solver-considerably-make-a-difference

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
        bounds.get_bound(decision_vars[v_idx].name())
        for v_idx in range(decision_vars.shape[0])
    ]

    soln = solver(
        # x0  = const_pose_guess(N_KNOTS, fk, Pose.STANDING_V).flatten(),

        x0  = prev_soln_guess(
            int(20 * TASK.duration), robot,
            "jump_20hz_1000ms.bin",
            interp_knots = N_KNOTS
        ).flatten(),

        lbg = flatten([c.lb for c in constraints]),
        ubg = flatten([c.ub for c in constraints]),
        lbx = [b[0] for b in x_bounds],
        ubx = [b[1] for b in x_bounds],
    )

    with open(OUTPUT_FILENAME, "wb") as wf:
        pickle.dump(soln, wf)
    