import pickle
import argparse

import knitro
import casadi as ca
from pinocchio import casadi as cpin

from poses import Pose
from robot import load_solo12
from guesses import *
from visualisation import visualise_solution
from utilities import integrate_state, flatten, switch_mrp_in_q, ca_to_np

from transcription import Constraint, VariableBounds
from dynamics import ADForwardDynamics
from kinematics import ADFootholdKinematics

from tasks import *

if __name__ == "__main__":
    TASK = BACKFLIP_LAND_TASK

    parser = argparse.ArgumentParser()
    parser.add_argument('--visualise', action='store_true')
    options = parser.parse_args()

    # The order of forces per foot WILL be as in this list:
    FEET = ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]

    MU = 1.0
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

    # launch = prev_soln_guess(40, robot, "backflip_launch_40hz_1000ms.bin")

    # q0, v0, a0, tau0 = launch.q_k[-1], launch.v_k[-1], launch.a_k[-1], launch.tau_k[-1]
    # q, v = q0, v0
    # qs = []

    # for _ in range(1000):
    #     dt = 0.01
    #     a, _ = fd(ca.SX(q), ca.SX(v), ca.SX(tau0), active_conctacts = [False] * 4)
    #     v += a * dt
    #     q = integrate_state(q, v * dt)

    #     qs.append(q)

    # from utilities import q_mrp_to_quat, ca_to_np
    # for q in qs:
    #     robot.display(ca_to_np(q_mrp_to_quat(q)))
    #     print(q)
    #     input()

    # exit()

    if options.visualise:
        visualise_solution(OUTPUT_FILENAME, N_KNOTS, DELTA_T, robot)
        exit()

    constraints = []
    bounds = VariableBounds()
    q_k, v_k, a_k, tau_k, f_pos_k, lambda_k = [], [], [], [], [], []

    for k in range(N_KNOTS):
        t = k * DELTA_T
        
        q_k.append(ca.SX.sym(f"q_{k}", robot.nq - 1))            # Orientations use MRP
        v_k.append(ca.SX.sym(f"v_{k}", robot.nv))                # 18 x 1
        a_k.append(ca.SX.sym(f"a_{k}", robot.nv))                # 18 x 1
        tau_k.append(ca.SX.sym(f"τ_{k}", len(actuated_joints)))  # 12 x 1 
        f_pos_k.append(ca.SX.sym(f"f_pos_{k}", len(FEET), 3))    # 4 x 3 
        lambda_k.append(ca.SX.sym(f"lambda_{k}", len(FEET), 3))   # 4 x 3 

        # Pointwise constraints (dynamics, kinematics, limits):
        # =========================================
        accel, forces = fd(
            q_k[k], v_k[k], tau_k[k],
            [TASK.contact_periods[f_id].overlaps(t) for f_id in range(len(FEET))]
        )

        # Forward dynamics accelerations, constrained if feet in contact:
        constraints.append(Constraint(a_k[k] - accel))

        # Robot torso cannot go below the ground:
        bounds.add_expr(q_k[k][2], lb = FLOOR_Z + 0.08, ub = ca.inf)

        # Joint torque limits in N*m:
        bounds.add_expr(tau_k[k], lb = -1.8, ub = 1.8)
        # bounds.add_expr(tau_k[k], lb = -2, ub = 2)

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

        # Feet constraints:
        # =============================================

        # Contact forces:
        constraints.append(Constraint(lambda_k[k] - forces))

        # Forward kinematics:
        constraints.append(Constraint(f_pos_k[k] - fk(q_k[k])))

        # Z contact force must be pointing up (repulsive).
        # TODO: Add limit, normalise by Δt?
        bounds.add_expr(lambda_k[k][:, 2], lb = 0.0, ub = ca.inf)

        # Friction cone constraints (pyramidal approximation):
        # abs(fx) <= fz * μ:
        constraints.append(
            Constraint(MU * lambda_k[k][:, 2] - ca.fabs(lambda_k[k][:, 0]), lb = 0.0, ub = ca.inf)
        )

        # abs(fy) <= fz * μ
        constraints.append(
            Constraint(MU * lambda_k[k][:, 2] - ca.fabs(lambda_k[k][:, 1]), lb = 0.0, ub = ca.inf)
        )

        # Feet cannot go below the ground:
        for f_idx in range(len(FEET)):
            bounds.add_expr(f_pos_k[k][f_idx, 2], lb = FLOOR_Z, ub = ca.inf)

    for idx, foot in enumerate(FEET):
        for interval in list(TASK.contact_periods[idx]):
            c_knot = math.ceil(interval[0] * FREQ_HZ - ε)
            
            # Add constraint for the feet height at the knot
            # where contact starts. The feet should stay in place
            # for the knots that follow, as we're enforcing the constraint
            # via the dynamics.
            # NOTE: The constraint will slightly drift still, around the
            # knot when contact happens. I think it's because
            # the constrained accelerations take a bit to be integrated.
            # When increasing N_KNOTS the drift becomes smaller, but it still
            # will cause trouble for the optimizer if a constraint forces it
            # to be zero.
            bounds.add_expr(f_pos_k[c_knot][idx, 2], lb = FLOOR_Z, ub = FLOOR_Z)

    #region Objective
    print("Creating optimization objective...")

    # Integrate trajectory error to minimize, normalised by the trajectory duration:
    objective = sum(
        DELTA_T / TASK.duration * 
            TASK.traj_error(k * DELTA_T, q_k[k], v_k[k], a_k[k], tau_k[k], lambda_k[k])

        for k in range(N_KNOTS)
    )

    #endregion

    #region Kinematic constraints
    print("Adding trajectory kinematic constraints...")
    constraints += TASK.get_kinematic_constraints(
        q_k, v_k, a_k,
        {"FLOOR_Z": FLOOR_Z, "FREQ_HZ": FREQ_HZ, "N_KNOTS": N_KNOTS, "DELTA_T": DELTA_T}
    )


    # Add continuity constraints, switch MRP:
    if TASK == BACKFLIP_LAND_TASK:
        launch = prev_soln_guess(160, robot, "backflip_launch_160hz_1000ms.bin")

        q_start_switched = switch_mrp_in_q(launch.q_k[-1])
        constraints.append(Constraint(q_k[0] - q_start_switched))
        constraints.append(Constraint(v_k[0] - launch.v_k[-1]))
        # constraints.append(Constraint(a_k[0] - launch.a_k[-1]))
        constraints.append(Constraint(tau_k[0] - launch.tau_k[-1]))
        # constraints.append(Constraint(lambda_k[0] - launch.lambda_k[-1]))

        land_guess = Trajectory(
            N_KNOTS,
            q_k = [q_start_switched] * N_KNOTS,
            v_k = [launch.v_k[-1]] * N_KNOTS,
            a_k = [launch.a_k[-1]] * N_KNOTS,
            tau_k = [launch.tau_k[-1]] * N_KNOTS,
            f_pos_k = [ca_to_np(fk(q_start_switched))] * N_KNOTS,
            lambda_k = [launch.lambda_k[-1]] * N_KNOTS,
        )
    #############################################################

    #endregion

    #region Problem and solver setup
    print("Creating NLP description...")

    decision_vars = ca.vertcat(
        *q_k, *v_k, *a_k, *tau_k, flatten(f_pos_k), flatten(lambda_k)
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
        # "outlev":       6,
        # "outmode":      knitro.KN_OUTMODE_BOTH
        "bar_feasible": knitro.KN_BAR_FEASIBLE_GET,
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
        # x0 = land_guess.flatten(),
        # x0  = const_pose_guess(N_KNOTS, Pose.STANDING_V, fk).flatten(),
        x0  = prev_soln_guess(
           int(40 * TASK.duration), robot, "backflip_land_40hz_1000ms.bin",
           interp_knots = N_KNOTS
        ).flatten(),

        lbg = flatten([c.lb for c in constraints]),
        ubg = flatten([c.ub for c in constraints]),
        lbx = [b[0] for b in x_bounds],
        ubx = [b[1] for b in x_bounds],
    )

    with open(OUTPUT_FILENAME, "wb") as wf:
        pickle.dump(soln, wf)
    