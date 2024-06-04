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
    TASK = BACKFLIP_LAND_TASK

    parser = argparse.ArgumentParser()
    parser.add_argument('--visualise', action='store_true')
    options = parser.parse_args()

    # The order of forces per foot WILL be as in this list:
    FEET = ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]

    MU = 0.7
    FREQ_HZ = 160
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
        f_pos_k.append(ca.SX.sym(f"f_pos_{k}", len(FEET), 3))    # 4  x 3
        
    #endregion

    #region Constraints and bounds
    print("Creating constraints and bounds...")
    constraints = []
    bounds = VariableBounds()

    for k in range(N_KNOTS):
        t = k * DELTA_T

        # Pointwise constraints (dynamics, kinematics, limits):
        # =========================================
        constraints.append(
            # Forward dynamics accelerations:
            Constraint(a_k[k] - fd(q_k[k], v_k[k], tau_k[k], λ_k[k]))
        )

        # Robot torso cannot go below the ground:
        bounds.add_expr(q_k[k][2], lb = FLOOR_Z + 0.08, ub = ca.inf)

        # Joint torque limits in N*m:
        bounds.add_expr(tau_k[k], lb = -2, ub = 2)

        constraints.append(
            # Forward foothold kinematics:
            Constraint(f_pos_k[k] - fk(q_k[k]))
        )
        
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
            # If the foot is on the ground on the next knot, contact forces should be now available.
            # This is so that the velocity of the next knot is such that the foot can be kept
            # in contact.
            # The forces create an impulse during knot k and k+1 to cancel out foot velocity.
            # If this is the last knot and we're in contact, we'll assume contact continues.
            t_prev, t_next = (k - 1) * DELTA_T, (k + 1) * DELTA_T

            if (contact_ivt.overlaps(t) and k == N_KNOTS - 1) or contact_ivt.overlaps(t_next):
                # Z contact force must be pointing up (repulsive).
                # It must not exceed 30N for each foot.
                bounds.add_expr(λ_k[k][foot_idx, 2], lb = 0.0, ub = 30.0)

                # Friction cone constraints (pyramidal approximation):
                constraints.append(
                    # abs(fx) <= fz * μ
                    Constraint(MU * λ_k[k][foot_idx, 2] - ca.fabs(λ_k[k][foot_idx, 0]), lb = 0.0, ub = ca.inf)
                )

                constraints.append(
                    # abs(fy) <= fz * μ
                    Constraint(MU * λ_k[k][foot_idx, 2] - ca.fabs(λ_k[k][foot_idx, 1]), lb = 0.0, ub = ca.inf)
                )

            else:
                # No contact forces available:
                bounds.add_expr(λ_k[k][foot_idx, :], lb = 0, ub = 0)

            # Foot Positioning
            # ----------------
            if contact_ivt.overlaps(t):
                # If in contact, the foot Z must be on the ground, and the foot mustn't slip:
                bounds.add_expr(f_pos_k[k][foot_idx, 2], lb = FLOOR_Z, ub = FLOOR_Z)

                # We enforce the non-slip constraint by setting the foot position
                # to be equal to the previous one, if there was contact.
                # Otherwise, the footholds are free to be chosen by the optimizer.
                if k > 0 and contact_ivt.overlaps(t_prev):
                    constraints.append(
                        Constraint(f_pos_k[k][foot_idx, :] - f_pos_k[k-1][foot_idx, :])
                    )

            else:
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
    
    knitro_settings = {
        "hessopt":      knitro.KN_HESSOPT_LBFGS,
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
        # x0  = const_pose_guess(N_KNOTS, fk, Pose.STANDING_V).flatten(),
        x0  = prev_soln_guess(160, robot, "trajectories/backflip_land_160hz_1000ms.bin").flatten(),

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
    