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

from constraints import Constraint, VariableBounds
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

    MU = 0.6
    FREQ_HZ = 20
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
    
    # Variables that help reformulate the problem in a way that's easier
    # to solve, but aren't useful after the optimization.
    # Will be appended last to the solution vector.
    slack_vars = []

    # Tuples of names of variables complementary to each other.
    # These variables must be bounded >= 0.
    complementarities = []

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

        # Joint torque limits in N*m:
        bounds.add(Bound(tau_k[k], lb = -1.9, ub = 1.9))

        # Forward foothold kinematics:
        constraints.append(Constraint(f_pos_k[k] - fk(q_k[k])))
        
        # Integration constraints, using implicit Euler:
        # ==============================================
        if k > 0:
            constraints.append(Constraint(v_k[k] - v_k[k-1] - DELTA_T * a_k[k]))
            constraints.append(Constraint(q_k[k] - integrate_state(q_k[k-1], DELTA_T * v_k[k])))

        # Feet contact constraints:
        # =========================
        for foot_idx in range(len(FEET)):
            contact_ivt = TASK.contact_periods[foot_idx]

            if contact_ivt.overlaps(t):
                # TODO: Add limit, normalise by Δt?
                bounds.add(Bound(λ_k[k][foot_idx, 2], lb = 0.0, ub = ca.inf))

                # Friction cone constraints (pyramidal approximation).
                # We introduce slack variables to calculate abs(fx) and abs(fx)
                # as otherwise the absolute value causes derivatives to be discontinuous.
                # This way, we remove the discontinuity in the constraints and instead
                # create new bounded variables with complementarity constraints:
                λxy_pos_k = ca.SX.sym(f"λxy_+_{foot_idx}_{k}", 2)
                λxy_neg_k = ca.SX.sym(f"λxy_-_{foot_idx}_{k}", 2)
                
                # λ_xy_pos >= 0, λ_xy_neg >= 0:
                bounds.add(Bound(λxy_pos_k, lb = 0.0, ub = ca.inf))
                bounds.add(Bound(λxy_neg_k, lb = 0.0, ub = ca.inf))

                # λ_xy_pos - λ_xy_neg = λ_xy:
                constraints.append(
                    # Substitute: x -> x_pos - x_neg with 0 =< x_pos \perp x_neg >= 0
                    # Then:      |x| = x_pos + x_neg
                    Constraint((λxy_pos_k - λxy_neg_k) - λ_k[k][foot_idx, :2].T)
                )
                
                # abs(λ_x,y) <= λz * μ:
                # In this case, the complementarity constraint λ_xy_pos \perp λ_xy_neg isn't strictly
                # required, as: if x = x_pos - x_neg then |x| = |x_pos - x_neg| <= x_pos + x_neg always.
                # That is, x_pos + x_neg will be always >= |x| and if x_pos + x_neg is below the
                # friction limit, then |x| must be as well.
                # If the solver needs more tangential force, it should figure out that the maximum
                # it can get would be by setting one of the variables to zero.
                # However, adding the constraint seems to help convergence speed in practice.
                constraints.append(
                    Constraint(
                        ca.repmat(MU * λ_k[k][foot_idx, 2], 2) - (λxy_pos_k + λxy_neg_k),
                        lb = 0.0, ub = ca.inf
                    )
                )

                for idx in range(2):
                    complementarities.append((λxy_pos_k[idx].name(), λxy_neg_k[idx].name()))
                
                slack_vars.append(λxy_pos_k)
                slack_vars.append(λxy_neg_k)

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
        if isinstance(constr, Bound):           bounds.add(constr)
        elif isinstance(constr, Constraint):    constraints.append(constr)
        else:                                   assert False
    #############################################################

    #endregion

    #region Problem and solver setup
    print("Creating NLP description...")

    decision_vars = ca.vertcat(
        *q_k, *v_k, *a_k, *tau_k,
        flatten(λ_k), flatten(f_pos_k), # ...

        # Appended last:
        flatten(slack_vars)
    )

    # Keep a dictionary of variables names to indices so that we convert the
    # complementarity constraints to index tuples for Knitro:
    var_indices = {
        decision_vars[v_idx].name(): v_idx
        for v_idx in range(decision_vars.shape[0])
    } 

    problem = {
        # Decision variables:
        "x": decision_vars,

        # Optimisation objective:
        "f": objective,

        # Constrained expressions:
        "g": flatten([c.expr for c in constraints])
    }

    print("Instantiating solver...")
    
    # NOTE: https://or.stackexchange.com/questions/3128/can-tuning-knitro-solver-considerably-make-a-difference
    knitro_settings = {
        # "hessopt":          knitro.KN_HESSOPT_LBFGS,
        "algorithm":        knitro.KN_ALG_BAR_DIRECT,
        "bar_murule":       knitro.KN_BAR_MURULE_ADAPTIVE,
        "linsolver":        knitro.KN_LINSOLVER_MA57,
        "feastol":          1e-3,
        "ftol":             1e-4,
        "presolve_level":   knitro.KN_PRESOLVE_ADVANCED,
        # "bar_feasible": knitro.KN_BAR_FEASIBLE_GET_STAY,
        # "ms_enable":    True,
        # "ms_numthreads": 8,
    }

    solver = ca.nlpsol(
        "S", "knitro", problem,
        {
            # "verbose": True,
            "knitro": knitro_settings,
            "complem_variables": [
                (var_indices[v1], var_indices[v2])
                for v1, v2 in complementarities
            ],
        }
    )

    #endregion

    #region Solution
    print("Calling solver...")

    x_bounds = [
        bounds.get_bound(decision_vars[v_idx].name())
        for v_idx in range(decision_vars.shape[0])
    ]

    soln = solver(
        x0  = np.append(
            const_pose_guess(N_KNOTS, fk, Pose.STANDING_V).flatten(),
            np.zeros(flatten(slack_vars).shape)
        ),

        # x0  = np.append(
        #     prev_soln_guess(int(40 * TASK.duration), robot, "jump_40hz_1000ms.bin", interp_knots = N_KNOTS).flatten(),
        #     np.zeros(flatten(slack_vars).shape)
        # ),

        lbg = flatten([c.lb for c in constraints]),
        ubg = flatten([c.ub for c in constraints]),
        lbx = [b[0] for b in x_bounds],
        ubx = [b[1] for b in x_bounds],
    )

    with open(OUTPUT_FILENAME, "wb") as wf:
        pickle.dump(soln, wf)
    