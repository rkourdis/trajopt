import knitro
import casadi as ca

from utilities import ca_to_np
from problem import Problem, Solution

# Transcribe and solve a problem using the Knitro
# NLP solver. CasADi should calculate gradients and
# Hessians in closed form:
def solve(problem: Problem) -> Solution:
    problem.transcribe()
    
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
    
    print("Instantiating solver...")

    vars, v_lb, v_ub    = problem.variables
    c_exprs, c_lb, c_ub = problem.constraints

    solver = ca.nlpsol(
        "S",
        "knitro",
        { "x": vars, "f": problem.objective, "g": c_exprs },
        {
            # "verbose": True,
            "knitro": knitro_settings,
            "complem_variables": problem.complementarities
        }
    )

    print("Starting solver...")

    soln = solver(
        x0 = problem.guess(),
        
        # Variable bounds:
        lbx = v_lb, ubx = v_ub,

        # Constraint bounds:
        lbg = c_lb, ubg = c_ub,
    )

    # Convert all CasADi matrices to NumPy before storing:
    np_soln = {k: ca_to_np(v) for k, v in soln.items()}

    return Solution(
        solver_output = np_soln,
        transcription_infos = [
            subp.transcription_info
            for subp in problem.subproblems
        ]
    )