import pickle
import argparse

import knitro
import casadi as ca

from tasks import *
from robot import Solo12
from problem import Problem
from guesses import StandingGuess
from transcription import Subproblem
from utilities import switch_mrp_in_q
from continuity import ContinuityInfo
from visualization import visualise_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true')
    options = parser.parse_args()

    GLOBAL_FREQ_HZ = Fraction("20")
    OUTPUT_FILENAME = f"solution_{GLOBAL_FREQ_HZ}hz.bin"

    solo = Solo12(visualize = options.visualize)

    # problem = Problem(
    #     subproblems = [
    #         Subproblem("launch", BackflipLaunch, GLOBAL_FREQ_HZ, solo, StandingGuess(robot = solo)),
    #         Subproblem("land",   BackflipLand,   GLOBAL_FREQ_HZ, solo, StandingGuess(robot = solo)),
    #     ],

    #     continuity_info = [
    #         ContinuityInfo(q = lambda x: switch_mrp_in_q(x))
    #     ]
    # )

    problem = Problem(
        subproblems = [
            Subproblem("jump", JumpTaskInPlace, GLOBAL_FREQ_HZ, solo, StandingGuess(robot = solo))
        ]
    )

    problem.transcribe()

    if options.visualize:
        visualise_solution(OUTPUT_FILENAME, problem, solo)
        exit()

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

    print("Calling solver...")

    soln = solver(
        x0 = problem.guess(),
        
        # Variable bounds:
        lbx = v_lb, ubx = v_ub,

        # Constraint bounds:
        lbg = c_lb, ubg = c_ub,
    )

    with open(OUTPUT_FILENAME, "wb") as wf:
        pickle.dump(soln, wf)
    