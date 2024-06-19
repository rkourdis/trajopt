import pickle
import argparse

from tasks import *
from solve import solve
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

    OUTPUT_FILENAME = f"solution.bin"
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
            Subproblem("jump_1", JumpTaskFwd, Fraction("20"), solo, StandingGuess(robot = solo)),
            Subproblem("jump_2", JumpTaskBwd, Fraction("40"), solo, StandingGuess(robot = solo)),
        ],

        continuity_info = [ContinuityInfo()]
    )

    # problem = Problem(
    #     subproblems = [
    #         Subproblem("stand", GetUpTask, GLOBAL_FREQ_HZ, solo, StandingGuess(robot = solo))
    #     ]
    # )

    if options.visualize:
        visualise_solution(OUTPUT_FILENAME, solo)
        exit()

    solution = solve(problem)

    with open(OUTPUT_FILENAME, "wb") as wf:
        pickle.dump(solution, wf)
    