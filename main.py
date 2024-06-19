import pickle
import argparse

from tasks import *
from guesses import *
from solve import solve
from robot import Solo12
from problem import Problem
from transcription import Subproblem
from utilities import switch_mrp_in_q
from continuity import ContinuityInfo
from visualization import visualise_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true')
    options = parser.parse_args()

    solo = Solo12(visualize = options.visualize)

    BASE_FREQ = Fraction("20")  # Hz
    OUTPUT_FILENAME = f"solution.bin"

    if options.visualize:
        visualise_solution(OUTPUT_FILENAME, solo)
        exit()

    # with open("base_solution.bin", "rb") as rf:
    #     prev_soln = pickle.load(rf)

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
            Subproblem(
                "jump_1", JumpTaskFwd, BASE_FREQ, solo, StandingGuess(solo)
                # PrevTrajGuess(Problem.load_subtrajectory(prev_soln, "jump_1"), 1)
            ),

            Subproblem(
                "jump_2", JumpTaskBwd, BASE_FREQ * 2, solo, StandingGuess(solo)
                # PrevTrajGuess(Problem.load_subtrajectory(prev_soln, "jump_2"), 2)
            ),
        ],

        continuity_info = [ContinuityInfo()]
    )

    solution = solve(problem)

    with open(OUTPUT_FILENAME, "wb") as wf:
        pickle.dump(solution, wf)
    