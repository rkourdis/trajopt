import pickle
import argparse

from tasks import *
from guesses import *
from solve import solve
from robot import Solo12
from problem import Problem
from export import export_hdf5
from transcription import Subproblem
from utilities import switch_mrp_in_q
from continuity import ContinuityInfo
from visualization import visualise_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize_file',     type = str,                help = "Filename of solution to visualize")
    parser.add_argument('--freq',               type = int, default = 20,  help = "Transcription frequency")
    parser.add_argument('--prev_solution_file', type = str,                help = "Previous solution file to use as initial guess")
    parser.add_argument('--interp_factor',      type = int, default = 1,   help = "Interpolation factor for previous solution guess")
    parser.add_argument("--hdf5_file",          type = str,                help = "Export trajectory to .hdf5")
    opts = parser.parse_args()

    solo = Solo12(visualize = bool(opts.visualize_file))

    if opts.visualize_file:
        visualise_solution(opts.visualize_file, solo)
        exit()

    if opts.prev_solution_file:
        # Load previous solution from file if we should use it as
        # the initial guess:
        with open(opts.prev_solution_file, "rb") as rf:
            prev_soln = pickle.load(rf)

        create_guess = lambda subp: \
            PrevTrajGuess(
                Problem.load_subtrajectory(prev_soln, subp),
                opts.interp_factor
            )
        
    else:
        # Otherwise, use the robot standing pose as the guess:
        create_guess = lambda _: StandingGuess(solo)

    problem = Problem(
        subproblems = [
            Subproblem("jump_up", JumpTaskInPlace, Fraction(opts.freq), solo, create_guess("jump_up"))
            # Subproblem("jump_fwd", JumpTaskFwd, Fraction(opts.freq), solo, create_guess("jump_fwd")),
            # Subproblem("jump_bwd", JumpTaskBwd, Fraction(opts.freq), solo, create_guess("jump_bwd")),
        ],

        continuity_info = [
            # ContinuityInfo()
        ]
    )

    solution = solve(problem)
    output_filename = f"solution_{opts.freq}hz.bin"

    with open(output_filename, "wb") as wf:
        pickle.dump(solution, wf)

    print(f"Saved solution to: {output_filename}")

    if opts.hdf5_file:
        trajectory = Problem.stitch_trajectories(
            Problem.load_trajectories(solution)
        )

        export_hdf5(trajectory, opts.hdf5_file)
        print(f"Trajectory exported to: {opts.hdf5_file}")
