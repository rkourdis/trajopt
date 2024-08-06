import pickle
import argparse
from pathlib import Path

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

    # Load previous solution from file if it's used as an initial guess.
    # Interpolate, if needed.
    if opts.prev_solution_file:
        with open(opts.prev_solution_file, "rb") as rf:
            prev_soln = pickle.load(rf)

        create_guess = lambda subp: \
            PrevTrajGuess(
                Problem.load_subtrajectory(prev_soln, subp),
                opts.interp_factor
            )
        
    # Otherwise, use the robot standing pose as the guess:
    else:
        create_guess = lambda _: StandingGuess(solo)

    # Problem definition:
    problem = Problem(
        subproblems = [
            # We split the backflip in two subproblems ('launch', 'land'), as the MRP orientation
            # representation has a singularity at full rotation. Quaternions are difficult to
            # optimize due to the unit norm constraint.
            Subproblem("flip_launch", BackflipLaunch, Fraction(opts.freq), solo, create_guess("flip_launch")),
            Subproblem("flip_land", BackflipLand, Fraction(opts.freq), solo, create_guess("flip_land")),
        ],

        continuity_info = [
            # We avoid the MRP singularity by forcing the solutions of 'launch' and 'land' to be
            # continuous in all variables except the MRP. The MRP at the end of the launch
            # will be switched to its shadow for the landing, allowing it to reach (0, 0, 0)
            # again.
            ContinuityInfo(q = lambda x: switch_mrp_in_q(x))
        ]
    )

    solution = solve(problem)
    output_filename = f"solution_{opts.freq}hz.bin"

    with open(output_filename, "wb") as wf:
        pickle.dump(solution, wf)

    print(f"Saved solution to: {output_filename}")

    # Save .hdf5 trajectory for hardware execution:
    if opts.hdf5_file:
        subtrajectories = Problem.load_trajectories(solution)

        # Stitch all subproblem solutions:
        stitched = Problem.stitch_trajectories(subtrajectories)

        export_hdf5(stitched, opts.hdf5_file)
        print(f"Trajectory exported to: {opts.hdf5_file}")

        for subp, subtraj in zip(problem.subproblems, subtrajectories):
            base_path = Path(opts.hdf5_file)
            subtraj_path = f"{base_path.stem}_{subp.name}{base_path.suffix}"

            export_hdf5(subtraj, subtraj_path)
            print(f"Exported subproblem '{subp.name}' trajectory to: {subtraj_path}")