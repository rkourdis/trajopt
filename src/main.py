import pickle
import argparse
from pathlib import Path
from fractions import Fraction

from solo_tasks import *
from bolt_tasks import *

from guesses import *
from solve import solve
from robot import Solo12, Bolt
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

    # robot = Solo12(visualize = bool(opts.visualize_file))
    robot = Bolt(visualize = bool(opts.visualize_file))
    # robot = Bolt(visualize = True)

    # print(robot)
    # print(list(robot.robot.model.joints))
    # print(list(robot.robot.model.names))
    # print(robot.cmodel.nq)
    # print(robot.actuated_joints)
    
    # import pinocchio as pin
    # q0 = np.zeros((robot.robot.model.nq - 1, 1))

    # q0[robot.q_off("FR_HFE")] = +np.pi/4
    # q0[robot.q_off("FL_HFE")] = +np.pi/4

    # q0[robot.q_off("FR_KFE")] = -np.pi/2
    # q0[robot.q_off("FL_KFE")] = -np.pi/2

    # import utilities as utils
    # from configurations import *
    # q = create_state_vector(robot.robot, BOLT_SITTING_JOINT_MAP)

    # q[4] = -0.03

    # robot.robot.display(utils.ca_to_np(utils.q_mrp_to_quat(q)))

    # while True:
    #     pass

    # print(q0)
    # exit()

    if opts.visualize_file:
        visualise_solution(opts.visualize_file, robot)
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

    else:
        if type(robot) == Bolt:
            create_guess = lambda _: EmptyGuess(robot)

        if type(robot) == Solo12:
            create_guess = lambda _: SoloStandingGuess(robot)

    # Problem definitions:
    active_problem = Problem(
        subproblems = [
            Subproblem("squat", Squat, Fraction(opts.freq), robot, create_guess("squat")),
        ],
    )

    # backflip_problem = Problem(
    #     subproblems = [
    #         # We split the backflip in two subproblems ('launch', 'land'), as the MRP orientation
    #         # representation has a singularity at full rotation. Quaternions are difficult to
    #         # optimize due to the unit norm constraint.
    #         Subproblem("flip_launch", BackflipLaunch, Fraction(opts.freq), solo, create_guess("flip_launch")),
    #         Subproblem("flip_land", BackflipLand, Fraction(opts.freq), solo, create_guess("flip_land")),
    #     ],

    #     continuity_info = [
    #         # We avoid the MRP singularity by forcing the solutions of 'launch' and 'land' to be
    #         # continuous in all variables except the MRP. The MRP at the end of the launch
    #         # will be switched to its shadow for the landing, allowing it to reach (0, 0, 0)
    #         # again.
    #         ContinuityInfo(q = lambda x: switch_mrp_in_q(x))
    #     ]
    # )

    # active_problem = upright_problem

    solution = solve(active_problem)
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

        for subp, subtraj in zip(active_problem.subproblems, subtrajectories):
            base_path = Path(opts.hdf5_file)
            subtraj_path = f"{base_path.stem}_{subp.name}{base_path.suffix}"

            export_hdf5(subtraj, subtraj_path)
            print(f"Exported subproblem '{subp.name}' trajectory to: {subtraj_path}")