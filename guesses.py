import numpy as np
import casadi as ca
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tasks import Task
from robot import Solo12
import utilities as utils
from poses import load_robot_pose, Pose
from kinematics import ADFootholdKinematics
from variables import KnotVars, CollocationVars

@dataclass
class GuessOracle(ABC):
    robot: Solo12
    task: Task      = None

    @abstractmethod
    def guess(self, k: int, *args, **kwargs) -> KnotVars:
        # NOTE: We won't provide guesses for the slack variables yet.
        pass

@dataclass
# Creates a constantly standing pose:
class StandingGuess(GuessOracle):
    def __post_init__(self):
        # Load pose from pickled closed form solution:
        self.q, self.v, self.τ, self.λ = load_robot_pose(Pose.STANDING_V)

        # Create numerical instance of FK to calculate feet positions:
        q_sym = ca.SX.sym("q_sym", self.q.shape)
        ad_fk = ADFootholdKinematics(self.robot)
        num_fk = ca.Function("num_fk", [q_sym], [ad_fk(q_sym)])

        self.f_pos = utils.ca_to_np(num_fk(self.q))

    # Provide standing guess. If `switch_mrp`, it switches the floating
    # base MRP to the shadow one, to avoid the 2π singularity:
    def guess(self, _: int) -> KnotVars:
        return KnotVars(
            np.copy(self.q), np.copy(self.v), np.zeros(self.v.shape),
            np.copy(self.τ), np.copy(self.λ), np.copy(self.f_pos)
        )

# Returns a previous trajectory for a subproblem as an initial guess.
# If `interp_factor` is set, the trajectory will be interpolated by simple
# repetition, as if the transcription frequency was the original multiplied
# by `interp_factor`:
class PrevTrajGuess(GuessOracle):
    def __init__(self, base_traj: CollocationVars[np.ndarray], interp_factor: int = 1):
        assert interp_factor >= 1, f"Interpolation factor must be >= 1"

        # Keep trajectory as is if there's no interpolation:
        if interp_factor == 1:
            self.traj = base_traj
            return

        # If interpolation is needed, calculate the new number of knots:
        interp_n_knots = (base_traj.n_knots - 1) * interp_factor + 1
        interp_traj = CollocationVars[np.ndarray](interp_n_knots)

        # Repeat each knot `interp_factor` times.
        # The last knot is an exception - it will not be repeated as it
        # corresponds to the end time:
        for base_k in range(base_traj.n_knots - 1):
            kv = base_traj.get_vars_at_knot(base_k)
            
            # NOTE: The knot duration is unused:
            dt = base_traj.knot_duration[base_k]

            for _ in range(interp_factor):
                interp_traj.append_knot(kv, dt / interp_factor)
        
        # Handle the final knot:
        interp_traj.append_knot(
            base_traj.get_vars_at_knot(-1),
            base_traj.knot_duration[-1] / interp_factor
        )

        self.traj = interp_traj

    def guess(self, k: int) -> KnotVars:
        return self.traj.get_vars_at_knot(k)