import numpy as np
import casadi as ca
from fractions import Fraction
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tasks import Task
from robot import Solo12
import utilities as utils
from variables import KnotVars
from poses import load_robot_pose, Pose
from kinematics import ADFootholdKinematics

@dataclass
class GuessOracle(ABC):
    robot: Solo12
    task: Task      = None

    @abstractmethod
    def guess(t: Fraction, *args, **kwargs) -> KnotVars:
        # We won't provide guesses for the slack variables yet.
        pass


@dataclass
# Creates a constantly standing pose:
class StandingGuess(GuessOracle):
    switched_mrp: bool   = False

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
    def guess(self, _: Fraction) -> KnotVars:
        q_ret = np.copy(
            self.q 
            if not self.switched_mrp
            else utils.ca_to_np(utils.switch_mrp_in_q(self.q))
        )

        return KnotVars(
            np.copy(q_ret), np.copy(self.v), np.zeros(self.v.shape),
            np.copy(self.τ), np.copy(self.λ), np.copy(self.f_pos)
        )

@dataclass
# Returns a previous solution as an initial guess trajectory.
# If `interp_factor` > 1, the trajectory will be interpolated
# to the original duration times interp_factor.
class PrevSolnGuess(GuessOracle):
    # solution: CollocationVars[np.ndarray]
    # interp_factor: int = 1

    # TODO: This somehow needs to know the transcription frequency.
    def __post_init__(self):
        raise NotImplementedError
