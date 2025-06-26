import numpy as np
import casadi as ca
import pinocchio as pin
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tasks import Task
from robot import LeggedRobot
import utilities as utils
from poses import load_robot_pose, Pose
from kinematics import ADFrameKinematics
from variables import KnotVars, CollocationVars

@dataclass
class GuessOracle(ABC):
    robot: LeggedRobot
    task: Task      = None

    @abstractmethod
    def guess(self, k: int, *args, **kwargs) -> KnotVars:
        # NOTE: We won't provide guesses for the slack variables yet.
        pass

@dataclass
# Returns a static standing pose for the Solo12:
class SoloStandingGuess(GuessOracle):
    def __post_init__(self):
        # Load pose from pickled closed-form (accel = 0) solution:
        self.q, self.v, self.τ, self.λ = load_robot_pose(Pose.SOLO_STANDING_V)

        # Do FK to calculate feet positions:
        q_sym = ca.SX.sym("q_sym", self.q.shape)
        ad_fk = ADFrameKinematics(self.robot)
        num_fk = ca.Function("num_fk", [q_sym], [ad_fk(q_sym)["feet"]])

        self.f_pos = utils.ca_to_np(num_fk(self.q))

    def guess(self, _: int) -> KnotVars:
        return KnotVars(
            np.copy(self.q), np.copy(self.v), np.zeros(self.v.shape),
            np.copy(self.τ), np.copy(self.λ), np.copy(self.f_pos)
        )

@dataclass
class EmptyGuess(GuessOracle):
    def __post_init__(self):
        self.q = np.zeros((self.robot.cmodel.nq - 1, 1))
        self.v = np.zeros((self.robot.cmodel.nv, 1))
        self.a = np.zeros((self.robot.cmodel.nv, 1))
        self.τ = np.zeros((len(self.robot.actuated_joints), 1))

        # Do FK to calculate feet positions:
        q_sym = ca.SX.sym("q_sym", self.q.shape)
        ad_fk = ADFrameKinematics(self.robot)
        num_fk = ca.Function("num_fk", [q_sym], [ad_fk(q_sym)["feet"]])

        self.f_pos = utils.ca_to_np(num_fk(self.q))

        # Spread weight of robot to all legs:
        pin.crba(
            self.robot.robot.model,
            self.robot.robot.data,
            utils.ca_to_np(utils.q_mrp_to_quat(self.q))
        )

        m = list(self.robot.robot.data.mass)[0]
        feet = len(self.robot.frames["feet"])

        self.λ = np.zeros((feet, 3))
        self.λ[:, 2] = m * 9.81 / feet

    def guess(self, _: int) -> KnotVars:
        return KnotVars(
            np.copy(self.q), np.copy(self.v), np.copy(self.a),
            np.copy(self.τ), np.copy(self.λ), np.copy(self.f_pos)
        )


# Returns a previous subproblem trajectory as an initial guess.
# If `interp_factor` is set, the trajectory will be interpolated
# by simple repetition, as if the transcription frequency was the
# original times `interp_factor`:
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