from typing import Optional
from fractions import Fraction
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from tasks import Task
from robot import Go2
from constraints import *
from guesses import GuessOracle
from variables import CollocationVars
from utilities import integrate_state
from dynamics import ADForwardDynamics
from kinematics import ADFrameKinematics

@dataclass(frozen = True)
# Struct that holds information about a transcribed subproblem.
# We'll store this alongside the global solution vector when the
# optimizer completes. This way we can load, visualize and execute a
# solution without needing to transcribe exactly the same problem again. 
class TranscriptionInfo:
    subproblem_name: str
    
    # The number of decision variables per knot should be constant and
    # not depend on the task being solved (excl. slack variables):
    n_knots: int

    # Discretization Δt, for trajectory execution and visualization:
    dt: float

    # Slack variable count (at the end of the variable vector):
    slack_var_count: int

    # Other auxiliary info:
    description: Optional[str] = None

    contact_mode: str = "fixed"

@dataclass
class Subproblem:
    name:           str
    task:           Task
    freq_hz:        Fraction
    robot:          Go2
    guess_oracle:   GuessOracle

    dvars:          CollocationVars[ca.SX]   = field(init = False)
    n_knots:        int                      = field(init = False)
    objective:      ca.SX                    = field(init = False)
    constraints:    list[ConstraintType]     = field(default_factory = list, init = False)

    transcribed:        bool                 = field(default = False, init = False)
    transcription_info: TranscriptionInfo    = field(default = None, init = False)

    def __post_init__(self):
        assert isinstance(self.freq_hz, Fraction)
        self.dt = 1 / self.freq_hz

        if self.task.duration % self.dt != 0:
            floor_k = self.task.duration // self.dt

            raise ValueError(
                f"Make sure task duration ({self.task.duration}s) is divisible by Δt ({self.dt}s). " + \
                f"Closest are: {floor_k * self.dt}s or {(floor_k + 1) * self.dt}s."
            )

        # We'll have (duration / dt) time periods. Add an extra knot for the final point:        
        self.n_knots = int(self.task.duration * self.freq_hz + 1)
        assert self.n_knots > 0, "Number of knots cannot be zero!"

        self.dvars = CollocationVars[ca.SX](self.n_knots)

    def _create_vars(self) -> None:
        for k in range(self.n_knots):
            self.dvars.q_k.append(ca.SX.sym(f"{self.name}_q_{k}", self.robot.cmodel.nq - 1))         # 18 x 1 (floating base MRP)
            self.dvars.v_k.append(ca.SX.sym(f"{self.name}_v_{k}", self.robot.cmodel.nv))             # 18 x 1
            self.dvars.a_k.append(ca.SX.sym(f"{self.name}_a_{k}", self.robot.cmodel.nv))             # 18 x 1
            self.dvars.τ_k.append(ca.SX.sym(f"{self.name}_τ_{k}", len(self.robot.actuated_joints)))  # 12 x 1 
            self.dvars.λ_k.append(ca.SX.sym(f"{self.name}_λ_{k}", len(self.robot.frames["feet"]), 3))          # 4  x 3
            self.dvars.f_pos_k.append(ca.SX.sym(f"{self.name}_f_pos_{k}", len(self.robot.frames["feet"]), 3))  # 4  x 3

            self.dvars.knot_duration.append(float(self.dt))     # Same Δt for all knots

    # Pointwise variable constraints at each knot (dynamics, kinematics, limits):
    #############################################################################
    def _add_pointwise_constraints(self, k: int) -> None:
        kv = self.dvars.get_vars_at_knot(k)

        self.constraints.extend([
            # Forward dynamics accelerations:
            Constraint(kv.a - self.fd(kv.q, kv.v, kv.τ, kv.λ)),

            # Forward foothold kinematics:
            Constraint(kv.f_pos - self.fk(kv.q)["feet"])
        ])

        # Bound each joint separately: Go2 calves have different torque/speed
        # limits, and front/rear thigh position limits also differ.
        for i in range(len(self.robot.actuated_joints)):
            self.constraints.extend([
                Bound(kv.τ[i], lb = -float(self.robot.τ_max[i]), ub = float(self.robot.τ_max[i])),
                Bound(kv.v[6 + i], lb = -float(self.robot.v_max[i]), ub = float(self.robot.v_max[i])),
                Bound(kv.q[6 + i], lb = float(self.robot.q_min[i]), ub = float(self.robot.q_max[i])),
            ])
    #############################################################################

    # Integration constraints (fully implicit Euler):
    #####################################################
    def _add_integration_constraints(self, k: int) -> None:
        assert k > 0, "Cannot add integration constraint on the first knot!"

        q, v, a         = self.dvars.q_k[k],   self.dvars.v_k[k], self.dvars.a_k[k]
        q_prev, v_prev  = self.dvars.q_k[k-1], self.dvars.v_k[k-1]

        # NOTE: Changing the velocity constraint to the equivalent:
        #           v - (v_prev + dt * a) == 0
        #       seems to make the jump problem more difficult!
        #       Instead of 199 iterations it will take 474, and the
        #       final objective is 1.83 instead of 1.45...
        self.constraints.extend([
            Constraint((v - v_prev) - float(self.dt) * a),
            Constraint(q - integrate_state(q_prev, float(self.dt) * v))
        ])
    #####################################################

    # Foot contact constraints (reaction forces, foothold positioning):
    ###################################################################
    def _add_contact_constraints(self, k: int) -> None:
        t = k * self.dt

        μ       = self.robot.μ
        floor_z = self.robot.floor_z

        def add_constraints_for_foot(foot_idx: int):
            λ, fp     = self.dvars.λ_k[k][foot_idx, :], self.dvars.f_pos_k[k][foot_idx, :]

            dz = ca.SX.sym(f"dz_{foot_idx}_{k}")
            self.dvars.slack_vars.append(dz)
            self.constraints.append(Constraint(dz - (fp[:, 2] - floor_z)))

            # Slip constraints if foot in contact:
            if k > 0:
                fp_prev = self.dvars.f_pos_k[k-1][foot_idx, :]

                # Set the foot XY difference between knots to be zero id dz == 0.
                # -s <= xy_slip <= s and s <= g * dz
                # where s, g >= 0: 
                xy_slip = (fp[:2] - fp_prev[:2]).T

                s, g = ca.SX.sym(f"s_{foot_idx}_{k}"), ca.SX.sym(f"g_{foot_idx}_{k}")
                self.dvars.slack_vars.extend([s, g])

                self.constraints.extend([
                    Bound(s, lb = 0.0, ub = ca.inf),
                    Bound(g, lb = 0.0, ub = ca.inf),

                    # -s <= xy_slip <= s
                    # Use the same s for both dims, it can be scaled arbitrarily large:
                    Constraint(ca.repmat(s, 2) - xy_slip, lb = 0.0, ub = ca.inf),
                    Constraint(xy_slip + ca.repmat(s, 2), lb = 0.0, ub = ca.inf),

                    # s <= g * dz to make s = 0 when dz = 0
                    Constraint(g * dz - s, lb = 0.0, ub = ca.inf),
                ])

            # Square friction cone - limits tangential forces if not in contact:
            self.constraints.extend(
                Bound(λ[2], lb = 0.0, ub = ca.inf),

                # -μ * λ_z <= λ_xy <= μ * λ_z <=> fabs(λ_xy) <= μ * λ_z
                Constraint(μ * λ[2] - λ[0], lb = 0.0, ub = ca.inf),
                Constraint(λ[0] + μ * λ[2], lb = 0.0, ub = ca.inf),

                Constraint(μ * λ[2] - λ[1], lb = 0.0, ub = ca.inf),
                Constraint(λ[1] + μ * λ[2], lb = 0.0, ub = ca.inf),
            )
            
            if self.task.contact_periods is not None:
                # If we know for sure that the foot is above the floor, disallow normal forces:
                # TODO: This is on the current knot - did we do forces instanteneously? But z?
                contact_times = self.task.contact_periods[self.robot.frames["feet"][foot_idx]]

                if not contact_times.overlaps(t):
                    self.constraints.extend([
                        Bound(λ[2], lb = 0.0, ub = 0.0),
                        Bound(dz, lb = 0.0, ub = ca.inf),
                    ])

                    return

            # Distance is complementary to normal force, discover
            # contact scheduling:
            self.constraints.append(Complementarity(dz.name(), λ[2].name()))
            return

        # Add constraints for all feet:
        for foot_idx in range(len(self.robot.frames["feet"])):
            add_constraints_for_foot(foot_idx)
    ###################################################################

    # Transcribe the subproblem. If `is_subsequent`, dynamics and contact constraints
    # will not be generated for the first knot, as continuity constraints between it
    # and the previous subproblem should be applied instead:
    def transcribe(self, is_subsequent: bool) -> None:
        assert not self.transcribed, "Problem has been already transcribed!"

        # Create decision variables for the problem:
        self._create_vars()

        # Instantiate dynamics and kinematics routines:
        self.fd = ADForwardDynamics(self.robot)
        self.fk = ADFrameKinematics(self.robot)

        # Add dynamics constraints for the above decision variables:
        for k in range(self.n_knots):
            if not is_subsequent or k > 0:
                self._add_pointwise_constraints(k)
                self._add_contact_constraints(k)

            if k > 0:
                self._add_integration_constraints(k)

        # Add task-specific trajectory constraints. A constraint might
        # hold for a range of knots:
        for t_period, get_constr in self.task.task_constraints:
            t_s = t_period.start
            t_e = t_period.end if t_period.end != None else self.task.duration

            if t_s < 0 or t_e > self.task.duration:
                raise ValueError(f"Task constraint period ([{t_s}, {t_e}]) is not in [0.0, {self.task.duration}].")
            
            assert t_s % self.dt == 0, f"Constraint start time ({t_s}) must be divisible by Δt ({self.dt})."
            assert t_e % self.dt == 0, f"Constraint end time ({t_e}) must be divisible by Δt ({self.dt})."

            for k in range(int(t_s / self.dt), int(t_e / self.dt) + 1):
                kvars = self.dvars.get_vars_at_knot(k)
                self.constraints.extend(get_constr(kvars, go2 = self.robot, fk = self.fk))

        # Create expression for the scalar problem objective:
        self.objective = ca.SX.zeros(1)
        
        for k in range(self.n_knots):
            err = self.task.traj_error(
                float(k * self.dt), self.dvars.get_vars_at_knot(k)
            )

            # We average the trajectory error over all knots:
            weight = 1.0 / self.n_knots
            self.objective += weight * err
        
        # Save transcription information to be stored in the solution file:
        self.transcription_info = TranscriptionInfo(
            subproblem_name     = self.name,
            n_knots             = self.n_knots,
            dt                  = float(self.dt),

            slack_var_count     = sum(
                                    sv.shape[0] * sv.shape[1]
                                    for sv in self.dvars.slack_vars
                                  ),

            description         = f"Τ={float(self.task.duration)}, " + 
                                  f"μ={self.robot.μ}, " +
                                  f"τ_max={self.robot.τ_max}",
            contact_mode        = self.contact_mode,
        )

        self.transcribed = True

    def create_guess(self) -> CollocationVars[np.ndarray]:
        guess = CollocationVars[np.ndarray](self.n_knots)

        for k in range(self.n_knots):
            guess.append_knot(
                kvars = self.guess_oracle.guess(k),
                duration = float(self.dt)
            )

        # Initialize all slack variables to zero:
        for sv in self.dvars.slack_vars:
            guess.slack_vars.append(np.zeros(sv.shape))

        return guess
