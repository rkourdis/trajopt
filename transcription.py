from fractions import Fraction
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from tasks import Task
from robot import Solo12
from constraints import *
from guesses import GuessOracle
from variables import CollocationVars
from utilities import integrate_state
from dynamics import ADForwardDynamics
from kinematics import ADFootholdKinematics

@dataclass
class Subproblem:
    name:           str
    task:           Task
    freq_hz:        Fraction
    robot:          Solo12
    guess_oracle:   GuessOracle

    dvars:          CollocationVars[ca.SX]   = field(init = False)
    n_knots:        int                      = field(init = False)
    objective:      ca.SX                    = field(init = False)
    constraints:    list[ConstraintType]     = field(default_factory = list, init = False)

    transcribed:    bool                     = field(default = False, init = False)

    def __post_init__(self):
        self.dt = 1 / self.freq_hz

        if self.task.duration % self.dt != 0:
            floor_k = self.task.duration // self.dt

            # TODO: Test this!
            raise ValueError(
                f"Make sure task duration ({self.task.duration})s is divisible by Δt ({self.dt}s). " + \
                f"Closest are: {floor_k * self.dt}s or {(floor_k + 1) * self.dt}s."
            )

        # We'll have (duration / dt) time periods. Add an extra knot for the final point:        
        self.n_knots                                = int(self.task.duration * self.freq_hz + 1)
        self.dvars                                  = CollocationVars[ca.SX](self.n_knots)

    def _create_vars(self) -> None:
        for k in range(self.n_knots):
            self.dvars.q_k.append(ca.SX.sym(f"{self.name}_q_{k}", self.robot.cmodel.nq - 1))         # 18 x 1 (floating base MRP)
            self.dvars.v_k.append(ca.SX.sym(f"{self.name}_v_{k}", self.robot.cmodel.nv))             # 18 x 1
            self.dvars.a_k.append(ca.SX.sym(f"{self.name}_a_{k}", self.robot.cmodel.nv))             # 18 x 1
            self.dvars.τ_k.append(ca.SX.sym(f"{self.name}_τ_{k}", len(self.robot.actuated_joints)))  # 12 x 1 
            self.dvars.λ_k.append(ca.SX.sym(f"{self.name}_λ_{k}", len(self.robot.feet), 3))          # 4  x 3
            self.dvars.f_pos_k.append(ca.SX.sym(f"{self.name}_f_pos_{k}", len(self.robot.feet), 3))  # 4  x 3

    # Pointwise variable constraints at each knot (dynamics, kinematics, limits):
    #############################################################################
    def _add_pointwise_constraints(self, k: int) -> None:
        kv = self.dvars.get_vars_at_knot(k)

        self.constraints.extend([
            # Forward dynamics accelerations:
            Constraint(kv.a - self.fd(kv.q, kv.v, kv.τ, kv.λ)),

            # Robot torso cannot go below the ground:
            Bound(kv.q[2], lb = self.robot.floor_z + 0.08, ub = ca.inf),

            # Joint torque limits in N*m:
            Bound(kv.τ, lb = -1.9, ub = 1.9),
    
            # Forward foothold kinematics:
            Constraint(kv.f_pos - self.fk(kv.q))
        ])
    #############################################################################

    # Integration constraints using fully implicit Euler:
    #####################################################
    def _add_integration_constraints(self, k: int) -> None:
        assert k > 0, "Cannot add integration constraint on the first knot!"

        q, v, a         = self.dvars.q_k[k],   self.dvars.v_k[k], self.dvars.a_k[k]
        q_prev, v_prev  = self.dvars.q_k[k-1], self.dvars.v_k[k-1]

        self.constraints.extend([
            Constraint(v - (v_prev + float(self.dt) * a)),
            Constraint(q - integrate_state(q_prev, float(self.dt) * v))
        ])
    #####################################################

    # Foot contact constraints (reaction forces, foothold positioning):
    ###################################################################
    def _add_contact_constraints(self, k: int, foot_idx: int) -> None:
        t, t_prev = k * self.dt, (k - 1) * self.dt
        λ, fp     = self.dvars.λ_k[k][foot_idx, :], self.dvars.f_pos_k[k][foot_idx, :]

        floor_z = self.robot.floor_z
        μ       = self.robot.μ

        # Get foot contact times from the task description:
        contact_times = self.task.contact_periods[self.robot.feet[foot_idx]]

        # If foot isn't in contact, don't allow forces and constrain Z >= floor:
        if not contact_times.overlaps(t):
            self.constraints.extend([
                Bound(λ,     lb = 0.0, ub = 0.0),
                Bound(fp[2], lb = floor_z, ub = ca.inf)
            ])

            return 

        # Foot in contact
        # ---------------

        # 1. Allow Z-up reaction force:
        self.constraints.append(Bound(λ[2], lb = 0.0, ub = ca.inf))

        # 2. Force the foot to be on the floor:
        self.constraints.append(Bound(fp[2], lb = floor_z, ub = floor_z))

        # 3. Force the foot to not slip in the XY plane if it was in contact previously:
        if k - 1 >= 0 and contact_times.overlaps(t_prev):
            fp_prev = self.dvars.f_pos_k[k-1][foot_idx, :]
            self.constraints.append(Constraint(fp[:2] - fp_prev[:2]))
        
        # 4. Make sure the tangential contact force lies in the friction cone.
        #    We will achieve this by adding the following constraints:
        #       fabs(λ_x) <= μ * λ_z  and  fabs(λ_y) <= μ * λ_z
        #    This is a pyramidal approximation of the friction cone.
        #    The fabs(.) operation introduces derivative discontinuity in the 
        #    constraints and makes optimization difficult. We will reformulate
        #    the constraints using complementary slack variables to remove
        #    that discontinuity:
        #       If x = x_pos - x_neg and 0 =< x_pos \perp x_neg >= 0,
        #       then |x| = x_pos + x_neg.
        λ_xy_pos = ca.SX.sym(f"{self.name}_λxy_pos_{foot_idx}_{k}", 2)
        λ_xy_neg = ca.SX.sym(f"{self.name}_λxy_neg_{foot_idx}_{k}", 2)

        self.dvars.slack_vars.extend([λ_xy_pos, λ_xy_neg])

        # 0 =< λ_xy_pos[:] \perp λ_xy_neg[:] >= 0:
        self.constraints.extend([
            Bound(λ_xy_pos, lb = 0.0, ub = ca.inf),
            Bound(λ_xy_neg, lb = 0.0, ub = ca.inf),
            Complementarity(λ_xy_pos[0].name(), λ_xy_neg[0].name()),
            Complementarity(λ_xy_pos[1].name(), λ_xy_neg[1].name()),
        ])

        # λ_xy = λ_xy_pos - λ_xy_neg:
        self.constraints.append(Constraint(λ[:2].T - (λ_xy_pos - λ_xy_neg)))

        # fabs(λ_xy) <= λ_z * μ:
        # NOTE: In this case, the complementarity constraint isn't strictly
        #       required, as λ_xy_pos + λ_xy_neg >= |λ_xy| always (triangle inequality).
        #       Therefore, if λ_xy_pos + λ_xy_neg is below the friction limit, then |λ_xy|
        #       must be as well. If the solver needs more tangential force, it should
        #       figure out that it can get maximum by setting one of the variables to zero.
        #       However, adding the constraint seems to help convergence speed in practice.
        self.constraints.append(
            Constraint(
                ca.repmat(μ * λ[2], 2) - (λ_xy_pos + λ_xy_neg),
                lb = 0.0, ub = ca.inf
            )
        )
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
        self.fk = ADFootholdKinematics(self.robot)

        # Add dynamics constraints for the above decision variables:
        for k in range(self.n_knots):
            if not is_subsequent or k > 0:
                self._add_pointwise_constraints(k)

                for foot_idx in range(len(self.robot.feet)):
                    self._add_contact_constraints(k, foot_idx)

            if k > 0:
                self._add_integration_constraints(k)

        # Add task-specific trajectory constraints:
        for t, get_constr in self.task.task_constraints:
            if t < 0 or t > self.task.duration:
                raise ValueError(f"Task constraint time ({t}) is not in [0.0, {self.task.duration}].")
            
            if t % self.dt != 0:
                raise ValueError(f"Task constraint time ({t}) is not divisible by Δt ({self.dt}).")

            kvars = self.dvars.get_vars_at_knot(int(t / self.dt))
            self.constraints.extend(get_constr(kvars, self.robot))

        # Create expression for the scalar problem objective:
        self.objective = ca.SX.zeros(1)
        
        for k in range(self.n_knots):
            err = self.task.traj_error(
                float(k * self.dt), self.dvars.get_vars_at_knot(k)
            )

            # We average the trajectory error over all knots:
            weight = 1.0 / self.n_knots
            self.objective += weight * err
            
        self.transcribed = True

    def create_guess(self) -> CollocationVars[np.ndarray]:
        guess = CollocationVars[np.ndarray](self.n_knots)

        for k in range(self.n_knots):
            guess.append_knot(self.guess_oracle.guess(self.dt * k))

        # Initialize all slack variables to zero:
        for sv in self.dvars.slack_vars:
            guess.slack_vars.append(np.zeros(sv.shape))

        return guess

    # Loads a solution to this subproblem from flattened numeric vectors:    
    def load_solution(
            self,
            soln_vec: np.ndarray,
        ) -> CollocationVars[np.ndarray]:

        return CollocationVars.unflatten(self.n_knots, soln_vec)
    