from itertools import chain
from dataclasses import dataclass

import numpy as np
import casadi as ca
import pinocchio as pin

from utilities import flatten, unflatten

@dataclass
class Trajectory:
    # This trajectory corresponds to a single task. The entire problem
    # may stitch two or more trajectories together as multiple tasks
    # can be solved at the same time.
    num_knots: int
    q_k: list[np.array]     = None    # 18x1
    v_k: list[np.array]     = None    # 18x1
    a_k: list[np.array]     = None    # 18x1
    tau_k: list[np.array]   = None    # 12x1
    λ_k: list[np.array]     = None    # 4x3
    f_pos_k: list[np.array] = None    # 4x3

    # For debugging purposes:
    slack_vars: np.array    = None

    def flatten(self) -> ca.DM:
        # assert False, "TODO!"

        return ca.vertcat(
            flatten(self.q_k), flatten(self.v_k), flatten(self.a_k),
            flatten(self.tau_k), flatten(self.λ_k), flatten(self.f_pos_k)
        )
    
    # Interpolate by simple repetition of knots:
    def interpolate(self, target_knots: int):
        # assert False, "TODO!"

        assert target_knots % self.num_knots == 0, \
            "Target knot count must be divisible by source"
        
        reps = target_knots // self.num_knots
        result = Trajectory(num_knots = target_knots)

        # np.repeat(reps) ...
        repeat = lambda vars: list(
            chain.from_iterable(
                [np.copy(var) for _ in range(reps)] for var in vars
            )
        )

        result.q_k = repeat(self.q_k)
        result.v_k = repeat(self.v_k)
        result.a_k = repeat(self.a_k)
        result.tau_k = repeat(self.tau_k)
        result.λ_k = repeat(self.λ_k)
        result.f_pos_k = repeat(self.f_pos_k)

        return result

    @staticmethod
    def load_from_vec(num_knots: int, robot: pin.RobotWrapper, vec: ca.DM):
        # assert False, "TODO!"

        traj = Trajectory(num_knots = num_knots)

        o, sz = 0, robot.nq - 1
        traj.q_k = unflatten(vec[o : o + num_knots * sz], (sz, 1))

        o, sz = o + sz * num_knots, robot.nv
        traj.v_k = unflatten(vec[o : o + num_knots * sz], (sz, 1))

        o, sz = o + sz * num_knots, robot.nv
        traj.a_k = unflatten(vec[o : o + num_knots * sz], (sz, 1))

        o, sz = o + sz * num_knots, 12
        traj.tau_k = unflatten(vec[o : o + num_knots * sz], (sz, 1))

        o, sz = o + sz * num_knots, 4 * 3
        traj.λ_k = unflatten(vec[o : o + num_knots * sz], (4, 3))

        o, sz = o + sz * num_knots, 4 * 3
        traj.f_pos_k = unflatten(vec[o : o + num_knots * sz], (4, 3))


        # TODO!!!

        # Slack variables - always appended to the end of the variable vector:
        o += sz * num_knots
        traj.slack_vars = vec[o:]

        return traj
