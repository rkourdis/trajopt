from itertools import chain
from dataclasses import dataclass

import numpy as np
import casadi as ca
import pinocchio as pin

from utilities import flatten, unflatten

@dataclass
class Trajectory:
    num_knots: int
    q_k: list[np.array]     = None    # 18x1
    v_k: list[np.array]     = None    # 18x1
    a_k: list[np.array]     = None    # 18x1
    tau_k: list[np.array]   = None    # 12x1
    λ_k: list[np.array]     = None    # 4x3
    f_pos_k: list[np.array] = None    # 4x3

    def flatten(self) -> ca.DM:
        return ca.vertcat(
            flatten(self.q_k), flatten(self.v_k), flatten(self.a_k),
            flatten(self.tau_k), flatten(self.λ_k), flatten(self.f_pos_k)
        )
    
    # Interpolate by simple repetition of knots:
    def interpolate(self, target_knots: int):
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

        return traj
    
@dataclass
class Constraint:
    expr:   ca.SX
    lb:     np.array
    ub:     np.array

    # Create a new elementwise constraint: lb <= expr[:] <= ub
    def __init__(self, expr: ca.SX, lb: float = 0.0, ub: float = 0.0):
        assert lb <= ub
        self.expr, self.lb, self.ub = expr, np.full(expr.shape, lb), np.full(expr.shape, ub)

class VariableBounds:
    def __init__(self):
        self.bounds: dict[str, tuple[float, float]] = {}

    def add_expr(self, expr: ca.SX, lb: float, ub: float):
        assert lb <= ub, "Lower bound must be <= upper bound"

        for i in range(expr.shape[0]):
            for j in range(expr.shape[1]):
                var = expr[i, j]
                assert var.is_symbolic(), "Bound expression contains non-leaf entries"
                
                # If there's an existing bound, make it tighter:
                existing_bound = self.get_bounds(var.name())
                self.bounds[var.name()] = (max(lb, existing_bound[0]), min(ub, existing_bound[1]))
    
    def get_bounds(self, var_name: str) -> tuple[float, float]:
        return b if (b := self.bounds.get(var_name)) else (-ca.inf, ca.inf)
    