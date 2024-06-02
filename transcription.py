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
    
    @staticmethod
    def load_from_vec(num_knots: int, robot: pin.RobotWrapper, vec: ca.DM):
        traj = Trajectory(num_knots = num_knots)

        o, sz = 0, robot.nq - 1
        traj.q_k = unflatten(vec[o : o + num_knots * sz], (sz, 1))

        o, sz = o + sz, robot.nv
        traj.v_k = unflatten(vec[o : o + num_knots * sz], (sz, 1))

        o, sz = o + sz, robot.nv
        traj.a_k = unflatten(vec[o : o + num_knots * sz], (sz, 1))

        o, sz = o + sz, robot.nv
        traj.tau_k = unflatten(vec[o : o + num_knots * sz], (sz, 1))

        o, sz = o + sz, 4 * 3
        traj.λ_k = unflatten(vec[o : o + num_knots * sz], (4, 3))

        o, sz = o + sz, 4 * 3
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
                assert var.name() not in self.bounds, f"Cannot constrain {var} twice"

                self.bounds[var.name()] = (lb, ub)
    
    def get_bounds(self, var_name: str) -> tuple[float, float]:
        return b if (b := self.bounds.get(var_name)) else (-ca.inf, ca.inf)
    