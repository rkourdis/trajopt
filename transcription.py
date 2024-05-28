from dataclasses import dataclass

import numpy as np
import casadi as ca

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
    