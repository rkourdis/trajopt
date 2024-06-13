from dataclasses import dataclass

import numpy as np
import casadi as ca

@dataclass
class Constraint:
    # Elementwise constraint: lb <= expr[:] <= ub
    expr:   ca.SX
    lb:     np.array
    ub:     np.array

    def __init__(self, expr: ca.SX, lb: float = 0.0, ub: float = 0.0):
        assert lb <= ub
        self.expr, self.lb, self.ub = expr, np.full(expr.shape, lb), np.full(expr.shape, ub)

@dataclass
class Bound:
    # Elementwise variable bound: lb <= variables[:] <= ub
    variables:  ca.SX
    lb:         float   = 0.0
    ub:         float   = 0.0

    def __post_init__(self):
        assert self.lb <= self.ub
        
        for i in range(self.variables.shape[0]):
            for j in range(self.variables.shape[1]):
                assert self.variables[i,j].is_symbolic(), \
                    "Bound expression contains non-leaf entries"


class VariableBounds:
    def __init__(self):
        self.bounds: dict[str, tuple[float, float]] = {}

    def add(self, bound: Bound):
        for i in range(bound.variables.shape[0]):
            for j in range(bound.variables.shape[1]):
                var = bound.variables[i, j]
                
                # If there's an existing bound, make it tighter:
                existing_bound = self.get_bound(var.name())
                self.bounds[var.name()] = (
                    max(bound.lb, existing_bound[0]),
                    min(bound.ub, existing_bound[1])
                )
    
    def get_bound(self, var_name: str) -> tuple[float, float]:
        return b if (b := self.bounds.get(var_name)) else (-ca.inf, ca.inf)
    