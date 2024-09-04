from abc import ABC
from dataclasses import dataclass

import numpy as np
import casadi as ca

from utilities import ca_iter

class ConstraintType(ABC):
    pass

@dataclass
class Constraint(ConstraintType):
    # Elementwise constraint: lb <= expr[:] <= ub
    expr:   ca.SX
    lb:     np.ndarray
    ub:     np.ndarray

    def __init__(self, expr: ca.SX, lb: float = 0.0, ub: float = 0.0):
        assert lb <= ub
        self.expr, self.lb, self.ub = expr, np.full(expr.shape, lb), np.full(expr.shape, ub)

@dataclass
class Bound(ConstraintType):
    # Elementwise variable bound: lb <= variables[:] <= ub
    variables:  ca.SX
    lb:         float   = 0.0
    ub:         float   = 0.0

    def __post_init__(self):
        assert self.lb <= self.ub
        
        for var in ca_iter(self.variables):
            assert var.is_symbolic(), \
                "Bound expression contains non-leaf entries"

@dataclass
class Complementarity(ConstraintType):
    # Variable complementarity constraint: 0 =< var1 \perp var2 >= 0
    var1: str
    var2: str
