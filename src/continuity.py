from typing import Callable
from dataclasses import dataclass

import casadi as ca

from variables import KnotVars
from constraints import Constraint

@dataclass
# This class holds continuity information for all trajectory decision
# variables. It will be used to generate constraints that ensure
# continuity between the last knot of each subproblem and the first knot
# of the next. A conversion method can be provided for each one of the
# variables, which: given the variable at the last knot of a subproblem
# (A[-1].x), returns what the variable must be in the first knot 
# of the following subproblem (B[0].x).
# Constraints of the form A[-1].x - B[0].x == 0 will then be generated.
class ContinuityInfo():
    q:      Callable[[ca.SX], ca.SX] = lambda x: x
    v:      Callable[[ca.SX], ca.SX] = lambda x: x
    a:      Callable[[ca.SX], ca.SX] = lambda x: x
    τ:      Callable[[ca.SX], ca.SX] = lambda x: x
    λ:      Callable[[ca.SX], ca.SX] = lambda x: x
    f_pos:  Callable[[ca.SX], ca.SX] = lambda x: x

    def create_constraints(self, before: KnotVars[ca.SX], after: KnotVars[ca.SX]) -> list[Constraint]:
        return [
            Constraint(self.q(before.q) - after.q, lb = 0.0, ub = 0.0),
            Constraint(self.v(before.v) - after.v, lb = 0.0, ub = 0.0),
            Constraint(self.a(before.a) - after.a, lb = 0.0, ub = 0.0),
            Constraint(self.τ(before.τ) - after.τ, lb = 0.0, ub = 0.0),
            Constraint(self.λ(before.λ) - after.λ, lb = 0.0, ub = 0.0),
            Constraint(self.f_pos(before.f_pos) - after.f_pos, lb = 0.0, ub = 0.0),
        ]