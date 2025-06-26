from dataclasses import dataclass
from typing import Callable, Optional

from fractions import Fraction

import casadi as ca
import intervaltree as ivt

from constraints import ConstraintType
from variables import KnotVars
from robot import LeggedRobot

@dataclass()
class TimePeriod:
    # Period between two times, inclusive of end.
    # Unbounded if end == None.
    start:  Fraction
    end:    Optional[Fraction] = None

    def __post_init__(self):
        assert self.start >= 0, "Start time must be >= 0!"

        if self.end != None:
            assert self.start <= self.end, "End time must be >= start time!"

    @staticmethod
    def point(t: Fraction):
        # This will be used for point constraints:
        return TimePeriod(t, t)

@dataclass
class Task:
    # Which robot this task targets:
    robot_type: type[LeggedRobot]

    # Overall trajectory duration (sec):
    duration: Fraction

    # Periods of ground contact for each foot:
    contact_periods: dict[str, ivt.IntervalTree]
    
    # Instantaneous trajectory error to minimise. Given t, q, v, a, τ, λ at a collocation
    # point returns how far away the trajectory is from the desired one at that time:
    traj_error: Callable[[float, KnotVars[ca.SX]], ca.SX]     

    # List of times at which a task-specific constraint must hold.
    # For each one, a "factory-like" callable is held which returns
    # constraint objects given the corresponding knot's decision variables.
    # Additional parameters are passed as kwargs.
    task_constraints: list[
        tuple[
            TimePeriod,
            Callable[[KnotVars[ca.SX], dict], list[ConstraintType]]
        ]
    ]
