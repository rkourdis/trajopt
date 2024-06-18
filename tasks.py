from typing import Callable
from fractions import Fraction
from dataclasses import dataclass

import casadi as ca
import intervaltree as ivt

from constraints import *
from robot import Solo12
from utilities import frac_ε
from variables import KnotVars
from poses import Pose, load_robot_pose

@dataclass
class Task:
    # Overall trajectory duration (sec):
    duration: Fraction

    # Periods of ground contact for each foot:
    contact_periods: dict[str, ivt.IntervalTree]
    
    # Instantaneous trajectory error to minimise. Given t, q, v, a, τ, λ at a collocation
    # point returns how far away the trajectory is from the desired one at that time:
    traj_error: Callable[[float, KnotVars[ca.SX]], ca.SX]     

    # List of times at which a task-specific constraint must hold.
    # For each one, a "factory-like" callable is held which can create
    # constraint objects given the corresponding knot's decision variables
    # and the robot model:
    task_constraints: list[
        tuple[
            Fraction,
            Callable[[KnotVars[ca.SX], Solo12], list[ConstraintType]]
        ]
    ]

JumpTaskFwd: Task = Task(
    duration = Fraction("1.0"),
    traj_error = lambda t, kvars: ca.norm_2(kvars.τ),

    contact_periods = {
        foot: ivt.IntervalTree([
            # Add ε at the end of the intervals because .overlaps() is
            # not inclusive of the end time:
            ivt.Interval(0.0, 0.3 + frac_ε), ivt.Interval(0.6, 1.0 + frac_ε)
        ])

        for foot in ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]
    },

    task_constraints = [
        (
            Fraction("0.0"), 

            lambda kv, solo: [
                # Feet in standing V at the beginning:
                Constraint(kv.q - load_robot_pose(Pose.STANDING_V)[0]),

                # Robot is static:
                Bound(kv.v)
            ]
        ),
        (
            Fraction("1.0"),

            lambda kv, solo: [
                # Torso has moved forward:
                Bound(kv.q[0], 0.4, ca.inf),
            
                # Torso is horizontal above the ground at a certain height:
                Bound(kv.q[2], solo.floor_z + 0.2, ca.inf),
                Bound(kv.q[3:6]),

                # Robot is static:
                Bound(kv.v)
            ]
        )
    ],
)

JumpTaskBwd: Task = Task(
    duration = Fraction("1.0"),
    traj_error = lambda t, kvars: ca.norm_2(kvars.τ),

    contact_periods = {
        foot: ivt.IntervalTree([ivt.Interval(0.0, 0.3 + frac_ε), ivt.Interval(0.6, 1.0 + frac_ε)])
        for foot in ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]
    },

    task_constraints = [
        # Initial constraints will be added for subproblem continuity. Add final only:
        (
            Fraction("1.0"), lambda kv, solo: [
                # Robot has gone back to original configuration:
                Constraint(kv.q - load_robot_pose(Pose.STANDING_V)[0]),
                
                # Robot is static:
                Bound(kv.v)
            ]
        )
    ],
)