import math
import numpy as np
import casadi as ca
from itertools import chain
from typing import Callable, Union

import intervaltree as ivt
from dataclasses import dataclass

from utilities import ε
from poses import Pose, load_robot_pose
from constraints import Constraint, Bound

@dataclass
class Task:
    # Task name:
    name: str

    # Overall trajectory duration:
    duration: float

    # Periods of ground contact for each foot:
    contact_periods: list[ivt.IntervalTree]
    
    # Instantaneous trajectory error to minimise. Given t, q, v, a, τ, λ at a collocation
    # point returns how far away the trajectory is from the desired one at that time:
    traj_error: Callable[[float, ca.SX, ca.SX, ca.SX, ca.SX, ca.SX], float]     

    # Given the kinematic decision variables (q_k, v_k, α_k, f_pos_k), return a list
    # of constraints or variable bounds that a task solution needs to satisfy: 
    get_kinematic_constraints: Callable[
        [list[ca.SX], list[ca.SX], list[ca.SX], list[ca.SX]],
        list[Union[Constraint, Bound]]
    ]

JUMP_TASK: Task = Task(
    name = "jump",
    duration = 1.0,

    contact_periods = [
        ivt.IntervalTree([
            ivt.Interval(0.0, 0.3 + ε),
            ivt.Interval(0.6, 1.0 + ε)
        ])
        for _ in range(4)
    ],

    # RMS of actuation torque:
    traj_error = lambda t, q, v, a, τ, λ: ca.sqrt(τ.T @ τ),

    get_kinematic_constraints = lambda q_k, v_k, a_k, f_pos_k, params: [
        # Feet in standing V at the beginning:
        Constraint(q_k[0] - load_robot_pose(Pose.STANDING_V)[0]),

        # Torso has moved forward by end of jump:
        Bound(q_k[-1][0], 0.4, ca.inf),

        # Torso is above the ground at a certain height at the end:
        Bound(q_k[-1][2], params["FLOOR_Z"] + 0.2, ca.inf),

        # The entire robot is static at the beginning and end:
        Bound(v_k[0]),
        Bound(v_k[-1]),
    ]
)