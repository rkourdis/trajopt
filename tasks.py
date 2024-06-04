import math
import numpy as np
import casadi as ca
from typing import Callable, Optional

import intervaltree as ivt
from dataclasses import dataclass

from utilities import ε
from transcription import Constraint
from poses import Pose, load_robot_pose

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
    # of constraints that a solution to the task needs to satisfy: 
    get_kinematic_constraints: Callable[
        [list[ca.SX], list[ca.SX], list[ca.SX], list[ca.SX]], list[Constraint]
    ]

JUMP_TASK: Task = Task(
    name = "jump",
    duration = 2.0,

    contact_periods = [
        ivt.IntervalTree([
            ivt.Interval(0.0, 0.3 + ε),
            ivt.Interval(0.7, 2.0 + ε)
        ])
        for _ in range(4)
    ],

    # RMS of actuation torque:
    traj_error = lambda t, q, v, a, τ, λ: ca.sqrt(τ.T @ τ),

    get_kinematic_constraints = lambda q_k, v_k, a_k, f_pos_k, params: [
        # Feet in standing V at the beginning:
        Constraint(q_k[0] - load_robot_pose(Pose.STANDING_V)[0]),

        # Torso has moved forward by end of jump:
        Constraint(q_k[-1][0], 0.4, ca.inf),

        # Torso is above the ground at a certain height at the end:
        Constraint(q_k[-1][2], lb = params["FLOOR_Z"] + 0.2, ub = ca.inf),

        # The entire robot is static at the beginning and end:
        Constraint(v_k[0]),
        Constraint(v_k[-1])
    ]
)

BACKFLIP_LAND_TASK: Task = Task(
    name = "backflip_land",
    duration = 1.0,

    # Flip will last 600ms:
    contact_periods = [
        ivt.IntervalTree([ivt.Interval(0.3, 1.0 + ε)])
        for _ in range(4)
    ],

    # Minimize torque when airborne, then contact forces:
    traj_error = \
        lambda t, q, v, a, τ, λ: 
            ca.sqrt(τ.T @ τ if t < 0.3 else ca.sum1(ca.diag(λ @ λ.T))),

    get_kinematic_constraints = lambda q_k, v_k, a_k, f_pos_k, params: [
        Constraint(q_k[0][0], lb = -0.2, ub = -0.2),      # We give it 40cm backwards slack
        Constraint(q_k[0][1], lb = 0.0,  ub = 0.0),
        Constraint(q_k[0][2], lb = 0.4,  ub = 0.4),       # 0.44145m for 600deg/s rotation rate
        Constraint(q_k[0][3:6] - ca.SX([0.0, 1.0, 0.0])), # Switched MRP

        # Feet configuration mid-flip
        Constraint(q_k[0][6:] - load_robot_pose(Pose.STANDING_V)[0][6:]),

        # Torso has moved backwards at touchdown:
        Constraint(q_k[math.ceil(0.3 * params["FREQ_HZ"])][0], lb = -0.4, ub = -0.4),

        # # Orientation is horizontal at touchdown:
        # Constraint(q_k[math.ceil(0.3 * params["FREQ_HZ"])][3:6]),

        # Flipping velocity:
        Constraint(v_k[0][3], lb = 0.0,     ub = 0.0),
        Constraint(v_k[0][4], lb = -ca.inf, ub = 0.0),
        Constraint(v_k[0][5], lb = 0.0,     ub = 0.0),

        # Torso is above the ground at a certain height at the end:
        Constraint(q_k[-1][2], lb = params["FLOOR_Z"] + 0.2, ub = ca.inf),

        # The entire robot is static and horizontal at the end. Legs in V configuration.
        Constraint(q_k[-1][3:6]),
        Constraint(v_k[-1]),
        Constraint(q_k[-1][6:] - load_robot_pose(Pose.STANDING_V)[0][6:]),

        # # Front legs are in front of back legs, by a bit!
        # Constraint(f_pos_k[-1][0, 0] - f_pos_k[-1][2, 0], lb = 0.1, ub = ca.inf),
        # Constraint(f_pos_k[-1][1, 0] - f_pos_k[-1][3, 0], lb = 0.1, ub = ca.inf),
    ]
)