import math
import numpy as np
import casadi as ca
from typing import Callable
from itertools import chain

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
    
    # Instantaneous trajectory error to minimise. Given t, q, v, a, τ at a collocation
    # point returns how far away the trajectory is from the desired one at that time:
    traj_error: Callable[[float, ca.SX, ca.SX, ca.SX, ca.SX], float]     

    # Given the kinematic decision variables (q_k, v_k, α_k), return a list
    # of constraints that a solution to the task needs to satisfy: 
    get_kinematic_constraints: Callable[
        [list[ca.SX], list[ca.SX], list[ca.SX]], list[Constraint]
    ]

    def __post_init__(self):
        # Merge all overlapping contact intervals:
        for tree in self.contact_periods:
            tree.merge_overlaps(strict = False)

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
    traj_error = lambda t, q, v, a, τ: ca.sqrt(τ.T @ τ),

    get_kinematic_constraints = lambda q_k, v_k, a_k, params: [
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

# BACKFLIP_LAND_TASK: Task = Task(
#     name = "backflip_land",
#     duration = 1.0,

#     # Flip will last 600ms:
#     contact_periods = [
#         ivt.IntervalTree([ivt.Interval(0.3, 1.0 + ε)]),  # FR
#         ivt.IntervalTree([ivt.Interval(0.3, 1.0 + ε)]),  # FL
#         ivt.IntervalTree([ivt.Interval(0.35, 1.0 + ε)]), # HR
#         ivt.IntervalTree([ivt.Interval(0.35, 1.0 + ε)]), # HL
#     ],

#     # Minimize joint velocity when airborne, then contact forces and orientation:
#     traj_error = \
#         lambda t, q, v, a, τ: 
#             # ca.sqrt(τ.T @ τ if t < 0.3 else ca.sum1(ca.diag(λ @ λ.T))),
#             ca.sqrt(v[6:].T @ v[6:]         
#                 if t < 0.3                  
#             else ca.sum1(ca.diag(λ @ λ.T))), # + 12.0 * q[3:6].T @ q[3:6],

#     get_kinematic_constraints = lambda q_k, v_k, a_k, params: [
#         Constraint(q_k[0][0], lb = -0.2, ub = -0.2),      # We give it 40cm backwards slack
#         Constraint(q_k[0][1], lb = 0.0,  ub = 0.0),
#         Constraint(q_k[0][2], lb = 0.4,  ub = 0.4),       # 0.44145m for 600deg/s rotation rate
#         Constraint(q_k[0][3:6] - ca.SX([0.0, 1.0, 0.0])), # Switched MRP

#         # Feet configuration mid-flip
#         Constraint(q_k[0][6:] - load_robot_pose(Pose.STANDING_V)[0][6:]),

#         # Torso has moved backwards at touchdown:
#         Constraint(q_k[math.ceil(0.3 * params["FREQ_HZ"])][0], lb = -0.4, ub = -0.4),

#         # # Orientation is horizontal at touchdown:
#         # Constraint(q_k[math.ceil(0.3 * params["FREQ_HZ"])][3:6]),

#         # Flipping velocity:
#         Constraint(v_k[0][3], lb = 0.0,     ub = 0.0),
#         Constraint(v_k[0][4], lb = -ca.inf, ub = 0.0),
#         Constraint(v_k[0][5], lb = 0.0,     ub = 0.0),

#         # # Torso is above the ground at a certain height at the end:
#         # Constraint(q_k[-1][2], lb = params["FLOOR_Z"] + 0.2, ub = ca.inf),

#         # The entire robot is static and horizontal at the end. Legs in V configuration.
#         Constraint(q_k[-1][3:6]),
#         Constraint(v_k[-1]),
#         Constraint(q_k[-1][6:] - load_robot_pose(Pose.STANDING_V)[0][6:]),

#         # # Front legs are in front of back legs, by a bit!
#         # Constraint(f_pos_k[-1][0, 0] - f_pos_k[-1][2, 0], lb = 0.1, ub = ca.inf),
#         # Constraint(f_pos_k[-1][1, 0] - f_pos_k[-1][3, 0], lb = 0.1, ub = ca.inf),
#     ] + \
    
    
#     list(
#         chain.from_iterable([
#             [
#                 # Keep FR/FL HFE at less than fully folded as there's the joint stopper:
#                 Constraint(q_k[k][7], lb = -ca.inf, ub = np.deg2rad(85)),
#                 Constraint(q_k[k][10], lb = -ca.inf, ub = np.deg2rad(85)),

#                 # Keep the front / hind two knees at an angle of less than 125 / 120 deg after the landing.
#                 # This is a hack to make sure the knees don't go below the ground :) 
#                 Constraint(q_k[k][8], lb =  -np.deg2rad(125), ub = ca.inf),     # Front
#                 Constraint(q_k[k][11], lb = -np.deg2rad(125), ub = ca.inf),
#                 Constraint(q_k[k][14], lb = -ca.inf, ub = np.deg2rad(120)),     # Hind
#                 Constraint(q_k[k][17], lb = -ca.inf, ub = np.deg2rad(120))
#             ]
#             for k in range(math.ceil(0.3 * params["FREQ_HZ"]), params["N_KNOTS"])
#         ]
#     ))
# )

BACKFLIP_LAUNCH_TASK: Task = Task(
    name = "backflip_launch",
    duration = 1.0,

    contact_periods = [
        ivt.IntervalTree([
            ivt.Interval(0.0, 0.4),
            ivt.Interval(0.6, 1.0 + ε)
        ])
        for _ in range(4)
    ],

    # RMS of actuation torque:
    traj_error = lambda t, q, v, a, τ: ca.sqrt(τ.T @ τ),

    get_kinematic_constraints = lambda q_k, v_k, a_k, params: [
        Constraint(q_k[0] - load_robot_pose(Pose.STANDING_V)[0]),

        # Due to a slight foot constraint drift, it's impossible to achieve both
        # torso Z and joint angles after the jump.
        # The drift seems to happen around the knot when contact is remade
        # and it might be because the constrained accelerations are not integrated
        # immediately.
        Constraint(q_k[-1][3:] - load_robot_pose(Pose.STANDING_V)[0][3:]),

        Constraint(v_k[0]),
        Constraint(v_k[-1])
    ] + \
    [ Constraint(q_k[k][3:6]) for k in range(len(q_k)) ] + \
    [ Constraint(q_k[k][0:2]) for k in range(len(q_k)) ]
)
