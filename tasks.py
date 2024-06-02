import numpy as np
import casadi as ca
from typing import Callable, Optional

import intervaltree as ivt
from dataclasses import dataclass

from utilities import ε
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

    # Trajectory boundaries:
    q_initial:  Optional[np.array]
    q_final:    Optional[np.array]
    v_initial:  Optional[np.array]
    v_final:    Optional[np.array]

JUMP_TASK: Task = Task(
    name = "jump",
    duration = 2.0,

    contact_periods = [
        ivt.IntervalTree([
            ivt.Interval(0.0, 0.3 + 1/40),
            ivt.Interval(0.7, 2.0 + 1/40)
        ])
        for _ in range(4)
    ],

    # RMS of actuation torque:
    traj_error = lambda t, q, v, a, τ: ca.sqrt(τ.T @ τ),

    # Feet must be in a standing V configuration at the beginning and end:
    q_initial = load_robot_pose(Pose.STANDING_V)[0],
    q_final   = None,

    # The robot should be stable at the beginning and end:
    v_initial = np.zeros((18, 1)),
    v_final   = np.zeros((18, 1)),
)