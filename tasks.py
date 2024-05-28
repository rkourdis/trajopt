import numpy as np
import casadi as ca
from typing import Callable, Optional

import intervaltree as ivt
from dataclasses import dataclass

from utilities import ε
from initialisations import Guess, load_initial_guess

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
    q_initial = load_initial_guess(Guess.STANDING_V)[0],
    q_final   = None,
    # q_final   = load_initial_guess(Guess.STANDING_V)[0],

    # The robot should be stable at the beginning and end:
    v_initial = np.zeros((18, 1)),
    v_final   = np.zeros((18, 1)),
)


"""
# Trajectory error function. 
        def traj_err(t, q, v, a, τ):
            if contact_times[0].overlaps(t):
                return τ.T @ τ #+ (q[:3].T @ q[:3]) * 1e-1

            return 
            # if 1.0 <= t and t < 1.5:
            #     com_orientation_err = q[:6] - ca.SX([0, 0, 0.2, 0.0, 0.0, 0.0])
            #     return com_orientation_err.T @ com_orientation_err + v[6:].T @ v[6:]

            # z_des = ca.sin(2 * ca.pi * t) * 0.1
            # com_orientation_err = q[:6] - ca.SX([0, 0, z_des, 0.0, 0.0, 0.0])
            # return com_orientation_err.T @ com_orientation_err + v[6:].T @ v[6:]
        

"""