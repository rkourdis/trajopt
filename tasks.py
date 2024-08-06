from dataclasses import dataclass
from typing import Callable, Optional

from fractions import Fraction
F = Fraction

import casadi as ca
import intervaltree as ivt

from constraints import *
from configurations import *
from utilities import frac_ε
from variables import KnotVars
from poses import Pose, load_robot_pose

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

# -------------------------------------------------
# Robot gets straight up from folded configuration:
# -------------------------------------------------

GetUpTask: Task = Task(
    duration = F("0.8"),
    traj_error = lambda t, kvars: ca.norm_2(kvars.τ),

    # Feet always in contact:
    contact_periods = {
        # NOTE: We add ε at the end of the intervals as .overlaps() is
        # not inclusive of the end time:
        foot: ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.8") + frac_ε)])
        for foot in ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]
    },

    task_constraints = [
        (
            TimePeriod.point(F("0.0")),

            lambda kv, **kwargs: [
                # All joints in the folded configuration:
                Constraint(kv.q[3:] - create_state_vector(kwargs["solo"].robot, FOLDED_JOINT_MAP)[3:]),

                # Torso XY at (0, 0).
                # We don't constrain torso Z. If the feet are stuck to the floor,
                # there's only one solution for the Z and the solver will find it.
                # Finding the height a-priori and enfocing the constraint
                # is valid, but might trouble the solver due to duplicate constraints.
                Bound(kv.q[:2]),

                # Torso and joints static at the beginning:
                Bound(kv.v),
            ]
        ),
        (
            TimePeriod.point(F("0.8")),

            lambda kv, **kwargs: [
                # End up static in the V configuration at (0, 0):
                Constraint(kv.q[3:] - load_robot_pose(Pose.STANDING_V)[0][3:]),
                Bound(kv.q[:2]),
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod(F("0.0"), end = None),
            lambda kv, **kwargs: [
                # Robot torso cannot go downwards during the entire trajectory.
                # This is to avoid solutions where the robot initially falls
                # to conserve torque:
                Bound(kv.v[2], lb = 0.0, ub = ca.inf),

                # Avoid solutions where the robot pitches, rolls or yaws:
                Bound(kv.v[3:6], lb = 0.0, ub = 0.0)
            ]
        )

    ],
)

# ----------------------------------
# Robot jumps up for 400ms in place:
# ----------------------------------

JumpTaskInPlace: Task = Task(
    duration = F("1.2"),

    # We minimize ground contact forces for a
    # smooth landing:
    traj_error = lambda t, kvars: ca.sqrt(
            kvars.λ[:, 2].T @ kvars.λ[:, 2]
            if t >= 0.7 and t <= 0.9 else
            0.0
        ),

    contact_periods = {
        foot: ivt.IntervalTree([
            ivt.Interval(F("0.0"), F("0.3") + frac_ε),
            ivt.Interval(F("0.7"), F("1.2") + frac_ε)
        ])

        for foot in ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]
    },

    task_constraints = [
        (
            TimePeriod.point(F("0.0")),

            # We constrain the entire initial state here (including torso Z),
            # despite what we did above.
            # Eh, it still works ¯\_(ツ)_/¯
            lambda kv, **kwargs: [
                Constraint(kv.q - load_robot_pose(Pose.STANDING_V)[0]),
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod.point(F("1.2")),

            # Ending state same as initial:
            lambda kv, **kwargs: [
                Constraint(kv.q - load_robot_pose(Pose.STANDING_V)[0]),
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod(start = F("0.0"), end = None),

            # Keep the robot torso above the ground by 6cm, always. This is to
            # avoid the landing being a bit _too_ low!
            lambda kv, **kwargs: [
                Bound(kv.q[2], lb = kwargs["solo"].floor_z + 0.06, ub = ca.inf),
            ]
        )
    ],
)

# -----------------------------------------------------------------------------
# Robot jumps 20cm forward.
# NOTE: I've broken my bot doing this and JumpTaskBwd. I don't remember exactly
#       why, but the reason might not have been the trajectory.
# -----------------------------------------------------------------------------

JumpTaskFwd: Task = Task(
    duration = F("1.0"),

    # Minimize torque during the jump:
    traj_error = lambda t, kvars: ca.norm_2(kvars.τ),

    # 300ms jump:
    contact_periods = {
        foot: ivt.IntervalTree([
            ivt.Interval(F("0.0"), F("0.3") + frac_ε),
            ivt.Interval(F("0.6"), F("1.0") + frac_ε)
        ])

        for foot in ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]
    },

    task_constraints = [
        (
            TimePeriod.point(F("0.0")), 

            # Feet in static standing V configuration at the beginning:
            lambda kv, **kwargs: [
                Constraint(kv.q[:2] - load_robot_pose(Pose.STANDING_V)[0][:2]),
                Constraint(kv.q[3:] - load_robot_pose(Pose.STANDING_V)[0][3:]),
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod.point(F("1.0")),

            lambda kv, **kwargs: [
                # Torso has moved forward by at least 40cm:
                Bound(kv.q[0], 0.4, ca.inf),
            
                # Torso is horizontal above the ground at a height of at least 20cm:
                Bound(kv.q[2], kwargs["solo"].floor_z + 0.2, ca.inf),
                Bound(kv.q[3:6]),

                # Robot is static:
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod(start = F("0.0"), end = None),

            # Keep torso above the floor by 8cm, always:
            lambda kv, **kwargs: [
                Bound(kv.q[2], lb = kwargs["solo"].floor_z + 0.08, ub = ca.inf),
            ]
        )
    ],
)

# --------------------------------------------------------------------------
# Robot jumps 20cm backward. To be used after `JumpTaskFwd` as there
# are no initial constraints!
# --------------------------------------------------------------------------

JumpTaskBwd: Task = Task(
    duration = F("1.0"),
    traj_error = lambda t, kvars: ca.norm_2(kvars.τ),

    contact_periods = {
        foot: ivt.IntervalTree([
            ivt.Interval(F("0.0"), F("0.3") + frac_ε),
            ivt.Interval(F("0.6"), F("1.0") + frac_ε)
        ])

        for foot in ["FR_FOOT", "FL_FOOT", "HR_FOOT", "HL_FOOT"]
    },

    task_constraints = [
        (
            # We don't add initial state constraints, continuity with `JumpTaskFwd`
            # will be enforced:
            TimePeriod.point(F("1.0")),
            
            lambda kv, **kwargs: [
                # Robot has gone back to original configuration:
                Constraint(kv.q[:2] - load_robot_pose(Pose.STANDING_V)[0][:2]),
                Constraint(kv.q[3:] - load_robot_pose(Pose.STANDING_V)[0][3:]),
                
                # Robot is static:
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod(start = F("0.0"), end = None),

            lambda kv, **kwargs: [
                Bound(kv.q[2], lb = kwargs["solo"].floor_z + 0.08, ub = ca.inf),
            ]
        )
    ],
)

# -------- #
# Backflip #
# -------- #

# --------------------------
# --- Helper constraints ---
# --------------------------

# We want the backflip to be left-right symmetric for simplicity:
LR_Symmetry_Constraints = lambda kv, **kwargs: [
    Constraint(kv.q[kwargs["solo"].q_off("FR_KFE")] - kv.q[kwargs["solo"].q_off("FL_KFE")]),
    Constraint(kv.q[kwargs["solo"].q_off("FR_HFE")] - kv.q[kwargs["solo"].q_off("FL_HFE")]),
    Constraint(kv.q[kwargs["solo"].q_off("FR_HAA")] - kv.q[kwargs["solo"].q_off("FL_HAA")]),
    Constraint(kv.q[kwargs["solo"].q_off("HR_KFE")] - kv.q[kwargs["solo"].q_off("HL_KFE")]),
    Constraint(kv.q[kwargs["solo"].q_off("HR_HFE")] - kv.q[kwargs["solo"].q_off("HL_HFE")]),
    Constraint(kv.q[kwargs["solo"].q_off("HR_HAA")] - kv.q[kwargs["solo"].q_off("HL_HAA")]),
]

# HFE joints are rotationally limited to avoid a cable getting twisted. These constraints
# express that:
HFE_Limit_Constraints = lambda kv, **kwargs: [
    Bound(kv.q[kwargs["solo"].q_off("FL_HFE")], lb = -5 * np.pi / 4, ub = np.pi/2),
    Bound(kv.q[kwargs["solo"].q_off("FR_HFE")], lb = -5 * np.pi / 4, ub = np.pi/2),
    Bound(kv.q[kwargs["solo"].q_off("HL_HFE")], lb = -np.pi/2,       ub = 5 * np.pi / 4),
    Bound(kv.q[kwargs["solo"].q_off("HR_HFE")], lb = -np.pi/2,       ub = 5 * np.pi / 4),
]

# -----------------------------
# Subproblem 1: Backflip Launch
# -----------------------------

BackflipLaunch: Task = Task(
    duration = F("0.75"),

    # TODO: By switching these you ask the solver to find
    #       either _any_ feasible solution or one that minimizes torque:

    traj_error = lambda t, kvars: ca.norm_2(kvars.τ),
    # traj_error = lambda t, kvars: 0.0,

    # The hind feet leave the ground 250ms after the front ones.
    # The torso is upside down 200ms after that - these durations
    # have been found through experimentation and looking at backflips.
    # It is possible to optimize for contact times by formulating the
    # optimization problem as an LCP, but that's outside the scope of this code.
    contact_periods = {
        "FR_FOOT": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.3") + frac_ε)]),
        "FL_FOOT": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.3") + frac_ε)]),
        "HR_FOOT": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.55") + frac_ε)]),
        "HL_FOOT": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.55") + frac_ε)]),
    },

    task_constraints = [
        (
            # Balance for the first knot:
            TimePeriod.point(F("0.0")),

            lambda kv, **kwargs: [
                Bound(kv.q[:2]),    # Torso at XY=(0,0)
                Bound(kv.q[3:6]),   # Torso horizontal
                Bound(kv.v),        # Entire robot static
            ]
        ),
        (
            # Flip constraints (final knot):
            TimePeriod.point(F("0.75")),

            lambda kv, **kwargs: [
                # Torso upside down:
                Bound(kv.q[4], lb = -1.0, ub = -1.0),
                
                # Pitching backwards:
                Bound(kv.v[3]),
                Bound(kv.v[4], lb = -ca.inf, ub = 0.0),
                Bound(kv.v[5]),
            ]
        ),
        (
            TimePeriod(start=F("0.0"), end=None),

            lambda kv, **kwargs: [
                # Torso always at least 12cm above the ground:
                Bound(kv.q[2], lb = kwargs["solo"].floor_z + 0.12, ub = ca.inf),

                # Don't allow the hind knees to go below +2cm:
                Constraint(
                    kwargs["fk"](kv.q)["knees"][2:, 2],
                    lb = kwargs["solo"].floor_z + 0.02,
                    ub = ca.inf
                )
            ] +
                # Symmetry and HFE limits:
                LR_Symmetry_Constraints(kv, **kwargs) +
                HFE_Limit_Constraints(kv, **kwargs)
        ),
    ],
)

# -----------------------------
# Subproblem 2: Backflip Landing
# -----------------------------

BackflipLand: Task = Task(
    duration = F("0.6"),

    # NOTE: Same as above:
    traj_error = lambda t, kvars: ca.norm_2(kvars.τ),
    # traj_error = lambda t, kvars: 0.0,

    # Front legs land first, hind 50ms after:
    contact_periods = {
        "FR_FOOT": ivt.IntervalTree([ivt.Interval(F("0.25"), F("0.6") + frac_ε)]),
        "FL_FOOT": ivt.IntervalTree([ivt.Interval(F("0.25"), F("0.6") + frac_ε)]),
        "HR_FOOT": ivt.IntervalTree([ivt.Interval(F("0.3"), F("0.6") + frac_ε)]),
        "HL_FOOT": ivt.IntervalTree([ivt.Interval(F("0.3"), F("0.6") + frac_ε)]),
    },

    task_constraints = [
        (
            # Backflip end constraints:
            TimePeriod.point(F("0.6")),

            lambda kv, **kwargs: [
                Bound(kv.q[3:6]),   # Torso horizontal
                Bound(kv.v),        # Robot static

                # Torso 20cm above the floor:
                Bound(kv.q[2], lb = kwargs["solo"].floor_z + 0.2, ub = ca.inf),
            ]
        ),
        (
            # Landing time constraints (when front legs land):
            TimePeriod.point(F("0.25")),

            lambda kv, **kwargs: [
                # The robot should have almost completed its flip before landing.
                # Otherwise, the front might go under the floor:
                Bound(kv.q[4], lb = -ca.inf, ub = +0.04),

                # Front feet land 14cm in front of the CoM for stability:
                Constraint(kv.f_pos[:2, 0] - kv.q[0], lb = 0.14, ub = ca.inf),
            ]
        ),
        (
            TimePeriod(start=F("0.0"), end=None),
            
            # Torso Z constraints, symmetry and HFE limits:
            lambda kv, **kwargs: [
                Bound(kv.q[2], lb = kwargs["solo"].floor_z + 0.12, ub = ca.inf)
            ] +
                LR_Symmetry_Constraints(kv, **kwargs) +
                HFE_Limit_Constraints(kv, **kwargs)
        ),
    ],
)