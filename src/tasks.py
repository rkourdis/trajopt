from dataclasses import dataclass, field
from typing import Callable, Optional

from fractions import Fraction
F = Fraction

import casadi as ca
import intervaltree as ivt

from constraints import *
from configurations import *
from utilities import frac_ε
from variables import KnotVars, CollocationVars
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

    # Periods of ground contact for each foot. Omit for contact schedule discovery:
    contact_periods: Optional[dict[str, ivt.IntervalTree]] = None

    # Instantaneous trajectory error to minimise. Given t, q, v, a, τ, λ at a collocation
    # point returns how far away the trajectory is from the desired one at that time:
    traj_error: Callable[[float, KnotVars[ca.SX]], ca.SX] = lambda t, kv: 0.0

    # List of times at which a task-specific constraint must hold.
    # For each one, a "factory-like" callable is held which returns
    # constraint objects given the corresponding knot's decision variables.
    # Additional parameters are passed as kwargs.
    per_knot_constraints: list[
        tuple[
            TimePeriod,
            Callable[[KnotVars[ca.SX], dict], list[ConstraintType]]
        ]
    ] = field(default_factory=list)

    # Called once with the complete CollocationVars[ca.SX] for this task.
    # Can couple endpoints, arbitrary knots, or sums over the trajectory.
    trajectory_constraints: list[
        Callable[[CollocationVars, dict], list[ConstraintType]]
    ] = field(default_factory=list)

ImplicitJumpTask: Task = Task(
    duration = F("1.2"),
    traj_error = lambda t, kv: ca.norm_2(kv.τ),
    per_knot_constraints = [
        (
            TimePeriod.point(F("0.0")),

            lambda kv, **kwargs: [
                Constraint(kv.q - load_robot_pose(Pose.STANDING, kwargs["go2"].floor_z)[0]),
                Bound(kv.v),
                Bound(kv.a),
            ],
        ),

        (
            TimePeriod.point(F("0.6")),
            lambda kv, **kwargs: [
                Bound(kv.q[2], lb = kwargs["go2"].floor_z + 0.7, ub = ca.inf),
                Bound(kv.v[6:]),
            ],
        ),
        (
            TimePeriod.point(F("1.2")),

            lambda kv, **kwargs: [
                Constraint(kv.q - load_robot_pose(Pose.STANDING, kwargs["go2"].floor_z)[0]),
                Bound(kv.v),
                Bound(kv.a),
            ],
        )
    ]
)

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
        for foot in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    },

    per_knot_constraints = [
        (
            TimePeriod.point(F("0.0")),

            lambda kv, **kwargs: [
                # All joints in the folded configuration:
                Constraint(kv.q[3:] - create_state_vector(kwargs["go2"].robot, FOLDED_JOINT_MAP)[3:]),

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
                # End up static in the standing configuration at (0, 0):
                Constraint(kv.q[3:] - load_robot_pose(Pose.STANDING, kwargs["go2"].floor_z)[0][3:]),
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

        for foot in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    },

    per_knot_constraints = [
        (
            TimePeriod.point(F("0.0")),

            # We constrain the entire initial state here (including torso Z),
            # despite what we did above.
            # Eh, it still works ¯\_(ツ)_/¯
            lambda kv, **kwargs: [
                Constraint(kv.q - load_robot_pose(Pose.STANDING, kwargs["go2"].floor_z)[0]),
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod.point(F("1.2")),

            # Ending state same as initial:
            lambda kv, **kwargs: [
                Constraint(kv.q - load_robot_pose(Pose.STANDING, kwargs["go2"].floor_z)[0]),
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod(start = F("0.0"), end = None),

            # Keep the robot torso above the ground by 18cm, always. This is to
            # avoid the landing being a bit _too_ low!
            lambda kv, **kwargs: [
                Bound(kv.q[2], lb = kwargs["go2"].floor_z + 0.18, ub = ca.inf),
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

    # 300ms jump, front legs leave the ground first:
    contact_periods = {
        "FR_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.3") + frac_ε), ivt.Interval(F("0.6"), F("1.0") + frac_ε)]),
        "FL_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.3") + frac_ε), ivt.Interval(F("0.6"), F("1.0") + frac_ε)]),
        "RR_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.35") + frac_ε), ivt.Interval(F("0.65"), F("1.0") + frac_ε)]),
        "RL_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.35") + frac_ε), ivt.Interval(F("0.65"), F("1.0") + frac_ε)]),
    },

    per_knot_constraints = [
        (
            TimePeriod.point(F("0.0")), 

            # Feet in static Go2 standing configuration at the beginning:
            lambda kv, **kwargs: [
                Constraint(kv.q[:2] - load_robot_pose(Pose.STANDING, kwargs["go2"].floor_z)[0][:2]),
                Constraint(kv.q[3:] - load_robot_pose(Pose.STANDING, kwargs["go2"].floor_z)[0][3:]),
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod.point(F("1.0")),

            lambda kv, **kwargs: [
                # Torso has moved forward by at least 40cm:
                Bound(kv.q[0], 0.4, ca.inf),
            
                # Torso is horizontal above the ground at a height of at least 20cm:
                Bound(kv.q[2], kwargs["go2"].floor_z + 0.2, ca.inf),
                Bound(kv.q[3:6]),

                # Robot is static:
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod(start = F("0.0"), end = None),

            # Keep torso above the floor by 18cm, always.
            # Also, keep knees above the floor by 2cm.
            # Disallow roll or yaw.
            lambda kv, **kwargs: [
                Bound(kv.q[2], lb = kwargs["go2"].floor_z + 0.18, ub = ca.inf),
                Constraint(kwargs["fk"](kv.q)["knees"][:, 2], lb = kwargs["go2"].floor_z + 0.02, ub = ca.inf),
                Bound(kv.v[3]),
                Bound(kv.v[5]),
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
        "FR_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.35") + frac_ε), ivt.Interval(F("0.65"), F("1.0") + frac_ε)]),
        "FL_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.35") + frac_ε), ivt.Interval(F("0.65"), F("1.0") + frac_ε)]),
        "RR_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.3") + frac_ε), ivt.Interval(F("0.6"), F("1.0") + frac_ε)]),
        "RL_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.3") + frac_ε), ivt.Interval(F("0.6"), F("1.0") + frac_ε)]),
    },

    per_knot_constraints = [
        (
            # We don't add initial state constraints, continuity with `JumpTaskFwd`
            # will be enforced:
            TimePeriod.point(F("1.0")),
            
            lambda kv, **kwargs: [
                # Robot has gone back to original configuration:
                Constraint(kv.q[:2] - load_robot_pose(Pose.STANDING, kwargs["go2"].floor_z)[0][:2]),
                Constraint(kv.q[3:] - load_robot_pose(Pose.STANDING, kwargs["go2"].floor_z)[0][3:]),
                
                # Robot is static:
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod(start = F("0.0"), end = None),

            lambda kv, **kwargs: [
                Bound(kv.q[2], lb = kwargs["go2"].floor_z + 0.18, ub = ca.inf),
                Constraint(kwargs["fk"](kv.q)["knees"][:, 2], lb = kwargs["go2"].floor_z + 0.02, ub = ca.inf),
                Bound(kv.v[3]),
                Bound(kv.v[5]),
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
    Constraint(kv.q[kwargs["go2"].q_off("FR_calf_joint")] - kv.q[kwargs["go2"].q_off("FL_calf_joint")]),
    Constraint(kv.q[kwargs["go2"].q_off("FR_thigh_joint")] - kv.q[kwargs["go2"].q_off("FL_thigh_joint")]),
    Constraint(kv.q[kwargs["go2"].q_off("FR_hip_joint")] + kv.q[kwargs["go2"].q_off("FL_hip_joint")]),
    Constraint(kv.q[kwargs["go2"].q_off("RR_calf_joint")] - kv.q[kwargs["go2"].q_off("RL_calf_joint")]),
    Constraint(kv.q[kwargs["go2"].q_off("RR_thigh_joint")] - kv.q[kwargs["go2"].q_off("RL_thigh_joint")]),
    Constraint(kv.q[kwargs["go2"].q_off("RR_hip_joint")] + kv.q[kwargs["go2"].q_off("RL_hip_joint")]),
]

# Joint position and speed limits are enforced globally from the Go2 URDF.

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
    contact_periods = {
        "FR_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.3") + frac_ε)]),
        "FL_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.3") + frac_ε)]),
        "RR_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.55") + frac_ε)]),
        "RL_foot": ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.55") + frac_ε)]),
    },

    per_knot_constraints = [
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
                Bound(kv.q[2], lb = kwargs["go2"].floor_z + 0.12, ub = ca.inf),

                # Don't allow the hind knees to go below +2cm:
                Constraint(
                    kwargs["fk"](kv.q)["knees"][2:, 2],
                    lb = kwargs["go2"].floor_z + 0.02,
                    ub = ca.inf
                )
            ] +
                # Left-right symmetry:
                LR_Symmetry_Constraints(kv, **kwargs)
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
        "FR_foot": ivt.IntervalTree([ivt.Interval(F("0.25"), F("0.6") + frac_ε)]),
        "FL_foot": ivt.IntervalTree([ivt.Interval(F("0.25"), F("0.6") + frac_ε)]),
        "RR_foot": ivt.IntervalTree([ivt.Interval(F("0.3"), F("0.6") + frac_ε)]),
        "RL_foot": ivt.IntervalTree([ivt.Interval(F("0.3"), F("0.6") + frac_ε)]),
    },

    per_knot_constraints = [
        (
            # Backflip end constraints:
            TimePeriod.point(F("0.6")),

            lambda kv, **kwargs: [
                Bound(kv.q[3:6]),   # Torso horizontal
                Bound(kv.v),        # Robot static

                # Torso 20cm above the floor:
                Bound(kv.q[2], lb = kwargs["go2"].floor_z + 0.2, ub = ca.inf),
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
            
            # Torso height and left-right symmetry:
            lambda kv, **kwargs: [
                Bound(kv.q[2], lb = kwargs["go2"].floor_z + 0.12, ub = ca.inf)
            ] +
                LR_Symmetry_Constraints(kv, **kwargs)
        ),
    ],
)
