from tasks import *

from constraints import *
from configurations import *
from robot import Bolt

from utilities import frac_ε

F = Fraction

LR_Symmetry_Constraints = lambda kv, **kwargs: [
    Constraint(kv.q[kwargs["robot"].q_off("FR_KFE")] - kv.q[kwargs["robot"].q_off("FL_KFE")]),
    Constraint(kv.q[kwargs["robot"].q_off("FR_HFE")] - kv.q[kwargs["robot"].q_off("FL_HFE")]),
    Constraint(kv.q[kwargs["robot"].q_off("FR_HAA")] + kv.q[kwargs["robot"].q_off("FL_HAA")]),
]

UprightTask: Task = Task(
    robot_type = Bolt,
    duration = F("0.8"),
    traj_error = lambda t, kvars: 0., #ca.norm_2(kvars.τ),

    # Feet always in contact:
    contact_periods = {
        # NOTE: We add ε at the end of the intervals as .overlaps() is
        # not inclusive of the end time:
        foot: ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.8") + frac_ε)])
        for foot in ["FR_FOOT", "FL_FOOT"]
    },

    task_constraints = [
        (
            TimePeriod.point(F("0.0")),

            lambda kv, **kwargs: [
                # All joints in the folded configuration:
                Constraint(kv.q[3:] - create_state_vector(kwargs["robot"].robot, BOLT_UPRIGHT_JOINT_MAP)[3:]),

                # Torso XY at (0, 0).
                Bound(kv.q[:2]),

                # Torso and joints static at the beginning:
                Bound(kv.v),
            ]
        ),
        (
            TimePeriod.point(F("0.8")),

            lambda kv, **kwargs: [
                Constraint(kv.q[3:] - create_state_vector(kwargs["robot"].robot, BOLT_UPRIGHT_JOINT_MAP)[3:]),
                Bound(kv.q[:2]),
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod(F("0.0"), end = None),
            lambda kv, **kwargs: [
                
            ] + LR_Symmetry_Constraints(kv, **kwargs)
        )
    ],
)

Squat: Task = Task(
    robot_type = Bolt,
    duration = F("0.8"),
    # traj_error = lambda t, kvars: 0.001 * ca.norm_2(kvars.v[6:])**2 + 100*kvars.q[4]**2,
    traj_error = lambda t, kvars: kvars.q[4]**2,

    # Feet always in contact:
    contact_periods = {
        # NOTE: We add ε at the end of the intervals as .overlaps() is
        # not inclusive of the end time:
        foot: ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.8") + frac_ε)])
        for foot in ["FR_FOOT", "FL_FOOT"]
    },

    task_constraints = [
        (
            TimePeriod.point(F("0.0")),

            lambda kv, **kwargs: [
                # # All joints in the folded configuration:
                Constraint(kv.q[6:] - create_state_vector(kwargs["robot"].robot, BOLT_SITTING_JOINT_MAP)[6:]),
                Bound(kv.q[:2]),

                Bound(kv.q[3]),
                Constraint(kv.q[4] + 0.03),
                Bound(kv.q[5]),

                # Bound(kv.q[3]),
                # Constraint(kv.q[4]),
                # Bound(kv.q[5]),

                # Bound(kv.v),
            ]
        ),
        (
            TimePeriod.point(F("0.8")),

            lambda kv, **kwargs: [
                Constraint(kv.q[3:] - create_state_vector(kwargs["robot"].robot, BOLT_SITTING_JOINT_MAP)[3:]),
                Bound(kv.v)
            ]
        ),
        (
            TimePeriod(F("0.0"), end = None),
            lambda kv, **kwargs: [
                Bound(kv.q[kwargs["robot"].q_off("FR_HFE")], -np.pi/2, np.pi/2),
                Bound(kv.q[kwargs["robot"].q_off("FL_HFE")], -np.pi/2, np.pi/2)
            ] + LR_Symmetry_Constraints(kv, **kwargs),
        )
    ],
)

# Find a stable configuration for the robot to balance that is as close
# as possible to the sitting one:
Stable: Task = Task(
    robot_type = Bolt,
    duration = F("0.4"),

    # ||q_joints - BOLT_SITTING_JOINT_MAP||_2
    traj_error = lambda t, kvars: \
        ca.norm_2(kvars.q[6:] - np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 0, np.pi/4, -np.pi/2, 0, np.pi/4, -np.pi/2]), axis=0).T[6:])**2,

    # Feet always in contact:
    contact_periods = {
        # NOTE: We add ε at the end of the intervals as .overlaps() is
        # not inclusive of the end time:
        foot: ivt.IntervalTree([ivt.Interval(F("0.0"), F("0.4") + frac_ε)])
        for foot in ["FR_FOOT", "FL_FOOT"]
    },

    task_constraints = [
        (
            TimePeriod(F("0.0"), end = None),
            lambda kv, **kwargs: [
                # XY = 0:
                Bound(kv.q[:2]),

                # Orientation = 0:
                Bound(kv.q[3:6]),
                
                # No movement along the entire trajectory:
                Bound(kv.v),
            ] + LR_Symmetry_Constraints(kv, **kwargs),
        )
    ],
)
