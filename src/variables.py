from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any, Optional

from utilities import flatten_mats, unflatten_mats

T = TypeVar('T')

@dataclass
class KnotVars(Generic[T]):
    # Helper struct to hold all variables for a single knot:
    q: T; v: T; a: T; τ: T; λ: T; f_pos: T

@dataclass(frozen = True)
# Struct that holds information about a transcribed subproblem.
# We'll store this alongside the global solution vector when the
# optimizer completes. This way we can load, visualize and execute a
# solution without needing to transcribe exactly the same problem again. 
class TranscriptionInfo:
    subproblem_name: str
    
    # The number of decision variables per knot should be constant and
    # not depend on the task being solved (excl. slack variables):
    n_knots: int

    # Discretization Δt, for trajectory execution and visualization:
    dt: float

    # Information about the robot's state and action space:
    nq:         int
    nv:         int
    nτ:         int
    feet_count: int

    # Slack variable count (at the end of the variable vector):
    slack_var_count: int

    # Other auxiliary info:
    description: Optional[str] = None

@dataclass
class CollocationVars(Generic[T]):
    # These variables correspond to the transcription of a single subproblem.
    # The entire problem may need to stitch two or more subproblems that are
    # solved at the same time.
    n_knots:  int

    q_k:        list[T] = field(default_factory=list)  # 18x1
    v_k:        list[T] = field(default_factory=list)  # 18x1
    a_k:        list[T] = field(default_factory=list)  # 18x1
    τ_k:        list[T] = field(default_factory=list)  # 12x1
    λ_k:        list[T] = field(default_factory=list)  # 4x3
    f_pos_k:    list[T] = field(default_factory=list)  # 4x3

    slack_vars: list[T] = field(default_factory=list)  # Subproblem-dependent
    
    # Time duration of each knot (Δt). Useful when stitching trajectories
    # discretized at different frequencies:
    knot_duration: list[float] = field(default_factory=list, init=False)

    # Flatten variable set into a column vector. Slack variables
    # are appended at the end:
    def flatten(self) -> T:
        return flatten_mats(
            self.q_k + self.v_k + self.a_k + \
            self.τ_k + self.λ_k + self.f_pos_k + \
            self.slack_vars
        )

    # This appends a single knot's variables to the lists.
    # NOTE: Slack variables need to be set separately.
    def append_knot(self, kvars: KnotVars, duration: float) -> None:
        if len(self.q_k) >= self.n_knots:
            raise RuntimeError("Cannot add more variables than the predefined knot count!")

        self.knot_duration.append(duration)

        self.q_k.append(kvars.q); self.v_k.append(kvars.v)
        self.a_k.append(kvars.a); self.τ_k.append(kvars.τ)
        self.λ_k.append(kvars.λ); self.f_pos_k.append(kvars.f_pos)

    # Return variable set for a single knot (excl. slack):
    def get_vars_at_knot(self, k: int) -> KnotVars[T]:
        if k >= len(self.q_k):
            raise ValueError(f"Invalid knot index: {k}")

        return KnotVars(
            self.q_k[k], self.v_k[k], self.a_k[k],
            self.τ_k[k], self.λ_k[k], self.f_pos_k[k]
        )
    
    @staticmethod
    # Interpolate variable set by a simple repetition of knots:
    def interpolate(self, target_knots: int):
        raise NotImplementedError
    
    @staticmethod
    # Load trajectory and slack variables from a flattened column vector.
    # We assume that all knots are of equal time duration.
    # Return the variables as well as the total variable count:
    # n_knots: int, slack_var_count: int, duration: float
    def unflatten(info: TranscriptionInfo, vec: T) -> tuple[Any, int]:
        assert vec.shape[1] == 1

        dvars = CollocationVars[T](n_knots = info.n_knots)
        dvars.knot_duration = [info.dt] * info.n_knots

        # Unflatten trajectory variables:
        o, sz = 0, info.nq
        dvars.q_k = unflatten_mats(vec[o : o + info.n_knots * sz], (sz, 1))

        o, sz = o + sz * info.n_knots, info.nv
        dvars.v_k = unflatten_mats(vec[o : o + info.n_knots * sz], (sz, 1))

        o, sz = o + sz * info.n_knots, info.nv
        dvars.a_k = unflatten_mats(vec[o : o + info.n_knots * sz], (sz, 1))

        o, sz = o + sz * info.n_knots, info.nτ
        dvars.τ_k = unflatten_mats(vec[o : o + info.n_knots * sz], (sz, 1))

        o, sz = o + sz * info.n_knots, info.feet_count * 3
        dvars.λ_k = unflatten_mats(vec[o : o + info.n_knots * sz], (info.feet_count, 3))

        o, sz = o + sz * info.n_knots, info.feet_count * 3
        dvars.f_pos_k = unflatten_mats(vec[o : o + info.n_knots * sz], (info.feet_count, 3))

        # Load slack variables as a single column vector:
        o += sz * info.n_knots
        dvars.slack_vars = [ vec[o : o + info.slack_var_count] ]

        o += info.slack_var_count
        return dvars, o
