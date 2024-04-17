import scipy
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from dataclasses import dataclass

import scipy.integrate

@dataclass
class State:
    # Second order dynamics:
    #    x_dd = f(t, x, x_d, u)
    #
    # x_d tells you how x evolves
    # x_dd tells you how x_d evolves, is dependent on x, x_d, u
    t: float
    x: float
    x_d: float

@dataclass
class Trajectory:
    path: list[State]

@dataclass
class Input:
    t: float
    u: float

# Evaluates f(t, x, x_d, u):
def dynamics(t: float, q: State, u: float) -> float:
    return u

# These integrate the dynamics from `initial_state` assuming input
# provided by input(t) above. The trajectory returned is sampled
# at specific time points.
#
# Information flows from the top derivatives down:
# x_d(t) = Integrate[x_dd(t), {t, t_start, t_end}]
# (which finds x_d at ONE t by integrating ALL x_dd before)
#   and
# x(t) = Integrate[x_d(t), {t, t_start, t_end}]

# Note: ODEs and IVPs were difficult because you basically had the derivative
# wrt the state, but you were lookimg for a CLOSED FORM solution.
# Here, you can just numerically find what the function values are at specific points!

# IVP: Given the dynamics, the initial FULL state and all inputs to the system,
#        find how the system evolves
# BVP: Given the dynamics, the initial PARTIAL state, the end PARTIAL state, and all inputs
#    to the system, find the ONLY possible evolution of the system with the provided inputs
#    that starts and ends at those partial states.
#
#       Note: The inputs to the system basically define A NEW "system" that will evolve on its own.

def simulate_definite_integrator(initial_state: State, end_time: float, trapezoidal: bool = False) -> Trajectory:
    # This calls the integrator once for the velocities at sample times.
    # Then, given those velocity samples, it assumes zero-order hold and calls
    # the integrator again for the positions.
    t_0 = initial_state.t
    samples_per_sec = 100

    sample_count = (end_time - t_0) * 100   # 100 samples / sec
    sample_times = np.linspace(t_0, end_time, sample_count)

    velocities = [
        initial_state.x_d + scipy.integrate.quad(input, t_0, t)[0]
        for t in sample_times
    ]

    if not trapezoidal:
        # ======================= Alternative 1 ============================
        # Zero order hold so that the integrator can ask for the velocity
        # at any point in time:
        zoh = lambda t: velocities[int((t - t_0) * samples_per_sec)]

        # Integrate velocities to obtain positions:
        positions = [
            initial_state.x + scipy.integrate.quad(zoh, t_0, t)[0]
            for t in sample_times
        ]
        # ===================================================================
    else:
        # Alternative 2: Trapezoidal integration. Basically assumes the value
        # to sum is the middle of the two values left and right for each integral.
        # Zero order hold assumes the left, will undershoot.
        positions = [
            initial_state.x + scipy.integrate.trapz(velocities[:idx], sample_times[:idx])
            for idx in range(len(sample_times))
        ]

    return Trajectory(
        path = [
           State(t, x, v) for t, x, v in
           zip(sample_times, positions, velocities)
        ]
    )

def simulate_rk45(initial_state: State, end_time: float) -> Trajectory:
    # The problem is: d[x, x_d] / dt = [x_d, input(t)]
    # Given a state [x, x_d] at time t, you know the FULL derivative
    # of the state at t. That means you can get the state next to it,
    # and propagate the information. That's integration of the IVP :)
    rhs = lambda t, y: [y[1], input(t)]
    soln = scipy.integrate.solve_ivp(
        fun = rhs,
        t_span =[initial_state.t, end_time],
        y0 = [initial_state.x, initial_state.x_d]
    )

    return Trajectory(path = [
        State(t, x, x_d)
        for t, x, x_d in zip(soln.t, soln.y[0], soln.y[1])
    ])

def compare_integrators():
    # Compare the naive and RK45 integrators using a test input:
    def input(t) -> float:
        return 0 if int(t) % 2 == 0 else 1
    
    t_start, t_end = 0, 10
    freq = 100

    # Sample inputs to plot:
    N = (t_end - t_start) * freq
    times = np.linspace(t_start, t_end, N)
    initial_state = State(0, 0, 0)
    
    inputs = [input(t) for t in times]
    plt.plot(times, inputs, label="u(t)")

    traj_naive = simulate_definite_integrator(initial_state, 10, False).path
    traj_naive_trapez = simulate_definite_integrator(initial_state, 10, True).path
    traj_rk45 = simulate_rk45(initial_state, 10).path

    plt.plot(
        [t.t for t in traj_naive],
        [t.x for t in traj_naive],
        label = "position (double integration, zoh)"
    )

    plt.plot(
        [t.t for t in traj_naive_trapez],
        [t.x for t in traj_naive_trapez],
        label = "position (double integration, trapezoidal)"
    )

    plt.plot(
        [t.t for t in traj_naive],
        [t.x_d for t in traj_naive],
        label = "velocity (double integration)"
    )

    plt.plot(
        [t.t for t in traj_rk45],
        [t.x for t in traj_rk45],
        label = "position (RK45)"
    )

    plt.plot(
        [t.t for t in traj_rk45],
        [t.x_d for t in traj_rk45],
        label = "velocity (RK45)"
    )

    plt.legend()
    plt.show()

# This will run trajectory optimization starting from the initial_state
# and returns the input to the system that achieves the desired end state,
# as well as the entire system trajectory with that input:
#
# We have FULL information about the start and end states and information about the "unforced"
# system, and we're looking for the forcing PART of the system! Essentially, part of the
# ENTIRE SYSTEM is missing (the entire system would essentially evolve with the FULL equations)
# and we're looking for that part! There may be MANY ways of forcing the system so that the 
# initial and end state constraints are met! That depends on how the "forcing" defines the full
# (forced) system, ie, how it will impact state evolution! That's what the _input_ Jacobian is trying
# to capture (d (dynamics) / d (input)) (I believe).
def optimize(initial_state: State, desired: State) -> tuple[list[Input], Trajectory]:
    freq = 50
    delta_t = 1 / freq
    N = (desired.t - initial_state.t) * freq + 1    # Add extra point for the end time

    x_k, xd_k, u_k = [], [], []     # Collocation point variables
    g_i = []                          # Equality constraints

    for idx in range(N):
        # Create decision variables at collocation points:
        x_k.append(ca.MX.sym(f"x_{idx}"))
        xd_k.append(ca.MX.sym(f"xd_{idx}"))
        u_k.append(ca.MX.sym(f"u_{idx}"))

        # We'll add constraints for current and previous points
        if idx == 0:
            continue    

        # Create dynamics constraint for velocities (using trapezoidal integration):
        g_i.append(xd_k[idx] - xd_k[idx-1] - 0.5 * delta_t * (u_k[idx] + u_k[idx-1]))

        # Same for positions:
        g_i.append(x_k[idx] - x_k[idx-1] - 0.5 * delta_t * (xd_k[idx] + xd_k[idx-1]))

    # Create optimization objective (minimum input**2):
    obj = sum(0.5 * delta_t * (u_k[idx]**2 + u_k[idx+1]**2) for idx in range(N-1))

    # Add equality constraint for trajectory boundaries.
    # CasADi bounds the g vector between the provided limits, and we'll use
    # that to force it to be zero.
    g_i.append(x_k[0] - initial_state.x)
    g_i.append(x_k[-1] - desired.x)
    g_i.append(xd_k[0] - initial_state.x_d)
    g_i.append(xd_k[-1] - desired.x_d)

    # Ask the thing to solve the problem:
    nlp = {
        "x": ca.vertcat(*x_k, *xd_k, *u_k),      # Minimize over all decision variables
        "f": obj,
        "g": ca.vertcat(*g_i)                     # Given all constraints
    }

    solver = ca.nlpsol("S", "ipopt", nlp)

    # Construct initial guess for all decision variables:
    x0 = []

    const_velocity = (desired.x - initial_state.x) / (desired.t - initial_state.t)
    x0 += list(const_velocity * delta_t * idx for idx in range(N)) # We assume a constant velocity during the trajectory
    x0 += list(const_velocity for _ in range(N))
    x0 += list(0 for _ in range(N))                                # No acceleration

    soln = solver(x0 = x0, lbg = 0, ubg = 0)    # All g constraints are equality constraints
    
    traj = Trajectory(path = [
        State(t = idx * delta_t, x = float(soln["x"][idx]), x_d = float(soln["x"][N + idx]))
        for idx in range(N)
    ])

    inputs = [
        Input(t = idx * delta_t, u = float(soln["x"][2 * N + idx]))
        for idx in range(N)
    ]    
    return (inputs, traj)

if __name__ == "__main__":
    inputs, traj = optimize(State(0, 0, 0), State(1, 1, 0))

    t, x, v = zip(*((s.t, s.x, s.x_d) for s in traj.path))
    plt.plot(t, x, label = "x(t)")
    plt.plot(t, v, label = "v(t)")

    # t_i, u_u = zip(*((inp.t, inp.u) for inp in inputs))
    # plt.plot(t_i, u_u, label = "u(t)")

    plt.legend()
    plt.show()