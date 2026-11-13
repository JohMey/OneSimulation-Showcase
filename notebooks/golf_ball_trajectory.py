"""OneSimulation example: flight of a spinning golf ball with drag and Magnus lift.
Requires the installed, activated OneSimulation runtime.

The trajectories of a dimpled and a smooth ball are integrated with three of
OneSimulation's time integrators (landing located by an event) and graded against
the exact drag-free parabola, the mechanical-energy balance, a high-accuracy
reference integration and the published carry distances of this 7-iron shot.
"""


# # Golf ball trajectory: why dimples carry further
#
# A golf ball leaves the club face with speed $U_0$, launch angle $\theta_0$ and
# backspin $\omega_0$. In flight it feels gravity, aerodynamic drag $D$ opposite
# to the velocity and Magnus lift $L$ perpendicular to it. The ball is treated
# as a point mass, so the model is a system of ordinary differential equations
# for the position $(x, y)$ and velocity $(\dot x, \dot y)$: a genuinely
# zero-dimensional model whose only independent variable is time. With
# $U=\sqrt{\dot x^2+\dot y^2}$, cross-section $A=\pi d^2/4$ and mass $m$,
#
# $$\ddot x = -\frac{\rho A U}{2m}\,\big(C_D\,\dot x + C_L\,\dot y\big),\qquad
#   \ddot y = \frac{\rho A U}{2m}\,\big(C_L\,\dot x - C_D\,\dot y\big) - g .$$
#
# The coefficients depend on the Reynolds number $Re=\rho U d/\mu$ and the spin
# parameter $S=\omega d/(2U)$; the spin decays as $\omega=\omega_0\,e^{-c\,U t/d}$
# with $c=10^{-4}$ (Smits and Smith, *Science and Golf II*, 2002). For the dimpled
# ball we use the Smits-Smith fit $C_D=0.24+0.18\,S$, $C_L=0.54\,S^{0.4}$, measured
# on spinning golf balls in the range of Bearman and Harvey (*Aeronautical
# Quarterly* 27, 1976). For the smooth ball the drag follows the standard
# smooth-sphere correlation of Morrison (*An Introduction to Fluid Mechanics*,
# 2013), while the lift is taken equal to that of the dimpled ball, as in the
# published comparison of the two balls (the real smooth-ball lift is smaller).
#
# **Numerics.** OneSimulation integrates the four ODEs with the adaptive
# Dormand-Prince Runge-Kutta 5(4) method; the landing $y=0$ (falling) is a
# terminal event located on the cubic Hermite dense output, the apex $\dot y=0$ a
# non-terminal event. For comparison the same flights are flown with the
# fixed-step classical Runge-Kutta method of the shared time-integrator registry
# (landing located by cubic Hermite interpolation) and with variable-step BDF2
# (Newton iterations on a finite-difference Jacobian).
#
# **Reference.** (i) With $C_D=C_L=0$ the trajectory is the exact parabola with
# carry $U_0^2\sin 2\theta_0/g$, flight time $2U_0\sin\theta_0/g$ and apex
# $U_0^2\sin^2\theta_0/(2g)$. (ii) Lift is perpendicular to the velocity and does no
# work, so the loss of kinetic plus potential energy must equal the drag work
# $\int D\,U\,dt$. (iii) Carry and flight time converge to a Dormand-Prince
# reference at relative tolerance $10^{-12}$. (iv) The published result for this
# shot is a carry of about 150 m for the dimpled ball and a smooth ball that flies
# about 25 % shorter.

from importlib.metadata import version

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# OneSimulation figure style: cream paper, dark-green ink, green-sand-copper field colours
CREAM, INK, GREEN, COPPER = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation", ["#214D43", "#5F9175", "#A8BC91", "#E4D6A0", "#CD9561", "#A85539"])
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation_diverging", ["#104F54", "#5E9996", "#F7F2E7", "#D2996E", "#A4472D"])
for _cmap in (FIELD_CMAP, DIVERGING_CMAP):
    if _cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmap)
plt.rcParams.update({
    "figure.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.facecolor": "#FFFAF2",
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": "#9FA593",
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#D5D7C8", "image.cmap": "onesimulation",
    "axes.prop_cycle": cycler(color=[GREEN, COPPER, "#5F9175", "#A85539", "#214D43", "#CD9561", "#8B9A8D"]),
})

from onesim.numerics.discretization.execution.backends.dynamic_analysis import (
    AdaptiveBDF,
    AdaptiveRungeKutta45,
    EventDefinition,
)
from onesim.numerics.time.integration.time_integration import DEFAULT_TIME_INTEGRATOR_REGISTRY

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The 7-iron launch of the published comparison: ball speed 187 km/h, launch
# angle 17.4 degrees and 6113 rpm of backspin. The ball has the regulation
# diameter of 42.67 mm and the maximum regulation mass of 45.93 g; the air is at
# 20 degrees Celsius at sea level.

diameter = 42.67e-3                       # m, d
mass = 45.93e-3                           # kg, m
air_density = 1.204                       # kg/m^3, rho
air_viscosity = 1.825e-5                  # Pa s, mu
gravity = 9.80665                         # m/s^2, g
launch_speed = 187.0 / 3.6                # m/s, U_0
launch_angle = np.radians(17.4)           # rad, theta_0
spin_rate = 6113.0 * 2.0 * np.pi / 60.0   # rad/s, omega_0
spin_decay = 1.0e-4                       # c in omega = omega_0 exp(-c U t / d)
area = np.pi * diameter**2 / 4.0

print(f"launch Reynolds number = {air_density * launch_speed * diameter / air_viscosity:.3e}, "
      f"launch spin parameter S = {spin_rate * diameter / (2 * launch_speed):.3f}")


# ## Geometry and mesh
#
# The ball enters only through its diameter, cross-section and mass: it is a point
# mass in a vertical plane and the model has no spatial mesh and no boundary. The
# ground is the event surface $y=0$. The state vector is
# $(x, y, \dot x, \dot y)$, starting at the origin with the launch velocity.


# ## Model

def dimpled_coefficients(reynolds, spin):
    return 0.24 + 0.18 * spin, 0.54 * spin**0.4


def smooth_sphere_drag(reynolds):
    """Morrison's smooth-sphere drag correlation, valid up to Re = 1e6."""
    return (24.0 / reynolds + 2.6 * (reynolds / 5.0) / (1.0 + (reynolds / 5.0) ** 1.52)
            + 0.411 * (reynolds / 2.63e5) ** -7.94 / (1.0 + (reynolds / 2.63e5) ** -8.0)
            + 0.25 * (reynolds / 1.0e6) / (1.0 + reynolds / 1.0e6))


def smooth_coefficients(reynolds, spin):
    return smooth_sphere_drag(reynolds), 0.54 * spin**0.4


def no_aerodynamics(reynolds, spin):
    return 0.0, 0.0


def flow_state(time, state):
    speed = np.hypot(state[2], state[3])
    omega = spin_rate * np.exp(-spin_decay * speed * time / diameter)
    return speed, air_density * speed * diameter / air_viscosity, omega * diameter / (2.0 * speed)


def make_rhs(coefficients, counter=None):
    def rhs(time, state):
        if counter is not None:
            counter[0] += 1
        speed, reynolds, spin = flow_state(time, state)
        drag, lift = coefficients(reynolds, spin)
        k = air_density * area * speed / (2.0 * mass)
        return np.array([state[2], state[3],
                         -k * (drag * state[2] + lift * state[3]),
                         k * (lift * state[2] - drag * state[3]) - gravity])
    return rhs


launch_state = np.array([0.0, 0.0, launch_speed * np.cos(launch_angle), launch_speed * np.sin(launch_angle)])
landing = EventDefinition("landing", lambda t, y: y[1], direction=-1, terminal=True)
apex = EventDefinition("apex", lambda t, y: y[3], direction=-1)


def fly_adaptive(solver, coefficients, counter=None):
    run = solver.integrate(make_rhs(coefficients, counter), launch_state, 0.0, 30.0,
                           initial_step=1e-3, events=(apex, landing))
    touchdown = next(event for event in run.events if event.event_id == "landing")
    top = next(event for event in run.events if event.event_id == "apex")
    return run, touchdown.time, touchdown.state[0], top.state[1]


def fly_rk4(coefficients, step, counter=None):
    """Fixed-step classical RK4; landing from the cubic Hermite interpolant of the last step."""
    rhs = make_rhs(coefficients, counter)
    rk4 = DEFAULT_TIME_INTEGRATOR_REGISTRY.resolve_id("classical_rk4")
    time, state = 0.0, launch_state.copy()
    while True:
        new = rk4.advance(state, lambda y, t: rhs(t, y), time, step)
        if new[1] < 0.0:
            break
        time, state = time + step, new
    left_rate, right_rate = rhs(time, state), rhs(time + step, new)

    def hermite(fraction):
        h00, h10 = 2 * fraction**3 - 3 * fraction**2 + 1, fraction**3 - 2 * fraction**2 + fraction
        h01, h11 = -2 * fraction**3 + 3 * fraction**2, fraction**3 - fraction**2
        return h00 * state + h10 * step * left_rate + h01 * new + h11 * step * right_rate

    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if hermite(middle)[1] > 0.0 else (low, middle)
    return time + 0.5 * (low + high) * step, hermite(0.5 * (low + high))[0]


# ## Solve
#
# **Production flights** of both balls with Dormand-Prince at a relative
# tolerance of $10^{-10}$ (step cap 10 ms for a smooth dense output).

production = AdaptiveRungeKutta45(relative_tolerance=1e-10, absolute_tolerance=1e-12, maximum_step=0.01)
flights = {}
for name, coefficients in (("dimpled", dimpled_coefficients), ("smooth", smooth_coefficients)):
    flights[name] = fly_adaptive(production, coefficients)

# **Drag-free limit** with all three integrators.
vacuum = {
    "Dormand-Prince RK45": fly_adaptive(AdaptiveRungeKutta45(relative_tolerance=1e-10, absolute_tolerance=1e-12),
                                        no_aerodynamics)[1:3],
    "Classical RK4 (dt = 10 ms)": fly_rk4(no_aerodynamics, 0.01),
    "BDF2 + Newton": fly_adaptive(AdaptiveBDF(order=2, relative_tolerance=1e-8, absolute_tolerance=1e-10),
                                  no_aerodynamics)[1:3],
}
exact_time = 2.0 * launch_speed * np.sin(launch_angle) / gravity
exact_carry = launch_speed**2 * np.sin(2.0 * launch_angle) / gravity
exact_apex = (launch_speed * np.sin(launch_angle)) ** 2 / (2.0 * gravity)
vacuum_apex = fly_adaptive(AdaptiveRungeKutta45(relative_tolerance=1e-10, absolute_tolerance=1e-12),
                           no_aerodynamics)[3]

# **Reference and method comparison** for the dimpled flight: Dormand-Prince at
# $10^{-12}$ is the reference; the error is the deviation of the carry. Work is
# counted in right-hand-side evaluations (for BDF2 including its
# finite-difference Jacobians).
reference = fly_adaptive(AdaptiveRungeKutta45(relative_tolerance=1e-12, absolute_tolerance=1e-14),
                         dimpled_coefficients)
comparison = {"Dormand-Prince RK45 (adaptive)": [], "BDF2 + Newton (adaptive)": [], "Classical RK4 (fixed step)": []}
for tolerance in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
    counter = [0]
    run, flight_time, carry, _ = fly_adaptive(
        AdaptiveRungeKutta45(relative_tolerance=tolerance, absolute_tolerance=tolerance * 1e-2),
        dimpled_coefficients, counter)
    comparison["Dormand-Prince RK45 (adaptive)"].append((tolerance, run.accepted_steps, counter[0],
                                                         abs(carry - reference[2])))
for tolerance in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    counter = [0]
    run, flight_time, carry, _ = fly_adaptive(
        AdaptiveBDF(order=2, relative_tolerance=tolerance, absolute_tolerance=tolerance * 1e-2),
        dimpled_coefficients, counter)
    comparison["BDF2 + Newton (adaptive)"].append((tolerance, run.accepted_steps, counter[0],
                                                   abs(carry - reference[2])))
for step in (0.2, 0.1, 0.05, 0.025, 0.0125):
    counter = [0]
    flight_time, carry = fly_rk4(dimpled_coefficients, step, counter)
    comparison["Classical RK4 (fixed step)"].append((step, int(np.ceil(flight_time / step)), counter[0],
                                                     abs(carry - reference[2])))


# ## Comparison with the reference

print("Drag-free limit vs the exact parabola:")
print(f"  exact: carry {exact_carry:.9f} m, flight time {exact_time:.9f} s, apex {exact_apex:.9f} m")
for method, (flight_time, carry) in vacuum.items():
    print(f"  {method:28s} carry {carry:.9f} m (rel. dev. {abs(carry - exact_carry) / exact_carry:.1e}), "
          f"time {flight_time:.9f} s (rel. dev. {abs(flight_time - exact_time) / exact_time:.1e})")
print(f"  apex from the RK45 apex event: {vacuum_apex:.9f} m (rel. dev. {abs(vacuum_apex - exact_apex) / exact_apex:.1e})")

print("\nMechanical-energy balance along the flights (lift does no work):")
energy_tracks = {}
for name, coefficients in (("dimpled", dimpled_coefficients), ("smooth", smooth_coefficients)):
    run, flight_time, carry, _ = flights[name]
    grid = np.linspace(0.0, flight_time, 4001)
    states = np.array([run.dense_state(t) for t in grid])
    speed = np.hypot(states[:, 2], states[:, 3])
    drag_power = np.array([0.5 * air_density * coefficients(*flow_state(t, y)[1:])[0] * area * flow_state(t, y)[0] ** 3
                           for t, y in zip(grid, states)])
    mechanical = 0.5 * mass * speed**2 + mass * gravity * states[:, 1]
    h = grid[1] - grid[0]
    drag_work = np.concatenate([[0.0], np.cumsum(0.5 * h * (drag_power[1:] + drag_power[:-1]))])
    simpson = h / 3.0 * (drag_power[0] + drag_power[-1] + 4 * drag_power[1:-1:2].sum() + 2 * drag_power[2:-1:2].sum())
    energy_tracks[name] = (grid, mechanical, drag_work)
    loss = mechanical[0] - mechanical[-1]
    print(f"  {name:8s}: energy lost {loss:.6f} J, drag work {simpson:.6f} J, "
          f"rel. dev. {abs(loss - simpson) / loss:.1e}  (launch energy {mechanical[0]:.3f} J)")

print("\nPublished result vs OneSimulation (aerodynamic fits, see text):")
carry_dimpled, carry_smooth = flights["dimpled"][2], flights["smooth"][2]
print(f"  dimpled carry  {carry_dimpled:7.1f} m  vs about 150 m   rel. dev. {abs(carry_dimpled - 150) / 150:.1%}")
shortfall = 1.0 - carry_smooth / carry_dimpled
print(f"  smooth carry   {carry_smooth:7.1f} m, {shortfall:.1%} shorter  vs about 25 % shorter")
for name, (run, flight_time, carry, height) in flights.items():
    print(f"  {name:8s}: flight time {flight_time:.3f} s, apex {height:.2f} m, landing angle "
          f"{np.degrees(np.arctan2(-run.states[-1, 3], run.states[-1, 2])):.1f} deg, "
          f"{run.accepted_steps} RK45 steps")

print(f"\nTime-integrator comparison, dimpled ball (reference carry {reference[2]:.10f} m):")
for method, rows in comparison.items():
    print(f"  {method}")
    for control, steps, calls, error in rows:
        label = f"dt {control * 1e3:6.1f} ms" if "fixed step" in method else f"tol {control:.0e}"
        print(f"    {label:14s} steps {steps:6d}   rhs evaluations {calls:7d}   carry error {error:.2e} m")
rk4_rows = comparison["Classical RK4 (fixed step)"]
print("  observed order of classical RK4 on step halving:",
      " ".join(f"{np.log2(a[3] / b[3]):.2f}" for a, b in zip(rk4_rows[:-1], rk4_rows[1:])))


# ## Figure

fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
ax = axes[0, 0]
for (name, (run, flight_time, carry, _)), colour in zip(flights.items(), (GREEN, COPPER)):
    grid = np.linspace(0.0, flight_time, 800)
    path = np.array([run.dense_state(t) for t in grid])
    ax.plot(path[:, 0], path[:, 1], color=colour, lw=2.0, label=f"{name.capitalize()} ball: carry {carry:.1f} m")
    ax.plot(carry, 0.0, "o", color=colour, ms=7)
x_vacuum = np.linspace(0.0, exact_carry, 400)
ax.plot(x_vacuum, x_vacuum * np.tan(launch_angle) - gravity * x_vacuum**2 / (2 * (launch_speed * np.cos(launch_angle)) ** 2),
        "--", color="#8B9A8D", lw=1.3, label=f"Drag-free parabola (exact): {exact_carry:.1f} m")
ax.axvspan(150 * 0.98, 150 * 1.02, color="#E4D6A0", alpha=0.6, label="Published dimpled carry, about 150 m")
ax.set(xlabel="Downrange distance x / m", ylabel="Height y / m", title="7-iron shot: 187 km/h, 17.4 deg, 6113 rpm",
       xlim=(0, 185), ylim=(0, 48))
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[0, 1]
for (name, (run, flight_time, carry, _)), colour, coefficients in zip(
        flights.items(), (GREEN, COPPER), (dimpled_coefficients, smooth_coefficients)):
    grid = np.linspace(0.0, flight_time, 400)
    values = np.array([coefficients(*flow_state(t, run.dense_state(t))[1:]) for t in grid])
    ax.plot(grid, values[:, 0], color=colour, lw=1.8, label=f"$C_D$ {name}")
    ax.plot(grid, values[:, 1], ":", color=colour, lw=1.8, label=f"$C_L$ {name}")
ax.set(xlabel="Time / s", ylabel="Coefficient", title="Drag and lift coefficients along the flight", ylim=(0.26, 0.5))
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(alpha=0.3)

ax = axes[1, 0]
for (name, (grid, mechanical, drag_work)), colour in zip(energy_tracks.items(), (GREEN, COPPER)):
    ax.plot(grid, mechanical, color=colour, lw=1.8, label=f"Kinetic + potential, {name}")
    ax.plot(grid, mechanical + drag_work, "--", color=colour, lw=1.4, label=f"... + drag work, {name}")
ax.set(xlabel="Time / s", ylabel="Energy / J", title="Energy balance: lift does no work")
ax.legend(loc="center right", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1, 1]
for (method, rows), marker, colour in zip(comparison.items(), ("o", "s", "^"), (GREEN, COPPER, "#8B9A8D")):
    rows = np.array([(r[2], r[3]) for r in rows])
    ax.loglog(rows[:, 0], rows[:, 1], marker + "-", color=colour, lw=1.5, ms=6, label=method)
ax.set(xlabel="Right-hand-side evaluations", ylabel="Carry error / m", title="Work-precision of three time integrators")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3, which="both")
plt.show()


# ## Going further
#
# Lower the launch angle with the parameterised spin and look for the optimum:
# with constant spin the best launch is near 20 degrees, while spin that drops
# with launch angle moves it towards 14 degrees. Replace the Smits-Smith fit by
# tabulated wind-tunnel coefficients, or add a cross-wind and a side-spin
# component to turn the planar model into a 3D hook-and-slice study (six ODEs, the
# same integrators and events). The work-precision panel is the typical picture
# for a smooth, non-stiff ODE: both explicit Runge-Kutta methods beat the implicit
# BDF2 route by two to three orders of magnitude in work. Fixed-step RK4 is fully
# competitive here because the flight is short and smooth; at loose tolerances
# the adaptive method takes so few, long steps that locating the landing on the
# cubic interpolant dominates its error, while at tight tolerances its step
# control pays off. BDF2 locates the event on a linear interpolant, which caps its
# carry accuracy near a few micrometres.
