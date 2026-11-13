"""OneSimulation example: longitudinal resonances of a rigid-walled tube.
Requires the installed, activated OneSimulation runtime.

The computed standing waves are checked against the closed-form resonance
frequencies and mode shapes of a one-dimensional acoustic cavity.
"""


# # Resonances of a rigid-walled tube
#
# A straight tube of length $L$ is filled with a gas of density $\rho$ and sound
# speed $c$. Both ends are **rigid walls**, so the acoustic velocity vanishes
# there and the pressure forms an antinode. The linear acoustic equations are
# solved in one dimension:
#
# $$\rho\,\frac{\partial u}{\partial t}=-\frac{\partial p}{\partial x},\qquad
#   \frac{1}{\rho c^{2}}\,\frac{\partial p}{\partial t}=-\frac{\partial u}{\partial x}.$$
#
# Combining them gives the wave equation $p_{tt}=c^{2}p_{xx}$. With rigid ends the
# admissible standing waves are
#
# $$p_n(x,t)=\cos\!\Big(\frac{n\pi x}{L}\Big)\,\cos(\omega_n t),\qquad
#   \omega_n=\frac{n\pi c}{L},\qquad
#   f_n=\frac{\omega_n}{2\pi}=\frac{n\,c}{2L}.$$
#
# **Reference.** We initialise the pressure with a single mode shape
# $\cos(n\pi x/L)$ at rest and let OneSimulation integrate the dynamics. Two
# independent closed-form checks follow: the spatio-temporal field must track
# $\cos(n\pi x/L)\cos(\omega_n t)$ over several periods, and the oscillation
# frequency read back from the signal must equal $f_n=n c/(2L)$.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import BoundaryFluxDifference, CentralDifference, FiniteDifference
from onesim.numerics.fvm import Rusanov
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.acoustics import LinearAcoustics
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.simulation.problem_definition.boundaries import ReflectingWall

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The values are freely adjustable. $L$ and $c$ fix the fundamental
# $f_1=c/(2L)$; the mesh and time step are chosen fine enough that the discrete
# dispersion error stays well below one percent. We follow the first two modes.

L = 0.50              # m, tube length
rho = 1.2             # kg/m^3, air density
c = 343.0             # m/s, speed of sound
p_amplitude = 1.0     # Pa, initial modal amplitude
modes = (1, 2)        # longitudinal mode numbers to resolve
cells = 120           # finite-difference cells along the tube
periods = 3           # how many fundamental periods of each mode to integrate
steps_per_period = 200

geometry = Line(start=(0.0, 0.0, 0.0), end=(L, 0.0, 0.0))
mesh = StructuredLineMesher(cells=cells).generate(geometry)
x = np.asarray(mesh.arclength_centers)
print(f"Tube length L = {L} m,  c = {c} m/s,  f_1 = c/(2L) = {c / (2 * L):.1f} Hz")
print(f"Mesh: {cells} cells,  dx = {L / cells * 1000:.2f} mm")


# ## Driving each mode and comparing with the closed form
#
# For every mode number $n$ the pressure is initialised to
# $p_0\cos(n\pi x/L)$ with the gas at rest. A central finite-difference
# discretisation is advanced with a classical fourth-order Runge-Kutta step.
# Both tube ends carry an explicit rigid-wall (zero normal velocity) boundary
# condition.

def analytic_pressure(n, t):
    omega = n * np.pi * c / L
    return p_amplitude * np.cos(n * np.pi * x / L) * np.cos(omega * t)


results = {}
print("Mode   f_n = n c/2L   measured f   rel. error   max field error")
for n in modes:
    omega = n * np.pi * c / L
    period = 2.0 * np.pi / omega
    f_expected = n * c / (2.0 * L)

    def mode_shape(x, y, z):
        return p_amplitude * np.cos(n * np.pi * x / L)

    model = LinearAcoustics(domain=geometry.line, density=rho, sound_speed=c)
    problem = Problem(
        name=f"Rigid tube, mode {n}",
        geometry=geometry,
        physics=(model,),
        initial_conditions=(
            InitialValue(model.pressure, mode_shape),
            InitialValue(model.velocity, 0.0),
        ),
        # Both ends of the line are rigid walls (zero normal velocity).
        boundary_conditions=(
            ReflectingWall(physics=model, boundary=geometry.boundaries.all),
        ),
    )
    simulation = Simulation(
        problem=problem,
        mesh=mesh,
        numerics=Numerics(
            spatial=FiniteDifference(
                gradient=BoundaryFluxDifference(derivative=CentralDifference(), boundary_flux=Rusanov())
            ),
            time_integrator=ClassicalRungeKutta4(),
        ),
    )
    simulation.validate().require_valid()
    result = simulation.run(start_time=0.0, end_time=periods * period, time_step=period / steps_per_period)

    field = result.field(model.pressure)
    times = np.asarray([float(t) for t in result.times])
    pressure_history = np.stack([np.asarray(field.at(float(t)).values) for t in times])
    analytic_history = np.stack([analytic_pressure(n, t) for t in times])
    max_field_error = float(np.max(np.abs(pressure_history - analytic_history)))

    # Read the oscillation frequency back from the antinode next to a wall by
    # timing its upward zero crossings.
    signal = pressure_history[:, 0]
    crossings = []
    for i in range(1, len(signal)):
        if signal[i - 1] < 0.0 <= signal[i]:
            crossings.append(times[i - 1] + (times[i] - times[i - 1]) * (-signal[i - 1]) / (signal[i] - signal[i - 1]))
    measured_f = 1.0 / np.mean(np.diff(np.asarray(crossings)))
    rel_f_error = abs(measured_f - f_expected) / f_expected

    print(f"  {n}      {f_expected:8.2f} Hz   {measured_f:8.2f} Hz   {rel_f_error:9.2e}   {max_field_error:9.2e} Pa")

    assert rel_f_error < 0.01, f"Mode {n}: resonance frequency off by more than 1 %."
    assert max_field_error < 0.03 * p_amplitude, f"Mode {n}: standing-wave field deviates more than 3 % of the amplitude."

    results[n] = dict(times=times, pressure_history=pressure_history, period=period, f_expected=f_expected)

# Physical sanity on the fundamental: the antinode pressure swings fully through
# both signs (a genuine oscillation, not a decaying transient).
fundamental = results[1]["pressure_history"][:, 0]
assert fundamental.max() > 0.95 * p_amplitude and fundamental.min() < -0.95 * p_amplitude


# ## Figure
#
# Left: the spatial mode shapes half a period apart, lines are the closed-form
# cosines, dots the computed field. Right: the antinode pressure of the
# fundamental tracing out its period.

cream = "#fbf5eb"
fig, (ax_shape, ax_trace) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream)
colors = plt.cm.viridis(np.linspace(0.15, 0.75, len(modes)))
for ax in (ax_shape, ax_trace):
    ax.set_facecolor("#fffaf2")

for n, col in zip(modes, colors):
    history = results[n]["pressure_history"]
    xf = np.linspace(0.0, L, 300)
    ax_shape.plot(xf * 100, p_amplitude * np.cos(n * np.pi * xf / L), color=col, lw=1.6, label=f"mode n={n}")
    ax_shape.plot(x * 100, history[0], "o", color=col, ms=3)
ax_shape.axhline(0.0, color="0.7", lw=0.8)
ax_shape.set(xlabel="Position x / cm", ylabel="Pressure p / Pa",
             title="Mode shapes — lines: cos(nπx/L), dots: OneSimulation")
ax_shape.legend(frameon=False)
ax_shape.grid(alpha=0.25)

t1 = results[1]["times"]
period1 = results[1]["period"]
ax_trace.plot(t1 * 1000, results[1]["pressure_history"][:, 0], color=colors[0], lw=1.4,
              label="OneSimulation")
ax_trace.plot(t1 * 1000, p_amplitude * np.cos(2 * np.pi * results[1]["f_expected"] * t1),
              "--", color="0.35", lw=1.2, label="cos(2π f₁ t)")
for m in range(periods + 1):
    ax_trace.axvline(m * period1 * 1000, color="0.8", lw=0.8)
ax_trace.set(xlabel="Time t / ms", ylabel="Antinode pressure / Pa",
             title=f"Fundamental oscillation, f₁ = {results[1]['f_expected']:.1f} Hz")
ax_trace.legend(frameon=False)
ax_trace.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Refine the mesh or shorten the time step and watch both the frequency error
# and the field error shrink. Replace one rigid wall with a pressure-release end
# (an open pipe termination); the resonances then move to the odd quarter-wave
# series $f_n=(2n-1)c/(4L)$ with mode shapes $\cos((2n-1)\pi x/(2L))$.
