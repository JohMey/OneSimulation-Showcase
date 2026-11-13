"""OneSimulation example: a small-amplitude acoustic pulse in a gas at rest.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against linear acoustics: the pulse splits
into two halves that travel at the speed of sound, and pressure and velocity
perturbations obey the acoustic impedance relation.
"""


# # An acoustic pulse in a gas at rest: sound speed and acoustic impedance
#
# A gas at rest ($\rho_0$, $p_0$, $u=0$) carries a small, smooth pressure bump.
# Because the amplitude is tiny compared with $p_0$, the compressible Euler
# equations reduce to **linear acoustics**: the bump splits into two identical
# halves that travel without change of shape in the $\pm x$ directions at the
# speed of sound
#
# $$a=\sqrt{\gamma\,p_0/\rho_0},$$
#
# and inside each travelling wave the perturbations are tied together by the
# acoustic impedance $\rho_0 a$ and the isentropic relation:
#
# $$\delta p=\pm\rho_0\,a\,\delta u,\qquad \delta\rho=\delta p/a^2 .$$
#
# OneSimulation integrates the full nonlinear 1D Euler equations with a
# WENO5 finite-difference scheme and classical RK4. We measure the speed of the
# right-running pulse from the position of its centroid and check the
# impedance relation point by point inside the pulse.

from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import (
    FiniteDifference, GlobalLaxFriedrichsSplit, WENO5Flux, WENOFluxDifference,
)
from onesim.numerics.time import ClassicalRungeKutta4, TimeStepLimits
from onesim.physics.hydrodynamics import (
    CompressibleEuler, IdealGasEquationOfState, TransmissiveBoundary,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Dimensionless gas at rest with unit density and pressure. The initial
# Gaussian pressure bump has a relative amplitude of 0.1 %, far inside the
# linear regime; density follows isentropically and the gas is initially at
# rest. The run is stopped while both pulses are still well inside the domain.

gamma = 1.4
rho0, p0 = 1.0, 1.0
a = np.sqrt(gamma * p0 / rho0)         # speed of sound
epsilon = 1.0e-3                       # relative pressure amplitude of the bump
x0, sigma = 0.5, 0.03                  # bump centre and width
end_time = 0.25                        # pulses then sit near x0 -/+ a*end_time
POINTS = 600
DT = 5.0e-4                            # CFL = a*DT/dx ~ 0.36
print(f"a = {a:.4f},  expected pulse travel a*t_end = {a*end_time:.4f},  "
      f"impedance rho0*a = {rho0*a:.4f}")


def total_energy(pressure, density, velocity):
    return pressure / (gamma - 1) + 0.5 * density * velocity**2


def initial_pressure(x):
    return p0 * (1.0 + epsilon * np.exp(-((x - x0) / sigma) ** 2))


def initial_density(x):
    return rho0 * (initial_pressure(x) / p0) ** (1.0 / gamma)   # isentropic


# ## Model and the actual computation
#
# Conserved variables are initialised from the bump; both ends are transmissive
# and lie far from the pulses, so they do not influence the result. A CFL limit
# guards the time step.

geometry = Line(start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0))
gas = CompressibleEuler(
    domain=geometry.line, equation_of_state=IdealGasEquationOfState(gamma=gamma),
)
problem = Problem(
    name="Acoustic pulse",
    geometry=geometry,
    physics=(gas,),
    initial_conditions=(
        InitialValue(gas.density, initial_density),
        InitialValue(gas.momentum, 0.0),
        InitialValue(gas.total_energy, lambda x: total_energy(initial_pressure(x), initial_density(x), 0.0)),
    ),
    boundary_conditions=(
        TransmissiveBoundary(physics=gas, boundary=geometry.boundaries.all),
    ),
)
mesh = StructuredLineMesher(cells=POINTS).generate(geometry)
numerics = Numerics(
    spatial=FiniteDifference(gradient=WENOFluxDifference(
        flux_splitting=GlobalLaxFriedrichsSplit(), reconstruction=WENO5Flux(),
    )),
    time_integrator=ClassicalRungeKutta4(),
    step_limits=TimeStepLimits(cfl_limit=0.4, stability_safety_factor=0.4),
)
simulation = Simulation(problem=problem, mesh=mesh, numerics=numerics)
simulation.validate().require_valid()
result = simulation.run(start_time=0.0, end_time=end_time, time_step=DT)


# ## Pulse speed and the acoustic impedance relation
#
# The final pressure perturbation consists of two half-height copies of the
# initial bump. The centroid of the right-running half must sit at
# $x_0+a\,t_\mathrm{end}$, that of the left-running half at $x_0-a\,t_\mathrm{end}$.
# Inside the right-running pulse the velocity must equal $\delta p/(\rho_0 a)$ and
# the density perturbation $\delta p/a^2$.

x = np.asarray(mesh.cell_centers[:, 0])
rho = np.asarray(result.field(gas.density).values)
pressure = np.asarray(result.field(gas.pressure).values)
velocity = np.asarray(result.field(gas.velocity).values)
assert np.isfinite([rho, pressure, velocity]).all()
dp = pressure - p0
drho = rho - rho0

right = x > x0
left = x < x0
centroid_right = float(np.sum(x[right] * dp[right]) / np.sum(dp[right]))
centroid_left = float(np.sum(x[left] * dp[left]) / np.sum(dp[left]))
speed_right = (centroid_right - x0) / end_time
speed_left = (x0 - centroid_left) / end_time
print(f"Right-running pulse: centroid {centroid_right:.4f}, speed {speed_right:.4f}  (sound speed {a:.4f})")
print(f"Left-running pulse:  centroid {centroid_left:.4f}, speed {speed_left:.4f}")

# Impedance relation inside the right-running pulse (where dp is appreciable).
core = right & (dp > 0.05 * dp[right].max())
impedance_error = float(np.max(np.abs(velocity[core] - dp[core] / (rho0 * a))) / (dp[right].max() / (rho0 * a)))
isentropic_error = float(np.max(np.abs(drho[core] - dp[core] / a**2)) / (dp[right].max() / a**2))
print(f"Max deviation from dp = rho0*a*du inside the pulse: {100*impedance_error:.2f} % of the peak")
print(f"Max deviation from drho = dp/a^2 inside the pulse: {100*isentropic_error:.2f} % of the peak")

# Each half carries half the initial amplitude (linear splitting) and half the
# integrated pressure excess.
peak_ratio = dp[right].max() / (epsilon * p0)
excess_initial = float(np.sum(initial_pressure(x) - p0) / POINTS)
excess_right = float(np.sum(dp[right]) / POINTS)
print(f"Peak of the right-running pulse: {peak_ratio:.3f} of the initial amplitude (linear theory: 0.5)")
print(f"Integrated pressure excess, right half: {excess_right/excess_initial:.4f} of the initial total (theory 0.5)")

assert abs(speed_right - a) / a < 0.01, "Right-running pulse does not travel at the speed of sound."
assert abs(speed_left - a) / a < 0.01, "Left-running pulse does not travel at the speed of sound."
assert impedance_error < 0.02, "The acoustic impedance relation dp = rho0 a du is violated."
assert isentropic_error < 0.02, "The isentropic relation drho = dp/a^2 is violated."
assert 0.45 < peak_ratio < 0.52, "Each half should carry about half the initial amplitude."
assert abs(excess_right / excess_initial - 0.5) < 0.01, "The pressure excess must split equally."
# The gas between the two pulses has returned to rest.
between = (x > x0 - 0.5 * a * end_time) & (x < x0 + 0.5 * a * end_time)
assert np.max(np.abs(dp[between])) < 0.02 * epsilon * p0, "The gas between the pulses must be undisturbed."
assert np.max(np.abs(velocity[between])) < 0.02 * epsilon * p0 / (rho0 * a)
print("All linear-acoustics checks passed.")


# ## Figure

cream = "#fbf5eb"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
fig.patch.set_facecolor(cream)

ax = axes[0]
ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(x, (initial_pressure(x) - p0) / (epsilon * p0), color="#b4aa98", lw=1.1, ls="--", label="initial bump, t = 0")
ax.plot(x, dp / (epsilon * p0), color="#39715e", lw=1.8, label=f"OneSimulation, t = {end_time:g}")
for position in (x0 - a * end_time, x0 + a * end_time):
    ax.axvline(position, color="#A85539", ls=":", lw=1.2)
ax.text(x0 + a * end_time, 0.56, r"$x_0 + a\,t$", color="#A85539", ha="center", fontsize=9)
ax.text(x0 - a * end_time, 0.56, r"$x_0 - a\,t$", color="#A85539", ha="center", fontsize=9)
ax.set(xlabel="Position x", ylabel=r"Pressure perturbation $\delta p\,/\,(\epsilon p_0)$", xlim=(0, 1),
       title="The bump splits into two pulses travelling at the sound speed")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(x, rho0 * a * velocity / (epsilon * p0), color="#4a7fb5", lw=1.8, label=r"$\rho_0 a\,u$")
ax.plot(x, dp / (epsilon * p0), color="#39715e", lw=1.2, ls="--", label=r"$\delta p$")
ax.set(xlabel="Position x", ylabel=r"Scaled perturbation / $(\epsilon p_0)$", xlim=(0, 1),
       title=r"Acoustic impedance: $\delta p = \pm\rho_0 a\,\delta u$")
ax.legend(frameon=False, fontsize=9)
plt.show()


# ## Going further
#
# Raise `epsilon` towards 0.1 or more: the compression side of each pulse
# travels faster than the rarefaction side (speed $u+a$ grows with pressure),
# the waves steepen and eventually form weak shocks, the nonlinear regime the
# Euler solver captures beyond linear acoustics. Replace the transmissive ends by
# impermeable walls and watch the pulses reflect with unchanged pressure sign and
# reversed velocity.
