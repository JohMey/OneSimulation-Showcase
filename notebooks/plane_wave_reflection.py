"""OneSimulation example: a plane acoustic pulse crossing a material interface.
Requires the installed, activated OneSimulation runtime.

The reflected and transmitted amplitudes are checked against the closed-form
acoustic impedance (Fresnel) relations for normal incidence.
"""


# # Reflection and transmission at an acoustic impedance jump
#
# A one-dimensional duct is filled with two gases that meet at a sharp interface
# at $x=x_i$. The left gas has density $\rho_1$ and sound speed $c_1$, the right
# gas $\rho_2$ and $c_2$; their characteristic acoustic impedances are
# $Z_i=\rho_i c_i$. A right-running pressure pulse starts in the left gas. For a
# right-running plane wave the particle velocity is $u=p/Z_1$, which is how the
# initial velocity is set so that the pulse travels purely to the right.
#
# The linear acoustic equations are solved with spatially varying material
# coefficients:
#
# $$\rho(x)\,\frac{\partial u}{\partial t}=-\frac{\partial p}{\partial x},\qquad
#   \kappa(x)\,\frac{\partial p}{\partial t}=-\frac{\partial u}{\partial x},\qquad
#   \kappa=\frac{1}{\rho c^{2}}.$$
#
# **Reference.** At normal incidence the pressure reflection and transmission
# coefficients depend only on the impedances:
#
# $$R=\frac{Z_2-Z_1}{Z_2+Z_1},\qquad T=\frac{2Z_2}{Z_2+Z_1}=1+R.$$
#
# After the pulse has split, the reflected pulse carries amplitude $R$ (relative
# to the incident unit amplitude) and the transmitted pulse carries $T$. We
# compare the computed pulse amplitudes and shapes against these predictions.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d1 import Line
from onesim.meshing.meshes.api import cartesian
from onesim.numerics import Numerics
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.acoustics import LinearAcoustics, StaggeredAcousticDifference
from onesim.simulation import ExecutionOptions, Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.simulation.problem_definition.boundaries import ReflectingWall

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The values are freely adjustable. The duct is long enough that the reflected
# and transmitted pulses are read off well before either reaches a wall, so the
# distant rigid ends do not influence the measurement window.

length = 8.0          # m, duct length
interface = 4.0       # m, position of the material interface
cells = 512           # finite-volume cells along the duct

rho_1, c_1 = 1.2, 340.0     # left gas: density [kg/m^3], sound speed [m/s]
rho_2, c_2 = 3.0, 476.0     # right gas (denser, stiffer)
z_1, z_2 = rho_1 * c_1, rho_2 * c_2

reflection = (z_2 - z_1) / (z_2 + z_1)
transmission = 2.0 * z_2 / (z_2 + z_1)

pulse_center = 2.0    # m, initial pulse position (left gas)
pulse_width = 0.25    # m, Gaussian half-width of the pulse

# Timing: when the pulse hits the interface, and a measurement time at which the
# split pulses sit cleanly inside each medium.
time_to_interface = (interface - pulse_center) / c_1
travel_after = 1.6 / c_2                       # let the transmitted pulse run 1.6 m
end_time = time_to_interface + travel_after
cfl = 0.6
time_step = cfl * (length / cells) / max(c_1, c_2)

print(f"Z1 = {z_1:.1f},  Z2 = {z_2:.1f} kg/(m^2 s)")
print(f"Predicted R = {reflection:.4f},  T = {transmission:.4f}")
print(f"end time = {end_time * 1000:.3f} ms,  time step = {time_step * 1e6:.2f} us")

geometry = Line(start=(0.0, 0.0, 0.0), end=(length, 0.0, 0.0))
coordinate = (np.arange(cells) + 0.5) * (length / cells)   # cell centres


# ## Simulation
#
# The density is a position callback and the compressibility an array evaluated
# on the cell centres; both jump at the interface. A conservative staggered
# pressure/velocity scheme is advanced with classical Runge-Kutta. The two far
# ends of the duct carry an explicit rigid-wall boundary condition.

compressibility = np.where(coordinate < interface, 1.0 / (rho_1 * c_1**2), 1.0 / (rho_2 * c_2**2))


def density(x, y=None, z=None):
    return np.where(x < interface, rho_1, rho_2)


def incident_pulse(x, y=None, z=None):
    relative = (x - pulse_center) / pulse_width
    return np.where(np.abs(relative) < 7.0, np.exp(-(relative**2)), 0.0)


def incident_velocity(x, y=None, z=None):
    # Right-running plane wave: u = p / Z1.
    return incident_pulse(x) / z_1


model = LinearAcoustics(domain=geometry.line, density=density, compressibility=compressibility)
problem = Problem(
    name="Plane wave at an impedance jump",
    geometry=geometry,
    physics=(model,),
    initial_conditions=(
        InitialValue(model.pressure, incident_pulse),
        InitialValue(model.velocity, incident_velocity),
    ),
    # Both far ends of the duct are rigid walls (zero normal velocity).
    boundary_conditions=(
        ReflectingWall(physics=model, boundary=geometry.boundaries.all),
    ),
)
simulation = Simulation(
    problem=problem,
    mesh=cartesian(shape=(cells,), extents=(length,), geometry=geometry, periodic=False).build(),
    numerics=Numerics(spatial=StaggeredAcousticDifference(), time_integrator=ClassicalRungeKutta4()),
)
simulation.validate().require_valid()
result = simulation.run(
    start_time=0.0,
    end_time=end_time,
    time_step=time_step,
    execution_options=ExecutionOptions(record_result_history=False),
)
pressure = np.asarray(result.field(model.pressure).values).reshape(-1)


# ## Comparison with the impedance relations
#
# Each outgoing pulse is a Gaussian of the incident shape scaled by its
# coefficient. The reflected pulse (left medium) is mirrored about the interface
# and keeps the incident width; the transmitted pulse (right medium) is widened
# by the speed ratio $c_2/c_1$.

left = coordinate < interface
right = coordinate >= interface

reflected_center = interface - c_1 * travel_after
transmitted_center = interface + c_2 * travel_after
transmitted_width = pulse_width * c_2 / c_1

reflected_expected = reflection * np.exp(-((coordinate[left] - reflected_center) / pulse_width) ** 2)
transmitted_expected = transmission * np.exp(-((coordinate[right] - transmitted_center) / transmitted_width) ** 2)

peak_reflected = float(pressure[left].max())
peak_transmitted = float(pressure[right].max())
rel_reflected = abs(peak_reflected - reflection) / reflection
rel_transmitted = abs(peak_transmitted - transmission) / transmission
shape_error_left = float(np.max(np.abs(pressure[left] - reflected_expected)))
shape_error_right = float(np.max(np.abs(pressure[right] - transmitted_expected)))

print("Reflected  amplitude:  predicted R =", f"{reflection:.4f}", " computed =", f"{peak_reflected:.4f}",
      f" (rel. error {rel_reflected:.2e})")
print("Transmitted amplitude: predicted T =", f"{transmission:.4f}", " computed =", f"{peak_transmitted:.4f}",
      f" (rel. error {rel_transmitted:.2e})")
print(f"Max shape deviation: reflected {shape_error_left:.4f} Pa, transmitted {shape_error_right:.4f} Pa")

assert rel_reflected < 0.03, "Reflected amplitude off by more than 3 % from (Z2-Z1)/(Z2+Z1)."
assert rel_transmitted < 0.03, "Transmitted amplitude off by more than 3 % from 2 Z2/(Z2+Z1)."
assert shape_error_left < 0.04, "Reflected pulse shape deviates from the scaled Gaussian."
assert shape_error_right < 0.06, "Transmitted pulse shape deviates from the scaled Gaussian."
# T = 1 + R must hold to machine precision for the predictions themselves.
assert abs(transmission - (1.0 + reflection)) < 1e-12


# ## Figure
#
# The computed field (solid) against the two predicted Gaussian pulses (dashed),
# with the interface marked.

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
ax.axvspan(0, interface, color="#eaf2f8", alpha=0.6)
ax.axvspan(interface, length, color="#f8efe6", alpha=0.7)
ax.axvline(interface, color="0.5", lw=1.0)
ax.plot(coordinate, pressure, color="#1f6f8b", lw=1.6, label="OneSimulation")
ax.plot(coordinate[left], reflected_expected, "--", color="0.3", lw=1.2, label="predicted R-pulse")
ax.plot(coordinate[right], transmitted_expected, "--", color="#b5651d", lw=1.2, label="predicted T-pulse")
ax.text(interface * 0.5, 1.05 * transmission, f"gas 1\nZ₁={z_1:.0f}", ha="center", va="top", fontsize=9)
ax.text(interface + (length - interface) * 0.5, 1.05 * transmission, f"gas 2\nZ₂={z_2:.0f}",
        ha="center", va="top", fontsize=9)
ax.set(xlabel="Position x / m", ylabel="Pressure p / Pa",
       title="Pulse split at an impedance jump — R and T from Z₂, Z₁")
ax.legend(frameon=False, loc="lower left")
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Swap the two gases so $Z_2<Z_1$: the reflection coefficient changes sign and
# the reflected pulse flips polarity, while $T=1+R$ stays below one. Match the
# impedances ($Z_2=Z_1$) with different sound speeds and the reflection vanishes
# even though the media differ.
