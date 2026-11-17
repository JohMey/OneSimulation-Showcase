"""OneSimulation example: transient heat conduction in a finite slab.
Requires the installed, activated OneSimulation runtime.

The slab is read from a STEP file, meshed with hexahedra and cooled from both surfaces; the
temperature profiles are checked against the closed-form Fourier series.
"""


# # Heat conduction in a finite slab
#
# A flat slab of thickness $L$ is initially at a uniform temperature $T_i$.
# From $t=0$ on, **both surfaces** are held at $T_s$, and the slab cools by pure
# heat conduction. The sides are insulated, so the problem is one-dimensional:
#
# $$\rho c_p \,\frac{\partial T}{\partial t} = k\,\frac{\partial^2 T}{\partial x^2},\qquad
#   T(x,0)=T_i,\quad T(0,t)=T(L,t)=T_s.$$
#
# The slab comes from the CAD file `heat_conduction_finite_slab.step`; its two surfaces are named
# `x_min` and `x_max`, the four lateral faces form the group `sides`. The script
# `heat_conduction_finite_slab_geometry.py` builds it with the CAD objects of OneSimulation.
# OneSimulation solves the heat equation with quadratic finite elements and the BDF2 time step.
#
# **Reference.** With $\theta=(T-T_s)/(T_i-T_s)$ and $\alpha=k/(\rho c_p)$, the
# classical Fourier series holds:
#
# $$\theta(x,t)=\sum_{n=1,3,5,\dots}\frac{4}{n\pi}\,
#   \sin\!\Big(\frac{n\pi x}{L}\Big)\,
#   \exp\!\Big[-\Big(\frac{n\pi}{L}\Big)^2\alpha t\Big].$$

import numpy as np

from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import theme, plots

theme.set()

# ## Geometry and mesh

slab = load("heat_conduction_finite_slab.step")
faces = slab.boundaries
thickness, cross, _ = slab.size
mesh = StructuredHexahedralMesher(cells=(60, 2, 2)).generate(slab)

fig = plots.figure(1, 2, size=(14, 4.5))
plots.geometry(slab, selection={"cooled surfaces": faces.x_min | faces.x_max, "sides": faces.sides}, labels=False,
               title="Slab with its cooled surfaces and insulated sides", ax=fig[0])
plots.mesh(mesh, groups=("all",), legend=False, title=f"{mesh.cell_count} hexahedra", ax=fig[1])
plots.show()

# ## Model
#
# The slab cools from 600 K to a surface temperature of 300 K over 0.6 of its characteristic time
# $L^2/\alpha$, in 40 time steps.

initial, surface = 600.0, 300.0
material = IsotropicSolid(density=1000.0, specific_heat_capacity=500.0, thermal_conductivity=50.0,
                          youngs_modulus=100e9, poisson_ratio=0.3, thermal_expansion_coefficient=1e-5)
diffusivity = material.thermal_conductivity / (material.density * material.specific_heat_capacity)
end_time = 0.6 * thickness**2 / diffusivity
model = HeatConduction(domain=slab.volume, material=material, heating=0.0)
problem = Problem(name="Heat conduction in a finite slab", geometry=slab, physics=(model,), boundary_conditions=(
    FixedTemperature(boundary=faces.x_min | faces.x_max, physics=model, value=surface),
    HeatFlux(boundary=faces.sides, physics=model, value=0.0)),
    initial_conditions=(InitialValue(model.temperature, initial),))
numerics = Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect(),
                    time_integrator=BDF(order=2, startup=BackwardEuler()))
result = Simulation(problem=problem, mesh=mesh, numerics=numerics).run(start_time=0.0, end_time=end_time, time_step=end_time / 40)

# ## Comparison with the series solution
#
# The temperature along the centre line at six stored times against the Fourier series with 150
# terms. The sudden surface step is a singularity at $t=0$; the first profile, two time steps
# after it, still carries its trace.

x = np.linspace(0.06 * thickness, 0.94 * thickness, 37)
times, profiles = result.field(model.temperature).time_series(np.column_stack((x, np.full_like(x, cross / 2), np.full_like(x, cross / 2))))
times, profiles = times[[2, 10, 17, 25, 32, 40]], profiles[[2, 10, 17, 25, 32, 40]]
n = np.arange(1, 301, 2)[:, None, None]
series = surface + (initial - surface) * np.sum(
    4 / (n * np.pi) * np.sin(n * np.pi * x / thickness) * np.exp(-(n * np.pi / thickness)**2 * diffusivity * times[:, None]), axis=0)
error = np.abs(profiles - series).max(axis=1)

print("time [s]   largest deviation from the series [K]")
for time, deviation in zip(times, error):
    print(f"{time:8.2f} {deviation:12.3f}")
print(f"largest deviation: {error.max():.3f} K ({100 * error.max() / (initial - surface):.2f} % of Ti - Ts), "
      f"after the first profile {error[1:].max():.3f} K")

# ## Figure

plots.line(points={f"t = {time:.0f} s": (1e3 * x, profile) for time, profile in zip(times, profiles)},
           reference={f"t = {time:.0f} s": (1e3 * x, profile) for time, profile in zip(times, series)},
           xlabel="position x [mm]", ylabel="temperature [K]",
           title="Cooling slab: simulation (markers) and Fourier series (dashed)")
plots.show()

# ## Going further
#
# Refine the mesh or halve the time step and watch the deviation from the series shrink. Change the
# thickness in `heat_conduction_finite_slab_geometry.py` and write the STEP file again: the end
# time follows the characteristic time $L^2/\alpha$. Or hold only one surface at $T_s$ and insulate
# the other; the series then runs over the odd multiples of the quarter wave.
