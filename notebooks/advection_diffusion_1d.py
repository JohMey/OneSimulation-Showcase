"""OneSimulation example: steady advection-diffusion of temperature in a channel.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the known closed-form Peclet profile.
"""


# # Steady advection-diffusion in a channel
#
# A fluid moves through a slender channel of length $L$ with a uniform axial
# velocity $u$. The two ends are held at fixed temperatures $T_0$ (inlet) and
# $T_1$ (outlet); the lateral walls are insulated, so the temperature only varies
# along the flow direction $x$. In steady state the material-temperature balance
#
# $$\rho c_p\,u\,\frac{dT}{dx}=k\,\frac{d^2T}{dx^2}$$
#
# describes a competition between advection (carrying heat downstream) and
# conduction (smoothing it out). OneSimulation solves the full three-dimensional
# convective heat equation with finite elements (Lagrange order 2) on a real mesh;
# the four lateral faces carry an explicit zero-flux condition, which reduces the
# problem to one dimension.
#
# **Reference.** With the Peclet number
#
# $$\mathrm{Pe}=\frac{u\,L\,\rho\,c_p}{k}$$
#
# the exact solution of the boundary value problem is
#
# $$T(x)=T_0+(T_1-T_0)\,\frac{e^{\mathrm{Pe}\,x/L}-1}{e^{\mathrm{Pe}}-1}.$$
#
# The dimensionless group $\mathrm{Pe}$ follows directly from the equation above:
# dividing by $k$ turns the advective term into $(\rho c_p u/k)\,dT/dx$, and the
# characteristic length $L$ makes it dimensionless. We compare the computed axial
# profile with this expression.

import numpy as np

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, GalerkinAdvection, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatTransfer, FixedTemperature, HeatFlux
from onesim.simulation import Problem, Simulation
from onesim.visualization import theme, plots

theme.set()


# ## Parameters
#
# The values are freely adjustable. The channel lies along $x$; its width and
# height are small and only serve to build a slender 3D mesh. The fluid properties
# are those of air near room temperature, and the slow velocity keeps the flow in
# a regime where advection and conduction are comparable (Peclet number of a few).

L = 0.10
cross = 0.01
rho = 1.2
cp = 1005.0
k = 0.025
T0 = 300.0
T1 = 360.0

peclet = 5.0
u = peclet * k / (L * rho * cp)
print(f"velocity u = {u:.4e} m/s,  Peclet number Pe = {u * L * rho * cp / k:.3f}")

geometry = Box(length=L, width=cross, height=cross)
mesh = StructuredHexahedralMesher(cells=(80, 2, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

material = IsotropicSolid(
    density=rho, specific_heat_capacity=cp, thermal_conductivity=k,
    youngs_modulus=100e9, poisson_ratio=0.3, thermal_expansion_coefficient=1e-5,
)
model = ConvectiveHeatTransfer(
    domain=geometry.volume, material=material, velocity=(u, 0.0, 0.0),
)
problem = Problem(
    name="Steady advection-diffusion in a channel",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=geometry.boundaries.x_min, physics=model, value=T0),
        FixedTemperature(boundary=geometry.boundaries.x_max, physics=model, value=T1),
        HeatFlux(
            boundary=(geometry.boundaries.y_min | geometry.boundaries.y_max
                      | geometry.boundaries.z_min | geometry.boundaries.z_max),
            physics=model, value=0.0,
        ),
    ),
)
simulation = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2), convection=GalerkinAdvection()),
        linear_solver=SparseDirect(),
    ),
)
simulation.validate().require_valid()
result = simulation.solve_stationary()


# ## Comparison with the closed-form profile
#
# We evaluate the computed temperature along the centre line and compare it with
# the Peclet profile. The exponential shape is the signature of advection: for
# small Peclet numbers the profile is nearly linear (conduction dominates), while
# for large Peclet numbers it bends sharply towards the downstream end.

sample_x = np.linspace(0.0, L, 41)
points = np.column_stack((
    sample_x,
    np.full_like(sample_x, cross / 2),
    np.full_like(sample_x, cross / 2),
))
numeric = np.asarray(result.field(model.temperature).evaluate(points)).reshape(-1)


def reference(x):
    return T0 + (T1 - T0) * (np.exp(peclet * x / L) - 1.0) / (np.exp(peclet) - 1.0)


exact = reference(sample_x)
max_abs_error = float(np.max(np.abs(numeric - exact)))
rel_error = max_abs_error / (T1 - T0)
print(f"Maximum absolute error against the Peclet profile: {max_abs_error:.4f} K"
      f"  ({100 * rel_error:.3f} % of T1-T0)")


# ## Figure

dense_x = np.linspace(0.0, L, 400)
plots.line(points={"OneSimulation FEM": (sample_x * 1000, numeric)},
           reference={"closed-form Péclet profile": (dense_x * 1000, reference(dense_x))},
           xlabel="position x [mm]", ylabel="temperature [K]", title=f"Advection-diffusion in a channel (Pe = {peclet:.0f})")
plots.show()


# ## Going further
#
# Raise the velocity (and with it the Peclet number) and watch the profile bend
# ever more sharply towards the outlet; near the downstream wall a thin thermal
# boundary layer forms whose width scales like $L/\mathrm{Pe}$. Resolving that
# layer needs more cells along $x$, so refine the mesh as you increase $u$ and
# confirm that the error against the closed-form profile stays small.
