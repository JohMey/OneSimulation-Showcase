"""OneSimulation example: steady advection-diffusion of temperature in a channel.
Requires the installed, activated OneSimulation runtime.

The channel is read from a STEP file, meshed with hexahedra and solved for the temperature of the
moving fluid; the axial profile is checked against the closed-form Peclet profile.
"""

# # Steady advection-diffusion in a channel
#
# A fluid moves through a slender channel of length $L$ with a uniform axial
# velocity $u$. The two ends are held at fixed temperatures $T_0$ (inlet) and
# $T_1$ (outlet); the lateral walls are insulated, so the temperature only varies
# along the flow direction $x$. In steady state the balance
#
# $$\rho c_p\,u\,\frac{dT}{dx}=k\,\frac{d^2T}{dx^2}$$
#
# describes a competition between advection (carrying heat downstream) and
# conduction (smoothing it out).
#
# The channel comes from the CAD file `advection_diffusion_1d.step`; inlet and outlet are named
# `x_min` and `x_max`, the four lateral walls form the group `walls`. The script
# `advection_diffusion_1d_geometry.py` builds it with the CAD objects of OneSimulation.
# OneSimulation solves the three-dimensional convective heat equation with quadratic finite
# elements.
#
# **Reference.** With the Peclet number $\mathrm{Pe}=u\,L\,\rho\,c_p/k$ the exact solution of the
# boundary value problem is
#
# $$T(x)=T_0+(T_1-T_0)\,\frac{e^{\mathrm{Pe}\,x/L}-1}{e^{\mathrm{Pe}}-1}.$$

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, GalerkinAdvection, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import (
    ConvectiveHeatTransfer,
    FixedTemperature,
    HeatFlux,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh

channel = load("advection_diffusion_1d.step")
faces = channel.boundaries
length, cross, _ = channel.size
mesh = StructuredHexahedralMesher(cells=(80, 2, 2)).generate(channel)

fig = plots.figure(1, 2, size=(14, 4.5))
plots.geometry(
    channel,
    selection={"inlet": faces.x_min, "outlet": faces.x_max},
    labels=False,
    title="Channel with its inlet and outlet",
    ax=fig[0],
)
plots.mesh(
    mesh, groups=("all",), legend=False, title=f"{mesh.cell_count} hexahedra", ax=fig[1]
)
plots.show()

# ## Model
#
# Air near room temperature. The velocity is chosen for a Peclet number of 5, where advection and
# conduction are comparable.

inlet, outlet, peclet = 300.0, 360.0, 5.0
air = IsotropicSolid(
    density=1.2,
    specific_heat_capacity=1005.0,
    thermal_conductivity=0.025,
    youngs_modulus=100e9,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=1e-5,
)
velocity = (
    peclet
    * air.thermal_conductivity
    / (length * air.density * air.specific_heat_capacity)
)
model = ConvectiveHeatTransfer(
    domain=channel.volume, material=air, velocity=(velocity, 0.0, 0.0)
)
problem = Problem(
    name="Steady advection-diffusion in a channel",
    geometry=channel,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=faces.x_min, physics=model, value=inlet),
        FixedTemperature(boundary=faces.x_max, physics=model, value=outlet),
        HeatFlux(boundary=faces.walls, physics=model, value=0.0),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2), convection=GalerkinAdvection()),
    linear_solver=SparseDirect(),
)
temperature = (
    Simulation(problem=problem, mesh=mesh, numerics=numerics)
    .solve_stationary()
    .field(model.temperature)
)

# ## Comparison with the closed-form profile

x = np.linspace(0.0, length, 41)
simulated = temperature.evaluate(
    np.column_stack((x, np.full_like(x, cross / 2), np.full_like(x, cross / 2)))
).reshape(-1)
exact = inlet + (outlet - inlet) * (np.exp(peclet * x / length) - 1) / (
    np.exp(peclet) - 1
)

print(f"velocity {1e3 * velocity:.4f} mm/s, Peclet number {peclet:.0f}")
print(
    f"largest deviation from the Peclet profile: {np.abs(simulated - exact).max():.2e} K "
    f"({100 * np.abs(simulated - exact).max() / (outlet - inlet):.4f} % of T1 - T0)"
)

# ## Figure

plots.line(
    1e3 * x,
    points={"simulation": simulated},
    reference={"closed-form Péclet profile": exact},
    xlabel="position x [mm]",
    ylabel="temperature [K]",
    title=f"Advection-diffusion in a channel (Pe = {peclet:.0f})",
)
plots.show()

# ## Going further
#
# Raise the Peclet number and watch the profile bend ever more sharply towards the outlet; near the
# downstream end a thin thermal boundary layer forms whose width scales like $L/\mathrm{Pe}$.
# Resolving that layer needs more cells along $x$, so refine the mesh as you increase the velocity.
