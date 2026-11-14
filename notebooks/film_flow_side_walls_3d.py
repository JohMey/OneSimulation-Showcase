"""OneSimulation example: gravity-driven film flow down an inclined open channel with side walls.
Requires the installed, activated OneSimulation runtime.

A liquid film runs down an inclined open channel. The side walls slow the liquid next to them, so
the velocity varies over the whole cross-section. The flow is computed in 3D with Taylor-Hood
finite elements and with staggered finite differences and compared with the exact series solution.
"""


# # Film flow down an inclined channel with side walls
#
# A liquid film of depth $H$ flows down an open channel of width $W$ that is inclined at the
# angle $\beta$. The liquid sticks to the bottom and to both side walls; its flat free surface is
# free of shear stress. Coordinates: $x$ down the channel, $y\in[0,W]$ across, $z\in[0,H]$ from the
# bottom to the surface.
#
# **Reference.** Far from the inlet the flow obeys $\mu\nabla^2u=-\rho g\sin\beta$ on the
# cross-section. With $\lambda_n=(2n+1)\pi/(2H)$ the solution is the series (White, *Viscous Fluid
# Flow*)
#
# $$u(y,z)=\frac{\rho g\sin\beta}{\mu}\left[Hz-\frac{z^2}{2}-\sum_{n=0}^{\infty}\frac{2}{H\lambda_n^3}
#   \sin(\lambda_n z)\,\frac{\cosh\lambda_n(y-W/2)}{\cosh(\lambda_n W/2)}\right],\qquad
#   \frac{Q}{W\,q_N}=1-\frac{12H}{W}\sum_{n=0}^{\infty}\frac{\tanh(\lambda_n W/2)}{(\lambda_n H)^5} .$$
#
# $q_N=\rho g\sin\beta\,H^3/(3\mu)$ is the flow rate per width of the Nusselt film without side
# walls.

import numpy as np

from onesim.cad.d2 import Plane
from onesim.cad.d3 import Box
from onesim.materials import NewtonianFluid
from onesim.meshing import PeriodicPair, StructuredHexahedralMesher, StructuredTetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import CentralDifference, CentralLaplacian, FiniteDifference
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import FlowTraction, IncompressibleNavierStokes, NoSlipWall
from onesim.results.integration import mean_value
from onesim.simulation import PeriodicBoundary, Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import theme, plots

theme.set()

depth, tilt = 0.002, np.deg2rad(10.0)
oil = NewtonianFluid(density=1200.0, dynamic_viscosity=0.10)
drive = oil.density * 9.81 * np.sin(tilt)
nusselt_rate = drive * depth**3 / (3 * oil.dynamic_viscosity)
nusselt_speed = drive * depth**2 / (2 * oil.dynamic_viscosity)


def exact_velocity(y, z, width):
    """Velocity down the channel at the point (y, z) of the cross-section from the series solution."""
    lam = (2 * np.arange(200) + 1) * np.pi / (2 * depth)
    side = np.abs(y - width / 2)
    walls = np.exp(lam * (side - width / 2)) * (1 + np.exp(-2 * lam * side)) / (1 + np.exp(-lam * width))
    return drive / oil.dynamic_viscosity * (depth * z - z**2 / 2 - np.sum(2 / (depth * lam**3) * np.sin(lam * z) * walls))


def exact_flow_rate(width):
    """Flow rate of the film in a channel of the given width from the series solution."""
    k = (2 * np.arange(400) + 1) * np.pi / 2
    return nusselt_rate * width * (1 - 12 * depth / width * np.sum(np.tanh(k * width / (2 * depth)) / k**5))


# ## Model
#
# Both methods solve the full Navier-Stokes equations on a short slab of the channel whose
# upstream and downstream faces are joined by a periodic condition. Gravity acts as a body force.

methods = {"Taylor-Hood finite elements": (FiniteElement(element=TaylorHood()), StructuredTetrahedralMesher, 1),
           "staggered finite differences": (FiniteDifference(gradient=CentralDifference(), diffusion=CentralLaplacian()),
                                            StructuredHexahedralMesher, 5)}


def solve(method, aspect, cells):
    """Velocity field for a channel of width aspect * depth with `cells` cells over the depth."""
    spatial, mesher, along = methods[method]
    width, across = aspect * depth, round(aspect * cells)
    channel = Box(length=along * width / across, width=width, height=depth)
    faces = channel.boundaries
    model = IncompressibleNavierStokes(
        domain=channel.volume, material=oil,
        body_force=lambda x, y, z: np.column_stack((np.full(len(x), drive), np.zeros(len(x)),
                                                    np.full(len(x), -oil.density * 9.81 * np.cos(tilt)))))
    problem = Problem(name="Film flow in an inclined channel", geometry=channel, physics=(model,), boundary_conditions=(
        NoSlipWall(boundary=faces.z_min | faces.y_min | faces.y_max, physics=model),
        FlowTraction(boundary=faces.z_max, physics=model, value=(0.0, 0.0, 0.0)),
        PeriodicBoundary(pair=PeriodicPair(first=faces.x_min, second=faces.x_max), physics=model)),
        initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),))
    numerics = Numerics(spatial=spatial, initial_guess={model.velocity: (0.0, 0.0, 0.0)},
                        nonlinear_solver=Newton(linear_solver=SparseDirect(), absolute_tolerance=1e-12, relative_tolerance=1e-10))
    mesh = mesher(cells=(along, across, cells)).generate(channel)
    velocity = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.velocity)
    return velocity, channel


aspects = (0.5, 1.0, 2.0, 4.0, 8.0)
channels = {aspect: solve("Taylor-Hood finite elements", aspect, 16) for aspect in aspects}
studies = {"Taylor-Hood finite elements": (4, 8, 16), "staggered finite differences": (8, 16, 32)}
surface_speed = lambda velocity, channel: velocity.evaluate([[channel.length / 2, channel.width / 2, depth]])[0, 0]
errors = {method: [abs(surface_speed(*solve(method, 4.0, cells)) / exact_velocity(2 * depth, depth, 4 * depth) - 1) for cells in levels]
          for method, levels in studies.items()}


# ## Results
#
# The flow does not change along the channel, so the flow rate is the mean velocity times the
# cross-section. Relative to the Nusselt film, narrow channels carry far less. The mesh study
# compares the speed at the middle of the free surface for both methods.

print(" W/H   flow rate / (W q_N)   exact series   deviation")
rates = {aspect: mean_value(velocity, component=0) * channel.width * depth for aspect, (velocity, channel) in channels.items()}
for aspect, rate in rates.items():
    exact = exact_flow_rate(aspect * depth)
    print(f"{aspect:4.1f} {rate / (aspect * depth * nusselt_rate):18.6f} {exact / (aspect * depth * nusselt_rate):14.6f} {rate / exact - 1:+11.1e}")
for method, levels in studies.items():
    print(f"{method}, W/H = 4, {levels} cells over the depth: surface-speed error " + ", ".join(f"{e:.1e}" for e in errors[method]))


# ## Figure
#
# The velocity over the cross-section for $W/H=4$, the flow rate against the channel width, and
# the mesh study of both methods.

velocity, channel = channels[4.0]
section = Plane(origin=(channel.length / 2, 0, 0), normal=(1, 0, 0))
wide = np.geomspace(0.25, 20, 200)

fig = plots.figure(layout=[["field", "field"], ["rate", "study"]], size=(13, 9))
plots.section(velocity, plane=section, component=lambda u, position: u[:, 0] / nusselt_speed, limits=(0, 1), contours=9,
              subdivisions=2, label="u / surface speed of the Nusselt film", xlabel="y across the channel [m]",
              ylabel="z above the bottom [m]", title="Velocity down the channel, W/H = 4", ax=fig["field"])
plots.line(wide, {"exact series": [exact_flow_rate(a * depth) / (a * depth * nusselt_rate) for a in wide]},
           points={"finite elements": (list(rates), [rate / (a * depth * nusselt_rate) for a, rate in rates.items()])},
           scale="logx", ylimits=(0, 1.02), xlabel="aspect ratio W/H", ylabel="flow rate / (W q_N)",
           title="Side walls reduce the flow rate", legend="upper left", ax=fig["rate"])
plots.line(y={method: ([1 / cells for cells in levels], errors[method]) for method, levels in studies.items()}, markers=True,
           scale="loglog", xlabel="cell size / depth", ylabel="relative error of the surface speed", title="Mesh study, W/H = 4",
           ax=fig["study"])
plots.show()


# ## Going further
#
# Tilt the channel further and the film speeds up with $\sin\beta$; a shear stress on the free
# surface (wind) changes the profile near the surface. The same slab with a round bottom gives the
# flow in a partly filled pipe.
