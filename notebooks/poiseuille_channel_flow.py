"""OneSimulation example: laminar Poiseuille flow in a plane channel.
Requires the installed, activated OneSimulation runtime.

The channel is read from a STEP file and meshed with tetrahedra; the velocity profile of the
Navier-Stokes solution is checked against the closed-form parabola.
"""

# # Poiseuille flow between parallel plates
#
# An incompressible Newtonian fluid fills the gap of height $h$ between two fixed,
# parallel no-slip plates. A constant streamwise body force $G$ (per unit volume,
# equivalent to a constant pressure gradient $-\partial p/\partial x = G$) drives the
# flow. Once the flow is fully developed it depends only on the wall-normal
# coordinate $y$, and the steady incompressible Navier-Stokes equations reduce to
#
# $$\mu\,\frac{d^2 u}{dy^2} + G = 0,\qquad u(0)=u(h)=0.$$
#
# The channel comes from the CAD file `poiseuille_channel_flow.step`; its faces are named `x_min`,
# `x_max`, `y_min`, `y_max`, `z_min` and `z_max`, and the plates are `y_min` and `y_max`. The script
# `poiseuille_channel_flow_geometry.py` builds it with the CAD objects of OneSimulation.
# OneSimulation solves the full 3D stationary Navier-Stokes system
# $\rho(u\cdot\nabla)u=-\nabla p+\mu\nabla^2 u+f$ with Taylor-Hood finite elements (quadratic
# velocity, linear pressure) and a Newton solver.
#
# **Reference.** The momentum balance integrates to the parabolic profile
#
# $$u(y)=\frac{G}{2\mu}\,y\,(h-y),\qquad
#   u_\max=\frac{G\,h^2}{8\mu},\qquad
#   \bar{u}=\tfrac{2}{3}\,u_\max.$$

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import NewtonianFluid
from onesim.meshing import StructuredTetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import (
    FixedVelocity,
    IncompressibleNavierStokes,
    NoSlipWall,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh

channel = load("poiseuille_channel_flow.step")
faces = channel.boundaries
length, gap, span = channel.size
mesh = StructuredTetrahedralMesher(cells=(6, 8, 2)).generate(channel)

fig = plots.figure(1, 2, size=(13, 5))
plots.geometry(
    channel,
    selection={"plates": faces.y_min | faces.y_max},
    labels=False,
    title="Channel piece with its two plates",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"{mesh.cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The plates are no-slip walls. The inlet, the outlet and the two side faces carry the fully
# developed profile, so the short piece stands for an infinitely long and wide channel.

forcing = 0.8
fluid = NewtonianFluid(density=1.0, dynamic_viscosity=1.0e-3)
parabola = lambda y: forcing / (2 * fluid.dynamic_viscosity) * y * (gap - y)
model = IncompressibleNavierStokes(
    domain=channel.volume,
    material=fluid,
    body_force=lambda y: np.column_stack((np.full_like(y, forcing), 0 * y, 0 * y)),
)
problem = Problem(
    name="Poiseuille channel flow",
    geometry=channel,
    physics=(model,),
    boundary_conditions=(
        NoSlipWall(boundary=faces.y_min | faces.y_max, physics=model),
        FixedVelocity(
            boundary=faces.x_min | faces.x_max | faces.z_min | faces.z_max,
            physics=model,
            value=lambda y: np.column_stack((parabola(y), 0 * y, 0 * y)),
        ),
    ),
    constraints=(MeanValue(model.pressure, 0.0),),
    initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),),
)
numerics = Numerics(
    spatial=FiniteElement(element=TaylorHood()),
    initial_guess={model.velocity: (0.0, 0.0, 0.0)},
    nonlinear_solver=Newton(
        linear_solver=SparseDirect(), absolute_tolerance=1e-11, relative_tolerance=0.0
    ),
)
velocity = (
    Simulation(problem=problem, mesh=mesh, numerics=numerics)
    .solve_stationary()
    .field(model.velocity)
)

# ## Comparison with the parabolic profile
#
# The streamwise velocity on a line from plate to plate in the middle of the channel. Taylor-Hood
# velocities are quadratic, so the parabola is represented to solver precision.

y = np.linspace(0.0, gap, 41)
u = velocity.evaluate(
    np.column_stack((np.full_like(y, length / 2), y, np.full_like(y, span / 2)))
)[:, 0]
peak = forcing * gap**2 / (8 * fluid.dynamic_viscosity)

print(
    f"Reynolds number: {fluid.density * 2 / 3 * peak * gap / fluid.dynamic_viscosity:.2f}"
)
print(f"maximum velocity: simulation {u.max():.5f} m/s, parabola {peak:.5f} m/s")
print(
    f"largest deviation from the parabola: {np.abs(u - parabola(y)).max():.1e} m/s ({100 * np.abs(u - parabola(y)).max() / peak:.1e} % of the maximum)"
)

# ## Figure

plots.line(
    points={"simulation": (1e3 * u, 1e3 * y)},
    reference={"parabola": (1e3 * parabola(y), 1e3 * y)},
    hlines=[0.0, 1e3 * gap],
    xlabel="streamwise velocity u [mm/s]",
    ylabel="wall-normal position y [mm]",
    title="Poiseuille channel flow: velocity profile between the plates",
)
plots.show()

# ## Going further
#
# Change the gap in `poiseuille_channel_flow_geometry.py` and write the STEP file again: the peak
# velocity grows with $h^2$. Replace the body force by a pressure difference between the faces
# `x_min` and `x_max` (a traction condition) and recover the same parabola, or raise the forcing
# until the Reynolds number leaves the creeping regime.
