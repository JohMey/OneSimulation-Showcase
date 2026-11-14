"""OneSimulation example: laminar Poiseuille flow in a plane channel.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form parabolic profile.
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
# OneSimulation solves the full 3D stationary Navier-Stokes system
# $\rho(u\cdot\nabla)u=-\nabla p+\mu\nabla^2 u+f$ with inf-sup stable Taylor-Hood
# finite elements (quadratic velocity, linear pressure) and a Newton solver.
#
# **Reference.** The momentum balance integrates to the parabolic profile
#
# $$u(y)=\frac{G}{2\mu}\,y\,(h-y),\qquad
#   u_\max=\frac{G\,h^2}{8\mu},\qquad
#   \bar{u}=\tfrac{2}{3}\,u_\max.$$
#
# We check the computed profile against this parabola.

import numpy as np

from onesim.cad.d3 import Box
from onesim.materials import NewtonianFluid
from onesim.meshing import StructuredTetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import FixedVelocity, IncompressibleNavierStokes, NoSlipWall
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import theme, plots

theme.set()


# ## Parameters
#
# The gap height lies along $y$; the streamwise length ($x$) and spanwise width
# ($z$) only serve to form a short 3D channel. The flow is fully developed, so the
# solution is independent of $x$ and $z$. The body force sets the flow rate.

length = 0.020
height = 0.010
span = 0.005
mu = 1.0e-3
rho = 1.0
body_force = 0.8

u_max_reference = body_force * height**2 / (8.0 * mu)
u_mean_reference = 2.0 / 3.0 * u_max_reference
reynolds = rho * u_mean_reference * height / mu
print(f"G = {body_force:.3f} N/m^3,  u_max = {u_max_reference:.4f} m/s,  Re = {reynolds:.2f} (laminar)")


# ## Model
#
# The plate gap $h$ is the box width (the $y$ extent). The two plates (y_min, y_max)
# are no-slip walls; the inlet, outlet and spanwise faces carry the fully developed
# parabolic trace, so every one of the six boundary faces has an explicit condition.

geometry = Box(length=length, width=height, height=span)
mesh = StructuredTetrahedralMesher(cells=(6, 8, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "tetrahedra")

fluid = NewtonianFluid(density=rho, dynamic_viscosity=mu)


def forcing(y):
    return np.column_stack((np.full_like(y, body_force), np.zeros_like(y), np.zeros_like(y)))


def parabolic_profile(y):
    u = body_force / (2.0 * mu) * y * (height - y)
    return np.column_stack((u, np.zeros_like(y), np.zeros_like(y)))


model = IncompressibleNavierStokes(domain=geometry.volume, material=fluid, body_force=forcing)
walls = geometry.boundaries.y_min | geometry.boundaries.y_max
developed = (geometry.boundaries.x_min | geometry.boundaries.x_max
             | geometry.boundaries.z_min | geometry.boundaries.z_max)
problem = Problem(
    name="Poiseuille channel flow",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        NoSlipWall(boundary=walls, physics=model),
        FixedVelocity(boundary=developed, physics=model, value=parabolic_profile),
    ),
    constraints=(MeanValue(model.pressure, 0.0),),
    initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),),
)
simulation = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=TaylorHood()),
        nonlinear_solver=Newton(
            linear_solver=SparseDirect(), absolute_tolerance=1e-11, relative_tolerance=0.0,
        ),
        initial_guess={model.velocity: (0.0, 0.0, 0.0)},
    ),
)
result = simulation.solve_stationary()


# ## Comparison with the parabolic profile
#
# We sample the streamwise velocity along a wall-normal line at mid-channel and
# compare with the closed-form parabola. Taylor-Hood velocities are quadratic, so
# the exact parabola is represented essentially to solver precision.

y_samples = np.linspace(0.0, height, 41)
line = np.column_stack((np.full_like(y_samples, length / 2),
                        y_samples,
                        np.full_like(y_samples, span / 2)))
u_numeric = np.asarray(result.field(model.velocity).evaluate(line))[:, 0]
u_exact = body_force / (2.0 * mu) * y_samples * (height - y_samples)

max_abs_error = float(np.max(np.abs(u_numeric - u_exact)))
rel_error = max_abs_error / u_max_reference
u_max_numeric = float(np.max(u_numeric))
print(f"Maximum velocity: FEM {u_max_numeric:.5f} m/s,  reference {u_max_reference:.5f} m/s")
print(f"Maximum absolute profile error: {max_abs_error:.3e} m/s  ({100*rel_error:.2e} % of u_max)")


# ## Figure

plots.line(points={"OneSimulation 3D FEM": (u_numeric * 1000, y_samples * 1000)},
           reference={"parabolic reference": (u_exact * 1000, y_samples * 1000)}, hlines=[0.0, height * 1000],
           xlabel="streamwise velocity u [mm/s]", ylabel="wall-normal position y [mm]",
           title="Poiseuille channel flow: velocity profile between no-slip plates")
plots.show()


# ## Going further
#
# Replace the constant body force with an explicit pressure difference between the
# inlet and outlet faces (a traction boundary condition) and recover the same
# parabola. Increase the forcing until the Reynolds number leaves the creeping
# regime; the fully developed profile stays parabolic, but entrance effects would
# appear in a longer channel with a uniform inlet. Refine the mesh across the gap
# and confirm the error remains at solver precision because the quadratic elements
# represent the parabola exactly.
