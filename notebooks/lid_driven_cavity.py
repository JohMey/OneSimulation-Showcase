"""OneSimulation example: lid-driven cavity flow at Reynolds number 100.
Requires the installed, activated OneSimulation runtime.

The cavity is read from a STEP file and meshed with tetrahedra; the velocity along the vertical
centre line of the Navier-Stokes solution is checked against the tabulated benchmark of Ghia,
Ghia and Shin (1982).
"""


# # Lid-driven cavity at Re = 100
#
# An incompressible Newtonian fluid fills a square cavity of side $L$. The top lid
# slides in its own plane at constant speed $U$ while the three remaining walls are
# fixed (no-slip). The Reynolds number is $Re=\rho U L/\mu = 100$. This recirculating
# flow has no closed-form solution and is a classic verification case for
# incompressible Navier-Stokes solvers.
#
# The cavity comes from the CAD file `lid_driven_cavity.step`, a thin slab whose face `y_max` is
# the lid. The script `lid_driven_cavity_geometry.py` builds it with the CAD objects of
# OneSimulation. OneSimulation solves the stationary Navier-Stokes system
# $\rho(u\cdot\nabla)u=-\nabla p+\mu\nabla^2 u$ with Taylor-Hood finite elements (quadratic
# velocity, linear pressure) and a Newton solver. The slab is periodic across its two faces normal
# to $z$, which reproduces the two-dimensional flow.
#
# **Reference.** The horizontal velocity $u_x$ along the vertical centre line $x=L/2$ in the
# benchmark of Ghia, Ghia & Shin (1982), "High-Re solutions for incompressible flow using the
# Navier-Stokes equations and a multigrid method", J. Comput. Phys. 48, 387-411, Table I, Re = 100.

import numpy as np

from onesim.cad.exchange import load
from onesim.materials import NewtonianFluid
from onesim.meshing import PeriodicPair, StructuredTetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import FixedVelocity, IncompressibleNavierStokes, NoSlipWall
from onesim.simulation import PeriodicBoundary, Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import theme, plots

theme.set()

# ## Geometry and mesh

cavity = load("lid_driven_cavity.step")
faces = cavity.boundaries
side, _, span = cavity.size
cells = 32
mesh = StructuredTetrahedralMesher(cells=(cells, cells, 1)).generate(cavity)

view = plots.Camera(elevation=62, azimuth=-90)
fig = plots.figure(1, 2, size=(13, 6.5))
plots.geometry(cavity, selection={"lid": faces.y_max, "fixed walls": faces.x_min | faces.x_max | faces.y_min}, labels=False,
               camera=view, title="Cavity with its lid and fixed walls", ax=fig[0])
plots.mesh(mesh, groups=("all",), legend=False, camera=view, title=f"{mesh.cell_count} tetrahedra", ax=fig[1])
plots.show()

# ## Model
#
# Density and viscosity give $Re=100$. The lid speed is uniform except for a linear ramp to zero
# inside the corner cell on each side, so the velocity is single-valued where the lid meets the
# fixed walls.

lid_speed = 1.0
fluid = NewtonianFluid(density=1.0, dynamic_viscosity=0.01)
model = IncompressibleNavierStokes(domain=cavity.volume, material=fluid)
problem = Problem(name="Lid-driven cavity", geometry=cavity, physics=(model,), boundary_conditions=(
    NoSlipWall(boundary=faces.x_min | faces.x_max | faces.y_min, physics=model),
    FixedVelocity(boundary=faces.y_max, physics=model, value=lambda x: np.column_stack((
        lid_speed * np.clip(np.minimum(x, side - x) * cells / side, 0.0, 1.0), 0 * x, 0 * x))),
    PeriodicBoundary(pair=PeriodicPair(first=faces.z_min, second=faces.z_max), physics=model)),
    constraints=(MeanValue(model.pressure, 0.0),), initial_conditions=(InitialValue(model.velocity, (0.0, 0.0, 0.0)),))
numerics = Numerics(spatial=FiniteElement(element=TaylorHood()), initial_guess={model.velocity: (0.0, 0.0, 0.0)},
                    nonlinear_solver=Newton(linear_solver=SparseDirect(), absolute_tolerance=1e-9, relative_tolerance=1e-10,
                                            maximum_iterations=40))
velocity = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary().field(model.velocity)

# ## Comparison with the Ghia-Ghia-Shin benchmark
#
# The horizontal velocity on the vertical centre line at the heights of the benchmark table.

benchmark_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                        0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
benchmark_u = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
                        -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000])
u = velocity.evaluate(np.column_stack((np.full_like(benchmark_y, side / 2), benchmark_y * side,
                                       np.full_like(benchmark_y, span / 2))))[:, 0] / lid_speed

print(f"Reynolds number: {fluid.density * lid_speed * side / fluid.dynamic_viscosity:.0f}")
print("   y / L    benchmark   simulation")
for height, reference, value in zip(benchmark_y, benchmark_u, u):
    print(f"{height:8.4f} {reference:+12.5f} {value:+12.5f}")
print(f"largest deviation from the benchmark: {np.abs(u - benchmark_u).max():.4f} of the lid speed")

# ## Figure

height = np.linspace(0.0, 1.0, 201)
profile = velocity.evaluate(np.column_stack((np.full_like(height, side / 2), height * side, np.full_like(height, span / 2))))[:, 0] / lid_speed
plots.line(y={"simulation": (profile, height)}, points={"Ghia et al. (1982), Re = 100": (benchmark_u, benchmark_y)}, vlines=0.0,
           xlabel="horizontal velocity u_x / U", ylabel="vertical position y / L",
           title="Lid-driven cavity at Re = 100: velocity on the vertical centre line")
plots.show()

# ## Going further
#
# Raise the Reynolds number toward 400 or 1000 and refine the mesh; the primary vortex
# strengthens, its centre drifts, and secondary corner vortices grow, all of which the same
# benchmark tables document. Make the slab as deep as the cavity is wide in
# `lid_driven_cavity_geometry.py`, write the STEP file again and replace the periodic pair by
# no-slip walls to see how a cavity of finite width departs from the two-dimensional flow.
