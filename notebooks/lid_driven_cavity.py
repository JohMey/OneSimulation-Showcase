"""OneSimulation example: lid-driven cavity flow at Reynolds number 100.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the tabulated Ghia-Ghia-Shin (1982)
benchmark for the velocity along the vertical centreline.
"""


# # Lid-driven cavity at Re = 100
#
# An incompressible Newtonian fluid fills a square cavity of side $L$. The top lid
# slides in its own plane at constant speed $U$ while the three remaining walls are
# fixed (no-slip). The Reynolds number is $Re=\rho U L/\mu = 100$. This recirculating
# flow has no closed-form solution and is a classic verification case for
# incompressible Navier-Stokes solvers.
#
# OneSimulation solves the full stationary Navier-Stokes system
# $\rho(u\cdot\nabla)u=-\nabla p+\mu\nabla^2 u$ with inf-sup stable Taylor-Hood
# finite elements (quadratic velocity, linear pressure) and a Newton solver. The
# cavity is modelled as a thin slab that is made periodic across its two faces normal
# to $z$, which reproduces the two-dimensional flow exactly while keeping the native
# 3D formulation.
#
# **Reference.** We compare the horizontal velocity $u_x$ sampled along the vertical
# centreline $x=L/2$ with the tabulated benchmark of Ghia, Ghia & Shin (1982),
# "High-Re solutions for incompressible flow using the Navier-Stokes equations and a
# multigrid method", J. Comput. Phys. 48, 387-411, Table I, Re = 100.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.materials import NewtonianFluid
from onesim.meshing import StructuredTetrahedralMesher
from onesim.meshing.generation.declarations import PeriodicPair
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import FixedVelocity, IncompressibleNavierStokes, NoSlipWall
from onesim.simulation import Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.simulation.initial_conditions.values import InitialValue
from onesim.simulation.problem_definition.boundaries import PeriodicBoundary

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The cavity spans $L\times L$ in the $x$-$y$ plane; the lid lies on the top face
# (y_max). The thin $z$ extent carries a periodic pair, so the resolved flow is
# independent of $z$ and therefore two-dimensional. The density and viscosity are
# chosen so that $Re = \rho U L/\mu = 100$.

side = 1.0            # m, cavity side length L
span = 0.0625         # m, thin out-of-plane extent (made periodic in z)
lid_speed = 1.0       # m/s, speed U of the sliding lid
rho = 1.0             # kg/m^3, density
mu = 0.01             # Pa·s, dynamic viscosity

reynolds = rho * lid_speed * side / mu
print(f"U = {lid_speed:.2f} m/s,  L = {side:.2f} m,  Re = {reynolds:.0f}")


# ## Model
#
# The left, right and bottom faces are no-slip walls; the top face is the moving lid;
# the two faces normal to $z$ form one periodic pair. Every one of the six boundary
# faces therefore carries an explicit condition. The lid speed is uniform except for
# a linear taper to zero inside the single corner cell on each side: this removes the
# mathematically singular velocity jump at the two top corners (where the lid meets
# the fixed walls) so the prescribed trace is single-valued there, while leaving the
# lid uniform over the interior and unchanged along the sampled centreline.

cells_per_side = 32
geometry = Box(length=side, width=side, height=span)
mesh = StructuredTetrahedralMesher(
    cells=(cells_per_side, cells_per_side, 1)).generate(geometry)
taper = side / cells_per_side
print("Mesh:", mesh.cell_count, "tetrahedra")

fluid = NewtonianFluid(density=rho, dynamic_viscosity=mu)


def lid_velocity(x):
    # Uniform lid speed U in the x-direction, tapered to zero within the corner cell.
    ramp = np.clip(np.minimum(x, side - x) / taper, 0.0, 1.0)
    return np.column_stack((lid_speed * ramp, np.zeros_like(x), np.zeros_like(x)))


model = IncompressibleNavierStokes(domain=geometry.volume, material=fluid)
fixed_walls = (geometry.boundaries.x_min | geometry.boundaries.x_max
               | geometry.boundaries.y_min)
spanwise = PeriodicPair(first=geometry.boundaries.z_min, second=geometry.boundaries.z_max)
problem = Problem(
    name="Lid-driven cavity",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        NoSlipWall(boundary=fixed_walls, physics=model),
        FixedVelocity(boundary=geometry.boundaries.y_max, physics=model, value=lid_velocity),
        PeriodicBoundary(pair=spanwise, physics=model),
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
            linear_solver=SparseDirect(), absolute_tolerance=1e-9, relative_tolerance=1e-10,
            maximum_iterations=40,
        ),
        initial_guess={model.velocity: (0.0, 0.0, 0.0)},
    ),
)
result = simulation.solve_stationary()


# ## Comparison with the Ghia-Ghia-Shin benchmark
#
# We sample the horizontal velocity $u_x$ along the vertical centreline $x=L/2$ at the
# tabulated benchmark heights and compare with the Re = 100 reference values.

benchmark_y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
benchmark_u = np.array([
    0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
    -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000])

sample_points = np.column_stack((
    np.full_like(benchmark_y, side / 2.0),
    benchmark_y * side,
    np.full_like(benchmark_y, span / 2.0)))
u_numeric = np.asarray(result.field(model.velocity).evaluate(sample_points))[:, 0] / lid_speed

max_abs_error = float(np.max(np.abs(u_numeric - benchmark_u)))
print("Horizontal velocity along the vertical centreline (u_x / U):")
for y, u_ref, u_fem in zip(benchmark_y, benchmark_u, u_numeric):
    print(f"  y/L = {y:.4f}   benchmark {u_ref:+.5f}   OneSimulation {u_fem:+.5f}")
print(f"Maximum absolute deviation from the Ghia benchmark: {max_abs_error:.4f} (of U)")

assert max_abs_error < 0.02, "Centreline velocity deviates more than the benchmark scatter."
# The lid drags the fluid forward near the top and the primary vortex drives a
# reversed flow over the lower half of the cavity.
assert u_numeric[-1] > 0.9, "The lid must carry the fluid forward at the top."
assert np.min(u_numeric) < -0.15, "A primary recirculation must reverse the lower flow."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
ax.grid(alpha=0.25)
ax.axvline(0.0, color="#999", lw=1)
ax.plot(benchmark_u, benchmark_y, "o", color="#214D43", ms=5, label="Ghia et al. (1982), Re=100")
ax.plot(u_numeric, benchmark_y, "-", color="#A85539", lw=1.8, label="OneSimulation FEM")
ax.set(xlabel="Horizontal velocity u_x / U", ylabel="Vertical position y / L",
       title="Lid-driven cavity at Re = 100: velocity on the vertical centreline")
ax.legend()
plt.show()


# ## Going further
#
# Raise the Reynolds number toward 400 or 1000 and refine the mesh; the primary
# vortex strengthens, its centre drifts, and secondary corner vortices grow, all of
# which the same benchmark tables document. Replace the periodic spanwise pair with
# no-slip walls to study how a finite-width cavity departs from the two-dimensional
# limit. Sample the vertical velocity along the horizontal centreline to compare with
# the companion benchmark column.
