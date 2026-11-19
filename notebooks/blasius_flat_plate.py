"""OneSimulation example: laminar boundary layer on a flat plate (Blasius).
Requires the installed, activated OneSimulation runtime.

A uniform stream flows along a flat plate. The air above the plate is read from a STEP file and
cut to its cross-section; velocity profiles, boundary-layer thickness and skin friction of the
two-dimensional Navier-Stokes simulation are compared with the Blasius similarity solution.
"""

# # Laminar flat-plate boundary layer (Blasius)
#
# A uniform stream of speed $U$ flows along a thin flat plate. Viscosity brings the fluid to rest
# on the plate, and a thin boundary layer grows along it. Blasius (1908) found the similarity
# solution of the boundary-layer equations: with $\eta = y\sqrt{U/(\nu x)}$,
#
# $$f''' + \tfrac{1}{2}\,f\,f'' = 0,\qquad f(0)=f'(0)=0,\qquad f'(\infty)=1,\qquad \frac{u}{U}=f'(\eta) .$$
#
# The slab of air above the plate comes from the CAD file `blasius_flat_plate.step`: the plate is
# the face `y_min`, the inflow `x_min`, the outflow `x_max` and the far field `y_max`. The script
# `blasius_flat_plate_geometry.py` builds it with the CAD objects of OneSimulation; the model is
# its cross-section.
#
# | side | condition |
# |---|---|
# | `x_min` (inflow) | Blasius profile $(u, v)$ at 0.2 m from the leading edge |
# | `y_min` (plate) | no slip |
# | `x_max`, `y_max` (outflow, far field) | traction free |
#
# **Reference.** From the Blasius solution (Schlichting and Gersten, *Boundary-Layer Theory*):
#
# $$\delta_{99}=5.0\,\frac{x}{\sqrt{Re_x}},\qquad c_f=\frac{0.664}{\sqrt{Re_x}},\qquad Re_x=\frac{Ux}{\nu}.$$
#
# The simulation solves the full Navier-Stokes equations, without the boundary-layer
# approximation. Its domain starts 0.2 m behind the leading edge, where the Blasius profile is
# the inflow, and follows the layer for 1 m.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import NewtonianFluid
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import (
    FixedVelocity,
    FlowTraction,
    NoSlipWall,
    PlanarIncompressibleNavierStokes,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme
from scipy.integrate import solve_ivp

theme.set()

# ## Geometry and mesh
#
# `section` cuts the slab at half its span; the cross-section keeps the names of the faces. The
# triangles are 2 mm high on the plate and grow with the distance from it up to 2 cm.

slab = load("blasius_flat_plate.step")
section = slab.section((0, 0, 1), origin=(0, 0, slab.size[2] / 2))
faces = section.boundaries
plate_length, height = section.size[:2]
mesh = TriangularMesher(
    target_size=0.02, size_field=lambda p: np.minimum(0.02, 0.002 + 0.1 * p[:, 1])
).generate(section)

fig = plots.figure(2, 1, size=(15, 8))
plots.geometry(
    slab,
    selection={"plate": slab.boundaries.y_min, "inflow": slab.boundaries.x_min},
    labels=False,
    camera=plots.Camera(elevation=62, azimuth=-90),
    title="Slab of air above the plate",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"Cross-section: {mesh.cell_count} triangles, graded towards the plate",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The plate is a no-slip wall, the inflow carries the Blasius profile, and the top and the outflow
# are open. Taylor-Hood elements (quadratic velocity, linear pressure) with a Newton solver.

start, stream = 0.2, 1.0
air = NewtonianFluid(density=1.0, dynamic_viscosity=1.0e-4)
viscosity = air.dynamic_viscosity / air.density
similarity = solve_ivp(
    lambda eta, f: (f[1], f[2], -0.5 * f[0] * f[2]),
    (0, 10),
    (0, 0, 0.332057),
    dense_output=True,
    rtol=1e-10,
    atol=1e-12,
).sol


def blasius(x, y):
    """Velocity components u and v of the Blasius solution at the distance x from the leading edge."""
    eta = np.minimum(y * np.sqrt(stream / (viscosity * x)), 10.0)
    f, slope, _ = similarity(eta)
    return stream * slope, 0.5 * np.sqrt(viscosity * stream / x) * (eta * slope - f)


model = PlanarIncompressibleNavierStokes(domain=section.domain(), material=air)
problem = Problem(
    name="Blasius boundary layer",
    geometry=section,
    physics=(model,),
    boundary_conditions=(
        NoSlipWall(boundary=faces.y_min, physics=model),
        FixedVelocity(
            boundary=faces.x_min,
            physics=model,
            value=lambda y: np.column_stack(blasius(start, y)),
        ),
        FlowTraction(
            boundary=faces.x_max | faces.y_max, physics=model, value=(0.0, 0.0)
        ),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=TaylorHood()),
    initial_guess={model.velocity: (stream, 0.0)},
    nonlinear_solver=Newton(
        linear_solver=SparseDirect(),
        absolute_tolerance=1e-9,
        relative_tolerance=1e-10,
        maximum_iterations=40,
    ),
)
velocity = (
    Simulation(problem=problem, mesh=mesh, numerics=numerics)
    .solve_stationary()
    .field(model.velocity)
)

# ## Results
#
# The velocity is sampled on a grid of stations along the plate. The thickness $\delta_{99}$ is
# where $u$ reaches 99 % of the stream; the skin friction follows from the slope of $u$ at the wall.

x = start + np.linspace(0.1, 0.9, 17) * plate_length
y = np.linspace(0, 0.1, 201)
grid = np.array([[position - start, level] for position in x for level in y])
u = velocity.evaluate(grid)[:, 0].reshape(len(x), len(y))
reynolds = stream * x / viscosity
thickness = np.array([np.interp(0.99 * stream, profile, y) for profile in u])
friction = (
    2
    * air.dynamic_viscosity
    * (4 * u[:, 5] - u[:, 10])
    / (2 * y[5])
    / (air.density * stream**2)
)

print(" x [m]    δ99 [mm]   Blasius    c_f·10³   Blasius")
for index in range(0, 17, 4):
    print(
        f"{x[index]:6.2f} {1e3 * thickness[index]:10.2f} {5e3 * x[index] / np.sqrt(reynolds[index]):9.2f}"
        f" {1e3 * friction[index]:10.3f} {664 / np.sqrt(reynolds[index]):9.3f}"
    )

# ## Figure
#
# Velocity profiles at three stations in the similarity variable, the boundary-layer thickness
# and the skin friction along the plate.

eta = np.linspace(0, 8, 200)
profiles = {
    f"x = {x[i]:.2f} m": (
        u[i, ::4] / stream,
        y[::4] * np.sqrt(stream / (viscosity * x[i])),
    )
    for i in (4, 8, 12)
}

fig = plots.figure(1, 3, size=(16, 5))
plots.line(
    points=profiles,
    reference={"Blasius f'(η)": (similarity(eta)[1], eta)},
    ylimits=(0, 8),
    xlabel="u / U",
    ylabel="η = y √(U / ν x)",
    title="Velocity profiles collapse onto one curve",
    legend="lower right",
    ax=fig[0],
)
plots.line(
    x,
    points={"simulation": 1e3 * thickness},
    reference={"5.0 x / √Re": 5e3 * x / np.sqrt(reynolds)},
    xlabel="distance from the leading edge [m]",
    ylabel="δ99 [mm]",
    title="Boundary-layer thickness",
    ax=fig[1],
)
plots.line(
    x,
    points={"simulation": 1e3 * friction},
    reference={"0.664 / √Re": 664 / np.sqrt(reynolds)},
    xlabel="distance from the leading edge [m]",
    ylabel="c_f × 10³",
    title="Skin friction",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Make the slab longer in `blasius_flat_plate_geometry.py` and write the STEP file again to follow
# the layer to higher Reynolds numbers. Add a pressure gradient through the outer boundary to
# obtain the Falkner-Skan wedge flows, or heat the plate and compare the thermal boundary layer
# with the Pohlhausen solution.
