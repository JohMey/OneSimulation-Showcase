"""OneSimulation example: homogenized permeability of a simple cubic packing of spheres (3D).
Requires the installed, activated OneSimulation runtime.

A quarter of the periodic unit cell of a simple cubic sphere array is read from STEP files for five
solid fractions. The three-dimensional creeping flow through the cell is solved with Taylor-Hood
elements; porosity and permeability of the homogenized medium follow from it and are compared with
the drag coefficients of Zick and Homsy and the dilute limit of Hasimoto, with a mesh study.
"""

# # Homogenized permeability of a simple cubic sphere packing
#
# A macroscopic model of a packed bed, a sintered filter or a granular soil treats the pores as a
# continuum with two properties: the porosity $\varepsilon$ and the permeability $K$ of Darcy's law
# $\langle\mathbf u\rangle=-(K/\mu)\nabla\langle p\rangle$. Both follow from one periodic cell of
# the microstructure. Here the cell of side $L$ holds one sphere of radius $a$, the solid fraction
# is $c=1-\varepsilon=4\pi a^3/(3L^3)$. A uniform pressure gradient $G$ along $x$, applied as a body
# force, drives the creeping flow; then $K=\mu\langle u\rangle/G$ with the superficial velocity
# $\langle u\rangle$, the flow rate per cell cross-section.
#
# Two mirror planes through the sphere centres and two between the spheres cut the cell into
# quarters, so one quarter is enough. It comes from the CAD files
# `periodic_sphere_array_permeability.step` ($c=0.216$) and
# `periodic_sphere_array_permeability_64.step` to `..._450.step`, written by
# `periodic_sphere_array_permeability_geometry.py`.
#
# $$-\nabla p+\mu\nabla^2\mathbf u+G\,\mathbf e_x=0,\qquad\nabla\cdot\mathbf u=0$$
#
# | face | condition |
# |---|---|
# | `x_min` / `x_max` | periodic pair |
# | `y_min`, `y_max`, `z_min`, `z_max` (mirror planes) | impermeable, shear free |
# | `sphere.surface` | no slip |
#
# **Reference.** The drag on each sphere, $F=GL^3$, is usually given relative to Stokes' drag on a
# single sphere in the superficial velocity, $K^*=F/(6\pi\mu a\langle u\rangle)$, so that
# $K=L^3/(6\pi a K^*)$. A. A. Zick and G. M. Homsy, "Stokes flow through periodic arrays of spheres",
# *J. Fluid Mech.* 115, 13-26 (1982), Table 2, simple cubic array: $K^*=2.810$, 4.292, 7.442, 15.4
# and 28.1 for $c=0.064$, 0.125, 0.216, 0.343 and 0.45. Dilute limit (H. Hasimoto, *J. Fluid
# Mech.* 5, 317-328, 1959): $K^*=1/(1-1.7601\,c^{1/3}+c-1.5593\,c^2)$.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import NewtonianFluid
from onesim.meshing import LocalRefinement, PeriodicPair, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import (
    FreeSlipLaw,
    IncompressibleNavierStokes,
    NoSlipWall,
    WallShear,
)
from onesim.results.derived import derived_field
from onesim.simulation import PeriodicBoundary, Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# Tetrahedra of a tenth of the cell side, finer on the sphere; the two periodic faces carry the
# same triangles. The mesh study at $c=0.216$ adds a coarser and a finer mesh.

fractions = (0.064, 0.125, 0.216, 0.343, 0.450)
zick_homsy = {0.064: 2.810, 0.125: 4.292, 0.216: 7.442, 0.343: 15.4, 0.450: 28.1}
cells = {
    fraction: load(f"periodic_sphere_array_permeability_{round(1000 * fraction)}.step")
    for fraction in fractions
}
side = cells[0.216].size[0]
radius = {
    fraction: side * (3 * fraction / (4 * np.pi)) ** (1 / 3) for fraction in fractions
}
cases = [(fraction, 0.10) for fraction in fractions] + [(0.216, 0.14), (0.216, 0.07)]
meshes = {
    (fraction, size): TetrahedralMesher(
        target_size=size * side,
        refinements=(
            LocalRefinement(
                region=cells[fraction].boundaries["sphere.surface"],
                target_size=0.6 * size * side,
            ),
        ),
        periodic_pairs=(
            PeriodicPair(
                first=cells[fraction].boundaries.x_min,
                second=cells[fraction].boundaries.x_max,
            ),
        ),
    ).generate(cells[fraction])
    for fraction, size in cases
}

cell = load("periodic_sphere_array_permeability.step")
view = plots.Camera(elevation=28, azimuth=-60)
fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    cell,
    selection={
        "sphere": cell.boundaries["sphere.surface"],
        "periodic pair": cell.boundaries.x_min | cell.boundaries.x_max,
    },
    labels=False,
    camera=view,
    title="Quarter of the unit cell, c = 0.216",
    ax=fig[0],
)
plots.mesh(
    meshes[0.216, 0.10],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{meshes[0.216, 0.10].cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Water and a pressure gradient of 1 Pa/m keep the Reynolds number far below one. The pressure,
# periodic and fixed only up to a constant, gets the mean value zero.

water, gradient = NewtonianFluid(density=1000.0, dynamic_viscosity=1.0e-3), 1.0
flows = {}
for fraction, size in cases:
    faces = cells[fraction].boundaries
    model = IncompressibleNavierStokes(
        domain=cells[fraction].volume, material=water, body_force=(gradient, 0.0, 0.0)
    )
    problem = Problem(
        name="Sphere array unit cell",
        geometry=cells[fraction],
        physics=(model,),
        boundary_conditions=(
            NoSlipWall(boundary=faces["sphere.surface"], physics=model),
            WallShear(
                boundary=faces.y_min | faces.y_max | faces.z_min | faces.z_max,
                physics=model,
                law=FreeSlipLaw(),
            ),
            PeriodicBoundary(
                pair=PeriodicPair(first=faces.x_min, second=faces.x_max), physics=model
            ),
        ),
        constraints=(MeanValue(model.pressure, 0.0),),
    )
    numerics = Numerics(
        spatial=FiniteElement(element=TaylorHood()),
        initial_guess={model.velocity: (0.0, 0.0, 0.0)},
        nonlinear_solver=Newton(
            linear_solver=SparseDirect(),
            absolute_tolerance=1e-20,
            relative_tolerance=1e-10,
        ),
    )
    flows[fraction, size] = (
        Simulation(problem=problem, mesh=meshes[fraction, size], numerics=numerics)
        .solve_stationary()
        .field(model.velocity)
    )

# ## Results
#
# The superficial velocity is the flow rate through the quarter divided by its volume times its
# length; the porosity is the fluid volume over the cell volume. The flat faces of the tetrahedra
# make the sphere slightly smaller than drawn, which shows in the porosity. Hasimoto's expansion
# holds only for dilute arrays.

volume = side**3 / 4
porosity = {
    fraction: derived_field(
        flows[fraction, 0.10],
        lambda u, points: np.ones(len(points)),
        name="one",
        unit="1",
    )
    .integrate()
    .value
    / volume
    for fraction in fractions
}
permeability = {
    key: water.dynamic_viscosity * velocity.integrate().value[0] / volume / gradient
    for key, velocity in flows.items()
}
drag = {
    (fraction, size): side**3
    / (6 * np.pi * radius[fraction] * permeability[fraction, size])
    for fraction, size in cases
}
hasimoto = lambda c: 1 / (1 - 1.7601 * c ** (1 / 3) + c - 1.5593 * c**2)

print(
    "  c      porosity  exact    K [m²]        K*      Zick & Homsy   deviation   Hasimoto"
)
for fraction in fractions:
    print(
        f"{fraction:6.3f} {porosity[fraction]:9.4f} {1 - fraction:7.4f} {permeability[fraction, 0.10]:12.4e} {drag[fraction, 0.10]:8.3f}"
        f" {zick_homsy[fraction]:10.3f} {100 * (drag[fraction, 0.10] / zick_homsy[fraction] - 1):+10.1f} %"
        + (f" {hasimoto(fraction):9.3f}" if fraction < 0.15 else "    (dilute only)")
    )
print(
    "\nmesh study at c = 0.216: K* = "
    + ", ".join(
        f"{drag[0.216, size]:.4f} ({meshes[0.216, size].cell_count} tetrahedra)"
        for size in (0.14, 0.10, 0.07)
    )
)

# ## Figure
#
# The flow speed on the faces of the quarter cell at $c=0.216$, and the drag coefficient over the
# solid fraction.

c = np.linspace(0.001, 0.12, 100)
fig = plots.figure(1, 2, size=(15, 6))
plots.surface(
    flows[0.216, 0.10],
    component=lambda u, position: 1e6 * np.linalg.norm(u, axis=1),
    limits="robust",
    contours=6,
    subdivisions=2,
    camera=view,
    label="speed [µm/s]",
    title="Speed on the cell faces, c = 0.216",
    ax=fig[0],
)
plots.line(
    c,
    reference={"Hasimoto (1959), dilute": hasimoto(c)},
    y={"simulation": (list(fractions), [drag[f, 0.10] for f in fractions])},
    points={"Zick & Homsy (1982)": (list(zick_homsy), list(zick_homsy.values()))},
    markers=True,
    scale="logy",
    xlimits=(0.0, 0.5),
    xlabel="solid fraction c",
    ylabel="drag coefficient K* = F / (6π μ a U)",
    title="Drag on the spheres of a simple cubic array",
    legend="upper left",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Shift the sphere into the corners and the face centres of the cell in
# `periodic_sphere_array_permeability_geometry.py` to model body-centred and face-centred packings,
# or let the spheres overlap to model a sintered material. Driving the flow along $y$ and $z$ in
# turn gives the full permeability tensor of an anisotropic cell.
