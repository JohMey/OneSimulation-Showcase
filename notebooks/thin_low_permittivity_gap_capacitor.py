"""OneSimulation example: a thin low-permittivity gap in a dielectric-filled capacitor.
Requires the installed, activated OneSimulation runtime.

A capacitor is filled with a dielectric of relative permittivity 20 that contains a thin circular
air gap. The gap is once resolved by the mesh and once replaced by a thin dielectric layer on its
mid-line, across which the potential jumps. Both are compared for four gap widths, together with
the exact solution for a coated cylinder.
"""

# # Thin low-permittivity gap in a dielectric capacitor
#
# Delaminations, air gaps and thin adhesive films in insulation have a much lower permittivity than
# the material around them. The field crowds into such a gap, and the potential drops across it. A
# **thin dielectric layer** replaces the gap by a face across which the potential jumps,
#
# $$D_n=\frac{\varepsilon_2}{d}\,\big(V_1-V_2\big),$$
#
# while the small displacement flux along the gap is carried with $\varepsilon_2 d$; the mesh no
# longer needs to resolve the gap.
#
# Two parallel plates 80 mm apart, 80 mm wide, hold a voltage of 1 V across a dielectric of
# $\varepsilon_{r1}=20$. In the middle a circular gap of outer radius $b=5$ mm and width $d$ is
# filled with a material of relative permittivity $\varepsilon_{r2}$; inside the circle is the
# dielectric again. The electrostatic potential solves
#
# $$\nabla\cdot(\varepsilon\,\nabla V)=0 .$$
#
# | edge | condition |
# |---|---|
# | `terminal`, upper plate | $V=1$ V |
# | `ground`, lower plate | $V=0$ |
# | `insulated`, left and right | $D_n=0$ |
#
# Far from the gap the field is $E_0=12.5$ V/m. Two models are solved for an air gap
# ($\varepsilon_{r2}=1$) of width $d=0.1,\;0.25,\;0.5,\;1$ mm: the **resolved model** meshes the
# gap as its own region, the **thin model** puts the dielectric layer on the circle $r=b-d/2$ in the
# middle of the gap. The thin model is a first-order approximation in $d/b$: it keeps the
# dielectric in the place of the gap, so its error grows in proportion to the width of the gap.
#
# The space between the plates and the gap come from the CAD files
# `thin_low_permittivity_gap_capacitor*.step`; the script `thin_low_permittivity_gap_capacitor_geometry.py`
# builds them with the CAD objects of OneSimulation. The two-dimensional model is their mid-plane
# section.
#
# **Reference.** A coated cylinder in a uniform field of an unbounded medium has a uniform field
# inside (Jackson, *Classical Electrodynamics*, 3rd ed., problem 4.10). With core and surroundings of
# $\varepsilon_1$ and a shell of $\varepsilon_2$ between $a=b-d$ and $b$,
#
# $$\frac{E_\text{in}}{E_0}=\frac{4\,\varepsilon_1\varepsilon_2}{(\varepsilon_1+\varepsilon_2)^2-(\varepsilon_2-\varepsilon_1)^2\,a^2/b^2}.$$
#
# The low-permittivity gap shields the inside as well. The plates are 16 gap radii apart, so they
# change the field inside the circle by about a percent.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import regionwise
from onesim.meshing import TriangularMesher, interface
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import (
    FixedPotential,
    PoissonEquation,
    ThinDielectricLayer,
    ZeroNormalElectricDisplacement,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns each part, `section` its mid-plane cross-section. The resolved meshes follow the
# gap with elements of half its width; the thin model needs only the circle of the mid-line, with
# elements of 0.6 mm around it.

name = "thin_low_permittivity_gap_capacitor"
mid = lambda part: load(f"{name}{part}.step").section((0, 0, 1), origin=(0, 0, 0.0025))
gap = mid("")
edges = gap.boundaries
radius, walls = 0.005, (100, 250, 500, 1000)
meshes = {}
for micrometres in walls:
    thickness = 1e-6 * micrometres
    meshes[micrometres, "resolved"] = TriangularMesher(
        target_size=0.004,
        embedded_regions=(
            mid(f"_core_{micrometres}").named("core"),
            mid(f"_wall_{micrometres}").named("wall"),
        ),
        region_sizes={"wall": thickness / 2, "core": 0.0008},
    ).generate(gap)
    meshes[micrometres, "thin layer"] = TriangularMesher(
        target_size=0.004,
        embedded_regions=(mid(f"_midline_{micrometres}").named("core"),),
        region_sizes={"core": 0.0006},
    ).generate(gap)

fig = plots.figure(1, 3, size=(18, 6))
plots.geometry(
    [
        load(f"{name}.step").named("dielectric between the plates"),
        load(f"{name}_wall_1000.step").named("gap, 1 mm"),
    ],
    labels=False,
    camera=plots.Camera(elevation=55, azimuth=-70),
    title="Dielectric with the circular gap",
    ax=fig[0],
)
plots.mesh(
    meshes[250, "resolved"],
    groups=("all",),
    legend=False,
    title=f"Resolved 0.25 mm gap: {meshes[250, 'resolved'].cell_count} triangles",
    ax=fig[1],
)
plots.mesh(
    meshes[250, "thin layer"],
    groups=("all",),
    legend=False,
    title=f"Thin dielectric layer: {meshes[250, 'thin layer'].cell_count} triangles",
    ax=fig[2],
)
plots.show()

# ## Models
#
# `regionwise` gives the gap of the resolved model its permittivity; `ThinDielectricLayer` acts on
# the faces between the two regions of the thin mesh, which `interface` returns.

epsilon0, bulk, er = 8.854187817e-12, 20.0, 1.0
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
models, results = {}, {}
for (micrometres, kind), mesh in meshes.items():
    law = (
        regionwise({"wall": er * epsilon0}, default=bulk * epsilon0, unit="F/m")
        if kind == "resolved"
        else regionwise((), default=bulk * epsilon0, unit="F/m")
    )
    model = PoissonEquation(domain=gap.domain(), permittivity=law)
    layer = (
        (
            ThinDielectricLayer(
                physics=model,
                boundary=interface(mesh, "background_core"),
                thickness=1e-6 * micrometres,
                permittivity=er * epsilon0,
            ),
        )
        if kind != "resolved"
        else ()
    )
    problem = Problem(
        name=f"Capacitor with a gap, {kind}",
        geometry=gap,
        physics=(model,),
        boundary_conditions=(
            FixedPotential(boundary=edges.terminal, physics=model, value=1.0),
            FixedPotential(boundary=edges.ground, physics=model, value=0.0),
            ZeroNormalElectricDisplacement(boundary=edges.insulated, physics=model),
        )
        + layer,
    )
    models[micrometres, kind] = model
    results[micrometres, kind] = Simulation(
        problem=problem, mesh=mesh, numerics=numerics
    ).solve_stationary()

# ## Results
#
# The field in the centre of the circle against the coated-cylinder formula, and the capacitance the
# gap takes from the filled capacitor ($\varepsilon_1\,w/h=177.08$ pF/m per metre of depth), from
# the charge on the upper plate.

exact = (
    lambda d: 4
    * bulk
    * er
    / ((bulk + er) ** 2 - (er - bulk) ** 2 * (radius - d) ** 2 / radius**2)
)
E0, empty = 1.0 / gap.size[1], bulk * epsilon0 * gap.size[0] / gap.size[1]
field = {
    key: -result.field(models[key].electric_field).evaluate(np.array([[0.0, 0.0]]))[
        0, 1
    ]
    / E0
    for key, result in results.items()
}
added = {
    key: result.field(models[key].potential).reaction(edges.terminal) - empty
    for key, result in results.items()
}
print(
    "gap [mm]    d/b    E_in/E_0 exact   resolved   thin layer   thin vs resolved   change of C' resolved [fF/m]   thin layer [fF/m]"
)
for micrometres in walls:
    a, b = field[micrometres, "resolved"], field[micrometres, "thin layer"]
    print(
        f"{micrometres / 1e3:9.2f} {micrometres / 5e3:6.2f} {exact(1e-6 * micrometres):16.4f} {a:10.4f} {b:12.4f} {100 * (b / a - 1):+16.1f} % "
        f"{1e15 * added[micrometres, 'resolved']:30.2f} {1e15 * added[micrometres, 'thin layer']:19.2f}"
    )

# For a gap of a fiftieth of the radius the thin layer gives the field of the resolved gap within a
# few percent on a fraction of the elements, with the potential jumping across the circle; at a
# fifth of the radius the deviation is about 20 %. It falls in proportion to $d/b$, as expected of a
# first-order thin-layer model.
#
# ## Figure
#
# The potential around the 0.25 mm gap in both models, and the potential along the vertical line
# from the centre upward for the 0.25 mm and the 1 mm gap.

fig = plots.figure(1, 3, size=(21, 6))
for panel, kind in enumerate(("resolved", "thin layer")):
    plots.surface(
        results[250, kind].field(models[250, kind].potential),
        component=lambda V, position: 1e3 * V,
        limits=(350, 650),
        contours=13,
        subdivisions=2,
        xlimits=(-0.012, 0.012),
        ylimits=(-0.012, 0.012),
        xlabel="x [m]",
        ylabel="y [m]",
        label="V [mV]",
        title=f"Potential, {kind}, 0.25 mm gap",
        ax=fig[panel],
    )
y = np.linspace(0.0, 0.012, 601)
line = np.column_stack((np.zeros_like(y), y))
marks = np.column_stack((np.zeros(40), np.linspace(0.00015, 0.01185, 40)))
plots.line(
    points={
        f"thin layer, d = {m / 1e3:g} mm": (
            1e3 * marks[:, 1],
            1e3
            * results[m, "thin layer"]
            .field(models[m, "thin layer"].potential)
            .evaluate(marks),
        )
        for m in (250, 1000)
    },
    reference={
        f"resolved, d = {m / 1e3:g} mm": (
            1e3 * y,
            1e3
            * results[m, "resolved"]
            .field(models[m, "resolved"].potential)
            .evaluate(line),
        )
        for m in (250, 1000)
    },
    vlines=[1e3 * radius],
    xlabel="y [mm]",
    ylabel="V [mV]",
    title="Potential along x = 0, from the centre upward",
    legend="upper left",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Give the gap a permittivity close to that of the dielectric and watch the jump vanish, or let the
# gap width vary along the circle (a callable of the position) to model a partial delamination.
