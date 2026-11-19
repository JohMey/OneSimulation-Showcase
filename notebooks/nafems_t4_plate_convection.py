"""OneSimulation example: two-dimensional steady conduction with convection (NAFEMS T4).
Requires the installed, activated OneSimulation runtime.

A steel plate is held at 100 °C along its base and cooled by convection on two sides. The plate is
read from a STEP file and cut to its cross-section; the temperature at the benchmark point is
compared with the NAFEMS target for linear and quadratic triangles on three meshes.
"""

# # Two-dimensional conduction with convection (NAFEMS T4)
#
# The NAFEMS thermal benchmark T4 is the standard test for steady two-dimensional conduction with
# a convective boundary. A rectangular plate, 0.6 m wide and 1.0 m high, conducts heat from its
# base into two cooled sides:
#
# $$\nabla\cdot(k\,\nabla T)=0,\qquad k=52\ \mathrm{W/(m\,K)}.$$
#
# | side | condition |
# |---|---|
# | base $y=0$ | fixed temperature $T=100$ °C |
# | left side $x=0$ | insulated, $q_n=0$ |
# | right side $x=0.6$ m and top $y=1.0$ m | convection $-k\,\partial_nT=h\,(T-T_\infty)$, $h=750$ W/(m² K), $T_\infty=0$ °C |
#
# The corner where the base meets the cooled right side is singular (the fixed temperature meets a
# strong convective sink), so the field changes fast near the benchmark point.
#
# The plate comes from the CAD file `nafems_t4_plate_convection.step`; the script
# `nafems_t4_plate_convection_geometry.py` builds it with the CAD objects of OneSimulation and
# names the heated base, the insulated side and the cooled sides. The two-dimensional model is the
# mid-plane section of the plate.
#
# **Reference.** NAFEMS, *The Standard NAFEMS Benchmarks* (TNSB, Rev. 3, 1990), test T4: the
# temperature at point E $=(0.6,\,0.2)$ m on the cooled side is **18.25 °C**.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import (
    ConvectiveHeatFlux,
    FixedTemperature,
    HeatConduction,
    HeatFlux,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the plate with its named faces; `section` cuts it at mid-depth and returns the
# planar region, whose sides keep the names of the faces they were cut from. Three triangular
# meshes halve the cell size from one to the next and are refined towards the cooled right side.

plate = load("nafems_t4_plate_convection.step")
faces = plate.boundaries
region = plate.section((0, 0, 1), origin=(0, 0, plate.size[2] / 2))
sides = region.boundaries
sizes = [0.2, 0.1, 0.05]
meshes = [
    TriangularMesher(
        target_size=size, grading=0.3, local_sizes={"x_max": size / 3}
    ).generate(region)
    for size in sizes
]

fig = plots.figure(1, 2, size=(13, 6))
plots.geometry(
    plate,
    selection={
        "base, 100 °C": faces.base,
        "insulated": faces.insulated,
        "cooled": faces.cooled,
    },
    labels=False,
    camera=plots.Camera(elevation=35, azimuth=-110),
    title="Plate with its named faces",
    ax=fig[0],
)
plots.mesh(
    meshes[1],
    groups=("all",),
    legend=False,
    title=f"Middle mesh: {meshes[1].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The steel conducts with 52 W/(m K); only the conductivity matters for the steady field. The same
# problem is solved with linear and with quadratic triangles on the three meshes.

steel = IsotropicSolid(
    density=7850.0,
    specific_heat_capacity=460.0,
    thermal_conductivity=52.0,
    youngs_modulus=2.1e11,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=1.2e-5,
)
model = HeatConduction(domain=region.domain(), material=steel, heating=0.0)
problem = Problem(
    name="NAFEMS T4",
    geometry=region,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=sides.y_min, physics=model, value=373.15),
        HeatFlux(boundary=sides.x_min, physics=model, value=0.0),
        ConvectiveHeatFlux(
            boundary=sides.x_max | sides.y_max,
            physics=model,
            heat_transfer_coefficient=750.0,
            ambient_temperature=273.15,
        ),
    ),
)
fields = {
    (order, size): Simulation(
        problem=problem,
        mesh=mesh,
        numerics=Numerics(
            spatial=FiniteElement(element=Lagrange(order=order)),
            linear_solver=SparseDirect(),
        ),
    )
    .solve_stationary()
    .field(model.temperature)
    for order in (1, 2)
    for size, mesh in zip(sizes, meshes)
}
field = fields[2, sizes[-1]]

# ## Results
#
# The temperature at point E for every run, and the energy balance (heat entering at the base, heat
# leaving by convection).

point_e = {
    key: float(f.evaluate([[0.6, 0.2]])[0]) - 273.15 for key, f in fields.items()
}
print("element     mesh size [m]   triangles   T(E) [°C]   deviation from 18.25 °C")
for (order, size), value in point_e.items():
    print(
        f"{'quadratic' if order == 2 else 'linear':10s} {size:13.3f} {meshes[sizes.index(size)].cell_count:11d} "
        f"{value:11.3f} {value - 18.25:+12.3f} K"
    )
print(
    f"heat per metre depth: in at the base {-field.boundary_flux(sides.y_min):.1f} W/m, "
    f"out by convection {field.boundary_flux(sides.x_max | sides.y_max):.1f} W/m"
)

# ## Figure
#
# The temperature field with isotherms, the temperature along the cooled right side with the
# benchmark point, and the convergence of $T(E)$ for both element orders.

y = np.linspace(0, 1, 201)
right = {
    f"{'quadratic' if order == 2 else 'linear'}, h = {size} m": (
        y,
        f.evaluate(np.column_stack((np.full_like(y, 0.6), y))).ravel() - 273.15,
    )
    for (order, size), f in fields.items()
    if size != sizes[1]
}

fig = plots.figure(1, 3, size=(20, 6))
plots.surface(
    field,
    component=lambda T, position: T - 273.15,
    limits=(0, 100),
    contours=11,
    subdivisions=3,
    xlabel="x [m]",
    ylabel="y [m]",
    label="T [°C]",
    title="Temperature, quadratic, finest mesh",
    ax=fig[0],
)
plots.line(
    y=right,
    points={"NAFEMS target at E": ([0.2], [18.25])},
    xlimits=(0, 1),
    xlabel="height y on the cooled side [m]",
    ylabel="T [°C]",
    title="Along the cooled side x = 0.6 m",
    ax=fig[1],
)
plots.line(
    y={
        ("quadratic" if order == 2 else "linear"): (
            sizes,
            [point_e[order, size] for size in sizes],
        )
        for order in (1, 2)
    },
    hlines={"NAFEMS target 18.25 °C": 18.25},
    markers=True,
    scale="logx",
    xlabel="mesh size [m]",
    ylabel="T(E) [°C]",
    title="Mesh study at point E",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Change the plate in `nafems_t4_plate_convection_geometry.py` and write the STEP file again, or
# raise the film coefficient: as $h\to\infty$ the cooled sides approach a fixed 0 °C and the
# solution becomes the classical Fourier series of a rectangle. Replace the convection by
# `RadiativeHeatFlux` for a furnace wall that loses heat by radiation.
