"""OneSimulation example: three-dimensional thermal bridge, EN ISO 10211 case 4.
Requires the installed, activated OneSimulation runtime.

An iron bar pierces an insulation layer and carries heat from the warm to the cold side. The heat
flow and the highest external surface temperature are compared with the reference values of the
standard, once with finite volumes and once with quadratic finite elements on the same meshes.
The wall is read from a STEP file.
"""

# # Thermal bridge in 3D: iron bar through an insulation layer
#
# EN ISO 10211 validates programs for thermal bridges in buildings with four test cases. Case 4 is
# the three-dimensional one: an insulation layer, 1 m × 1 m and 0.2 m thick, is pierced by an iron
# bar of 0.1 m × 0.05 m cross-section that starts flush with the cold surface and sticks 0.4 m
# into the warm room.
#
# | part | conductivity [W/(m K)] |
# |---|---|
# | insulation layer | 0.1 |
# | iron bar | 50 |
#
# The steady heat equation $\nabla\cdot(k\nabla T)=0$ holds. Both sides exchange heat with the air
# through the surface resistance 0.10 m² K/W, i.e. $h=10$ W/(m² K); the room air is at 1 °C, the
# outside air at 0 °C, and the cut edges of the layer are adiabatic. One quarter is computed and
# mirrored about the two symmetry planes.
#
# The quarter comes from the CAD file `thermal_bridge_3d_iso10211.step` with the face groups
# `external`, `internal` and `adiabatic`; `thermal_bridge_3d_iso10211_bar.step` holds the bar, which
# marks the region of the iron. The script `thermal_bridge_3d_iso10211_geometry.py` builds both
# with the CAD objects of OneSimulation.
#
# **Reference.** The standard gives the heat flow 0.540 W and the highest external surface
# temperature 0.805 °C; a program passes within 1 % and 0.005 K.

from dataclasses import replace

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.d3 import Box
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import Geometric, Piecewise, StructuredHexahedralMesher, region_mesh
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.fvm import CentralDifference, FiniteVolume, LeastSquaresGradient
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import ConvectiveHeatFlux, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and materials
#
# `load` returns the wall with its named faces and the bar. `regionwise` gives the region of the
# bar its own conductivity (harmonic mean on interface faces).

wall, bar = load("thermal_bridge_3d_iso10211.step"), load(
    "thermal_bridge_3d_iso10211_bar.step"
)
faces = wall.boundaries
material = replace(
    IsotropicSolid(
        density=500.0,
        specific_heat_capacity=1700.0,
        thermal_conductivity=0.1,
        youngs_modulus=1e9,
        poisson_ratio=0.3,
        thermal_expansion_coefficient=0.0,
    ),
    thermal_conductivity=regionwise(
        ((bar, 50.0),), default=0.1, unit="W/m/K", interfaces="harmonic"
    ),
)

# ## Mesh
#
# A graded hexahedral mesh of the bounding box is cut to the wall with `region_mesh`; it has a node
# plane on every surface of the bar and of the layer, and the cells shrink towards them.

corner, (x_end, y_tip, z_end) = wall.bounding_box
x_bar, _, z_bar = bar.bounding_box[1]
t_insulation = next(face.center[1] for face in wall.faces if face.id == "y_max")
hull = Box(
    length=x_end - corner[0], width=y_tip, height=z_end - corner[2], origin=corner
)


def graded_mesh(level):
    """Hexahedral mesh of the wall for a refinement level."""
    cells, first = (lambda base: int(round(base * level))), 0.008 / level
    grading = (
        Piecewise(
            (
                (x_bar - corner[0], Geometric(cells(2), first=first, toward="end")),
                (x_end - x_bar, Geometric(cells(5), first=first)),
            )
        ),
        Piecewise(
            (
                (t_insulation / 2, Geometric(cells(3), first=first)),
                (t_insulation / 2, Geometric(cells(3), first=first, toward="end")),
                (y_tip - t_insulation, Geometric(cells(3), first=first)),
            )
        ),
        Piecewise(
            (
                (z_bar - corner[2], Geometric(cells(1), first=first, toward="end")),
                (z_end - z_bar, Geometric(cells(5), first=first)),
            )
        ),
    )
    return region_mesh(
        StructuredHexahedralMesher(
            cells=tuple(g.cells for g in grading), grading=grading
        ).generate(hull),
        wall,
    )


meshes = {level: graded_mesh(level) for level in (1, 2, 3, 4)}

view = plots.Camera(elevation=18, azimuth=40)
fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    wall,
    selection={
        "external": faces.external,
        "internal": faces.internal,
        "adiabatic": faces.adiabatic,
    },
    labels=False,
    camera=view,
    title="Quarter of the layer with the bar",
    ax=fig[0],
)
plots.mesh(
    meshes[2],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{meshes[2].cell_count} hexahedra (level 2)",
    ax=fig[1],
)
plots.show()

# ## Model and solution
#
# The same problem is solved with two spatial methods on three meshes each.

model = HeatConduction(domain=wall.volume, material=material, heating=0.0)
problem = Problem(
    name="ISO 10211 case 4",
    geometry=wall,
    physics=(model,),
    boundary_conditions=(
        ConvectiveHeatFlux(
            boundary=faces.internal,
            physics=model,
            heat_transfer_coefficient=10.0,
            ambient_temperature=274.15,
        ),
        ConvectiveHeatFlux(
            boundary=faces.external,
            physics=model,
            heat_transfer_coefficient=10.0,
            ambient_temperature=273.15,
        ),
        HeatFlux(boundary=faces.adiabatic, physics=model, value=0.0),
    ),
)
methods = {
    "finite volumes": (
        FiniteVolume(gradient=LeastSquaresGradient(), diffusion=CentralDifference()),
        (2, 3, 4),
    ),
    "finite elements (Q2)": (FiniteElement(element=Lagrange(order=2)), (1, 2, 3)),
}
runs = {
    name: [
        (
            Simulation(
                problem=problem,
                mesh=meshes[level],
                numerics=Numerics(spatial=spatial, linear_solver=SparseDirect()),
            )
            .solve_stationary()
            .field(model.temperature),
            meshes[level],
        )
        for level in levels
    ]
    for name, (spatial, levels) in methods.items()
}
field = runs["finite elements (Q2)"][-1][0]

# ## Results
#
# Heat flow through the full wall (four quarters) and the highest external surface temperature.

heat_flow = {
    name: [4 * f.boundary_flux(faces.external) for f, _ in results]
    for name, results in runs.items()
}
warmest = {
    name: [
        float(np.max(f.sample(along=faces.external, subdivisions=4).values)) - 273.15
        for f, _ in results
    ]
    for name, results in runs.items()
}
print(
    "method                 cells   heat flow [W]  (ref. 0.540)   max. external T [°C]  (ref. 0.805)"
)
for name, results in runs.items():
    for (_, mesh), flow, T in zip(results, heat_flow[name], warmest[name]):
        print(
            f"{name:20s} {mesh.cell_count:7d} {flow:12.4f}  ({100 * (flow / 0.540 - 1):+.2f} %) {T:18.4f}  ({T - 0.805:+.4f} K)"
        )
inflow = -4 * field.boundary_flux(faces.internal)
print(
    f"energy balance (Q2, finest mesh): in {inflow:.6f} W, out {heat_flow['finite elements (Q2)'][-1]:.6f} W"
)

# ## Figure
#
# The surface temperature seen from the warm side, the section through the bar axis, the external
# surface around the bar end, and the mesh study of the heat flow for both methods.

celsius = lambda T, position: T - 273.15
mirror = {"planes": [((1, 0, 0), (0.5, 0, 0)), ((0, 0, 1), (0, 0, 0.5))]}
fig = plots.figure(2, 2, size=(15, 11))
plots.surface(
    field,
    component=celsius,
    symmetry=mirror,
    camera=view,
    limits=(0.0, 1.0),
    label="T [°C]",
    title="Warm side: surface temperature",
    ax=fig[0],
)
plots.section(
    field,
    plane=Plane(origin=(0.5, 0.0, 0.5), normal=(0, 0, 1)),
    component=celsius,
    symmetry={"planes": [((1, 0, 0), (0.5, 0, 0))]},
    limits=(0.0, 1.0),
    contours=19,
    label="T [°C]",
    title="Section through the bar axis",
    xlimits=(0.15, 0.85),
    ylimits=(0, 0.62),
    ylabel="y [m] (cold face at y = 0)",
    ax=fig[1],
)
plots.surface(
    field,
    boundary=faces.external,
    component=celsius,
    symmetry=mirror,
    contours=12,
    subdivisions=4,
    label="T [°C]",
    title="External surface around the bar end",
    xlimits=(0.25, 0.75),
    ylimits=(0.35, 0.65),
    ax=fig[2],
)
plots.line(
    y={
        name: (
            [mesh.cell_count for _, mesh in results],
            [100 * (flow / 0.540 - 1) for flow in heat_flow[name]],
        )
        for name, results in runs.items()
    },
    hlines={"tolerance of the standard ±1 %": 1.0, "": -1.0},
    markers=True,
    scale="logx",
    xlabel="hexahedral cells (quarter model)",
    ylabel="heat flow: deviation from 0.540 W [%]",
    title="Mesh study",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Replace the iron by stainless steel (15 W/(m K)), or shorten the bar in
# `thermal_bridge_3d_iso10211_geometry.py` and write the files again: the heat flow through the
# bridge falls and the warm spot on the outside fades. The difference between the heat flow and
# the one-dimensional value $U A\,\Delta T$ of the bare layer is the point thermal transmittance
# of the bridge.
