"""OneSimulation example: the Scordelis-Lo roof, a cylindrical shell under its own weight.
Requires the installed, activated OneSimulation runtime.

A thin cylindrical roof rests on two end diaphragms and sags under its own weight. The roof is read
from a STEP file as a thin solid, reduced to its mid-surface and meshed with quadrilateral shell
elements; the vertical displacement at the middle of the free edge is compared with the reference
value of the benchmark on three meshes.
"""

# # Scordelis-Lo roof
#
# The roof is a segment of a circular cylinder: radius $R=25$ m, length $L=50$ m, spanning
# $40°$ on either side of the crown, thickness $t=0.25$ m. It rests at its two curved ends on rigid
# diaphragms that hold it in their plane but let it move along the axis and rotate freely; the
# two straight edges are free. The load is the dead weight, 90 per unit area of the mid-surface,
# acting downwards. The material has $E=4.32\cdot10^8$ and $\nu=0$ (consistent units).
#
# | boundary | condition |
# |---|---|
# | curved ends $x=\pm L/2$ (`diaphragm`) | $u_y=u_z=0$ |
# | straight edges (`free_edge`) | free |
# | one point on the crown at mid-length | $u_x=0$ (removes the rigid motion along the axis) |
#
# The roof comes from the CAD file `scordelis_lo_roof_shell.step` as a thin solid with the named
# faces `diaphragm` and `free_edge`; the script `scordelis_lo_roof_shell_geometry.py` builds it
# with the CAD objects of OneSimulation. `midsurface` turns the thin solid into the sheet halfway
# through its wall, which carries the thickness for the shell model.
#
# **Reference.** A. C. Scordelis and K. S. Lo, *Computer analysis of cylindrical shells*,
# J. American Concrete Institute 61 (1964) 539; R. H. MacNeal and R. L. Harder, *A proposed
# standard set of problems to test finite element accuracy*, Finite Elements in Analysis and
# Design 1 (1985) 3: the vertical displacement at the middle of the free edge is 0.3024 (deep
# shell theory).

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import SheetMesher
from onesim.physics.solid_mechanics import ElasticShell, ShellSurface
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# The mid-surface is meshed with paved quadrilaterals of three sizes.

roof = load("scordelis_lo_roof_shell.step")
faces = roof.boundaries
sheet = roof.midsurface()
radius, length, half_angle = 25.0, 50.0, np.radians(40.0)
sizes = [4.0, 2.0, 1.0]
meshes = [SheetMesher(target_size=size, paved=True).generate(sheet) for size in sizes]

view = plots.Camera(elevation=30, azimuth=-60)
fig = plots.figure(1, 2, size=(15, 5.5))
plots.geometry(
    roof,
    selection={"diaphragm": faces.diaphragm, "free edge": faces.free_edge},
    labels=False,
    camera=view,
    title=f"Roof, {sheet.thickness:.2f} m thick, with its named faces",
    ax=fig[0],
)
plots.mesh(
    meshes[1],
    camera=view,
    title=f"Mid-surface mesh: {len(meshes[1].quadrilaterals)} quadrilaterals",
    ax=fig[1],
)
plots.show()

# ## Model
#
# `ElasticShell` is a shear-deformable shell with five degrees of freedom per node, three
# displacements and two rotations about the local tangent axes. The supports are given node by
# node: the nodes on the two diaphragm ends and the node on the crown at mid-length.

shell = ElasticShell(
    youngs_modulus=4.32e8, poisson_ratio=0.0, thickness=sheet.thickness
)
results = []
for mesh in meshes:
    nodes = np.asarray(mesh.coordinates)
    ends = np.flatnonzero(np.isclose(np.abs(nodes[:, 0]), length / 2, atol=1e-6))
    crown = int(np.argmin(np.linalg.norm(nodes - (0.0, 0.0, radius), axis=1)))
    results.append(
        shell.solve(
            surface=ShellSurface(
                coordinates=nodes, quadrilaterals=np.asarray(mesh.quadrilaterals)
            ),
            supports=[(int(node), axis, 0.0) for node in ends for axis in ("y", "z")]
            + [(crown, "x", 0.0)],
            surface_load=(0.0, 0.0, -90.0),
        )
    )

# ## Results
#
# The vertical displacement at the middle of the free edge, the point
# $(0,\ R\sin40°,\ R\cos40°)$, against the reference value, and the balance of the reactions
# with the weight of the roof.

reference = 0.3024
edge = (0.0, radius * np.sin(half_angle), radius * np.cos(half_angle))
sags = [
    -result.displacement[
        np.argmin(np.linalg.norm(np.asarray(mesh.coordinates) - edge, axis=1)), 2
    ]
    for mesh, result in zip(meshes, results)
]
print(
    "cell size   quadrilaterals   nodes   displacement at the free edge   deviation from 0.3024"
)
for size, mesh, sag in zip(sizes, meshes, sags):
    print(
        f"{size:9.1f} {len(mesh.quadrilaterals):16d} {len(mesh.coordinates):7d} {sag:31.4f} {100 * (sag / reference - 1):+19.2f} %"
    )
print(
    f"vertical reactions {results[-1].reaction_force[:, 2].sum():.1f}, weight of the roof {90.0 * results[-1].area:.1f}"
)

# ## Figure
#
# The vertical displacement on the deformed roof (magnified 10 times), the sag along the free
# edge and near the crown, and the mesh study at the reference point.

mesh, result = meshes[-1], results[-1]
nodes = np.asarray(mesh.coordinates)
on_edge = np.flatnonzero(np.isclose(nodes[:, 1], edge[1], atol=1e-3))
on_crown = np.flatnonzero(np.abs(nodes[:, 1]) < 0.5)
fig = plots.figure(1, 3, size=(19, 5.5))
plots.mesh(
    mesh,
    values=result.displacement[:, 2],
    displacement=result.displacement,
    scale=10.0,
    camera=view,
    label="u_z",
    title="Vertical displacement, magnified 10 times",
    ax=fig[0],
)
plots.line(
    points={
        "free edge": (nodes[on_edge, 0], result.displacement[on_edge, 2]),
        "near the crown": (nodes[on_crown, 0], result.displacement[on_crown, 2]),
    },
    hlines={"reference at the middle of the free edge": -reference},
    xlabel="x along the roof",
    ylabel="u_z",
    title="Sag along the roof",
    legend="center",
    ax=fig[1],
)
plots.line(
    [len(mesh.coordinates) for mesh in meshes],
    {"simulation": sags},
    markers=True,
    hlines={"reference 0.3024": reference},
    scale="logx",
    ylimits=(0.28, 0.32),
    xlabel="nodes",
    ylabel="vertical displacement at the free edge",
    title="Mesh study",
    legend="lower right",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Change the opening angle or the thickness in `scordelis_lo_roof_shell_geometry.py` and write
# the STEP file again: a thinner roof makes locking visible in less robust shell elements. Add a
# layered `layer_stiffness` for a composite roof, or replace the dead weight by a snow load on
# the projected area.
