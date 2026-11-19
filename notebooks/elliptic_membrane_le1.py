"""OneSimulation example: elliptic membrane under outward pressure (NAFEMS benchmark LE1).
Requires the installed, activated OneSimulation runtime.

A thin plate shaped like an elliptic ring is pulled outwards by a uniform pressure on its outer
edge. The plate is read from a STEP file and cut to its mid-plane; the plane-stress stress at the
benchmark point is compared with the published NAFEMS target on linear and quadratic triangles
of three sizes.
"""

# # Elliptic membrane under outward pressure (NAFEMS LE1)
#
# The membrane is a thin elliptic ring: the inner edge is the ellipse $(x/2)^2+(y/1)^2=1$, the
# outer edge the ellipse $(x/3.25)^2+(y/2.75)^2=1$ (metres), the thickness is 0.1 m. A uniform
# outward pressure of 10 MPa acts on the outer edge. Because of symmetry only the quarter
# $x\ge0,\ y\ge0$ is modelled, in plane stress.
#
# The quarter comes from the CAD file `elliptic_membrane_le1.step`, which names its faces; the
# script `elliptic_membrane_le1_geometry.py` builds it with the CAD objects of OneSimulation.
#
# | boundary | condition |
# |---|---|
# | $x=0$ | $u_x=0$ |
# | $y=0$ | $u_y=0$ |
# | `outer` | normal traction $+10$ MPa (outward) |
# | `inner` | traction free |
#
# Steel with $E=210$ GPa and $\nu=0.3$.
#
# **Reference.** NAFEMS, *The Standard NAFEMS Benchmarks*, rev. 3 (1990), test LE1: the hoop
# stress at point $D=(2,0)$, at the end of the long axis of the inner ellipse, is
# $\sigma_{yy}=92.7$ MPa.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TriangularMesher
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the quarter with its named faces and `section` the mid-plane region, whose sides
# keep the names of the faces they were cut from. The mesher follows the two ellipses with short
# boundary segments; three meshes with interior cells of 0.2, 0.1 and 0.05 m grade away from them.

membrane = load("elliptic_membrane_le1.step")
faces = membrane.boundaries
inner, outer, thickness = (2.0, 1.0), (3.25, 2.75), 0.1
region = membrane.section((0, 0, 1), origin=(0, 0, thickness / 2))
sides = region.boundaries
sizes = [0.2, 0.1, 0.05]
meshes = [
    TriangularMesher(
        target_size=size, grading=0.25, local_sizes={"inner": size / 3}
    ).generate(region)
    for size in sizes
]

fig = plots.figure(1, 2, size=(14, 5.5))
plots.geometry(
    membrane,
    selection={
        "outer (pressure)": faces.outer,
        "inner": faces.inner,
        "symmetry": faces.symmetry_x | faces.symmetry_y,
    },
    labels=False,
    camera=plots.Camera(elevation=60, azimuth=-70),
    title="Quarter membrane with its named faces",
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
# The outward pressure is a traction along the normal of the outer ellipse, $p\,\mathbf n$ with
# $\mathbf n\propto(x/a^2,\,y/b^2)$. The same model is solved with linear and with quadratic
# triangles on the three meshes.

pressure = 10e6
steel = SmallStrainSolid(
    youngs_modulus=210e9,
    poisson_ratio=0.3,
    kinematics="plane_stress",
    thickness=thickness,
)
outward = (
    lambda x, y: pressure
    * np.stack((x / outer[0] ** 2, y / outer[1] ** 2), axis=-1)
    / np.hypot(x / outer[0] ** 2, y / outer[1] ** 2)[..., None]
)
runs = {
    (order, size): steel.solve(
        mesh=mesh,
        times=(0.0,),
        order=order,
        tractions=((sides.outer, outward),),
        displacements=(
            (sides["part.x_min"], "x", 0.0),
            (sides["part.y_min"], "y", 0.0),
        ),
    )
    for order in (1, 2)
    for size, mesh in zip(sizes, meshes)
}

# ## Results
#
# The hoop stress follows from Hooke's law for plane stress with the gradient of the displacement
# field, evaluated exactly at point D.

hoop = (
    lambda result, points: 210e9
    / (1 - 0.3**2)
    * (lambda g: g[:, 1, 1] + 0.3 * g[:, 0, 0])(
        result.space.gradient(result.displacement[0], np.atleast_2d(points))
    )
)
target = 92.7
at_d = {key: hoop(result, (inner[0], 0.0))[0] / 1e6 for key, result in runs.items()}
print(
    "elements             cell size [m]   triangles   σ_yy at D [MPa]   deviation from 92.7 MPa"
)
for (order, size), value in at_d.items():
    print(
        f"{['linear', 'quadratic'][order - 1]:20s} {size:13.3f} {meshes[sizes.index(size)].cell_count:11d} "
        f"{value:17.2f} {100 * (value / target - 1):+14.2f} %"
    )

# ## Figure
#
# The hoop stress in the quarter on the finest quadratic mesh, the hoop stress along the symmetry
# line $y=0$ from D to the outer edge, and the mesh study at D for both element orders. Linear
# triangles carry a constant stress per cell, so their value at D is that of the cell at D and
# jumps with the mesh; quadratic triangles hit the target within 0.05 % on every mesh.

mesh, result = meshes[-1], runs[2, sizes[-1]]
x = np.linspace(inner[0], outer[0], 120)
fig = plots.figure(1, 3, size=(18, 5.5))
plots.heatmap(
    mesh.coordinates[:, 0],
    mesh.coordinates[:, 1],
    hoop(result, mesh.coordinates) / 1e6,
    triangles=np.asarray(mesh.cells),
    limits=(-10, 95),
    label="σ_yy [MPa]",
    title="Hoop stress σ_yy",
    ax=fig[0],
)
plots.line(
    points={
        "linear triangles": (
            x[::4],
            hoop(runs[1, sizes[-1]], np.column_stack((x, 0 * x)))[::4] / 1e6,
        )
    },
    y={"quadratic triangles": (x, hoop(result, np.column_stack((x, 0 * x))) / 1e6)},
    hlines={"NAFEMS target at D": target},
    xlabel="x on the line y = 0 [m]",
    ylabel="σ_yy [MPa]",
    title="Hoop stress from D to the outer edge",
    ax=fig[1],
)
plots.line(
    [mesh.cell_count for mesh in meshes],
    {
        f"{name} triangles": [at_d[order, size] for size in sizes]
        for order, name in ((1, "linear"), (2, "quadratic"))
    },
    hlines={"NAFEMS target 92.7 MPa": target},
    markers=True,
    xlabel="triangles",
    ylabel="σ_yy at D [MPa]",
    title="Mesh study at point D",
    legend="lower right",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Change the half axes in `elliptic_membrane_le1_geometry.py` and write the STEP file again, or
# compare the plane-stress answer with the thick-plate variant of the benchmark (LE10), which
# has the same outline and bends under a pressure on its face.
