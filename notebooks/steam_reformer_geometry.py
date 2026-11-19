"""OneSimulation example: the geometry of a tubular steam reformer.
Requires the installed, activated OneSimulation runtime.

A steam reformer of nineteen catalyst tubes in a cylindrical shell is built with the CAD objects of
OneSimulation: the tubes are cylinders placed on two rings around a centre tube by circular
patterns, the shell space is the shell less the tubes. Catalyst beds and shell space form an
assembly with one volume per part, meshed as one.

Volumes and the heat-transfer area between shell space and tubes are compared with closed forms.
"""

# # Steam reformer geometry
#
# In a tubular steam reformer methane and steam react in tubes filled with catalyst while hot flue
# gas flows around the tubes in the shell and supplies the heat of the endothermic reaction. A model
# needs two kinds of domains: the catalyst beds inside the tubes and the shell space around them,
# with the tube walls as the faces through which the heat passes.
#
# | part | built from | count |
# |---|---|---|
# | `tube_0` ... `tube_18` | `Cylinder`, 24 mm diameter, 0.4 m long; centre tube, ring of 6 at 60 mm, ring of 12 at 115 mm | 19 |
# | `shell` | `Cylinder` of 300 mm diameter and 0.5 m height less all tubes, with 50 mm plenums above and below the tubes | 1 |
#
# The positions of a ring come from rotating one tube about the axis (`rotated`); the shell space
# is the shell cylinder with every tube subtracted.
#
# **Reference.** With shell radius $R_s$ and height $H_s = H + 2p$, tube radius $r$, length $H$ and
# $N$ tubes: $V_\text{tubes} = N\pi r^2 H$, $V_\text{shell} = \pi R_s^2 H_s - N\pi r^2 H$, the
# heat-transfer area over walls and tube ends is $A = N(2\pi r H + 2\pi r^2)$, and a cross-section
# through the tubes has $\pi(R_s^2 - N r^2)$.

from functools import reduce

import numpy as np
from onesim.cad.d3 import Assembly, Cylinder
from onesim.meshing import TetrahedralMesher
from onesim.meshing.quality import quality
from onesim.visualization import plots, theme

theme.set()

# ## Tubes and shell
#
# A tube on the x axis at the ring radius, turned to its place on the ring.

shell_radius, tube_radius, height, plenum = 0.15, 0.012, 0.4, 0.05
rings = ((0.0, 1), (0.06, 6), (0.115, 12))
tubes = [
    Cylinder(radius=tube_radius, length=height, origin=(ring, 0.0, plenum)).rotated(
        (0, 0, 1), 2 * np.pi * k / count + np.pi / count
    )
    for ring, count in rings
    for k in range(count)
]
tubes = [tube.named(f"tube_{index}") for index, tube in enumerate(tubes)]
shell = reduce(
    lambda body, tube: body - tube,
    tubes,
    Cylinder(radius=shell_radius, length=height + 2 * plenum),
).named("shell")
reformer = Assembly(parts=[shell, *tubes])
print(f"{len(tubes)} tubes; shell space: {len(shell.faces)} faces")

mesh = TetrahedralMesher(
    target_size=0.03, curvature_angle=0.5, max_candidate_cells=400_000
).generate(reformer)
view = plots.Camera(elevation=30, azimuth=-60)
fig = plots.figure(1, 2, size=(16, 8))
plots.render(
    tubes,
    camera=view,
    material="steel",
    shadows=False,
    title="Steam reformer: the 19 catalyst tubes",
    ax=fig[0],
)
plots.mesh(
    mesh,
    cell_values=quality(mesh)["shape"],
    limits=(0, 1),
    plane=((0.0, 0.0, plenum + height / 2), (1.0, 0.0, 0.0)),
    label="shape measure",
    camera=plots.Camera(elevation=10, azimuth=0),
    title=f"Cut through three tubes: {mesh.cell_count} cells",
    ax=fig[1],
)
plots.show()

# ## Results
#
# Volumes of the shell space and the tubes in the CAD model and the mesh, the contact area between
# them, and the area of an end face of the shell space.

count = len(tubes)
cells = quality(mesh)["size"]
meshed = {
    part.id: cells[list(mesh.region_cells(reformer.volume_of(part.id)))].sum()
    for part in reformer.volumes
}
print(
    f"shell space: CAD {shell.measure * 1e3:.5f} l, mesh {meshed['shell'] * 1e3:.5f} l, "
    f"closed form {(np.pi * shell_radius**2 * (height + 2 * plenum) - count * np.pi * tube_radius**2 * height) * 1e3:.5f} l"
)
print(
    f"tubes:       CAD {sum(tube.measure for tube in tubes) * 1e3:.5f} l, mesh {sum(meshed[tube.name] for tube in tubes) * 1e3:.5f} l, "
    f"closed form {count * np.pi * tube_radius**2 * height * 1e3:.5f} l"
)
print(
    f"heat-transfer area: {sum(contact.area for contact in reformer.interfaces):.6f} m2 in {len(reformer.interfaces)} contacts, "
    f"closed form N (2 pi r H + 2 pi r^2) {count * (2 * np.pi * tube_radius * height + 2 * np.pi * tube_radius**2):.6f} m2"
)
section = shell.section((0, 0, 1), (0.0, 0.0, plenum + height / 2))
print(
    f"cross-section of the shell space: {section.measure * 1e4:.4f} cm2, closed form {np.pi * (shell_radius**2 - count * tube_radius**2) * 1e4:.4f} cm2"
)

# ## Figure
#
# The cross-section of the shell space with the nineteen tube openings, and the mesh cut across the
# tubes at half height.

fig = plots.figure(1, 2, size=(16, 7))
plots.geometry(
    section,
    selection={"shell space": section.boundaries.all},
    labels=False,
    title="Cross-section at half height",
    ax=fig[0],
)
plots.mesh(
    mesh,
    cell_values=quality(mesh)["shape"],
    limits=(0, 1),
    plane=((0.0, 0.0, plenum + height / 2), (0.0, 0.0, 1.0)),
    label="shape measure",
    camera=plots.Camera(elevation=60, azimuth=-60),
    title="Mesh cut at half height",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Add baffles as thin discs subtracted from the shell, give the tube walls their own thickness with
# `AnnularCylinder`, or solve the heat flow from the hot shell space into the catalyst beds.
